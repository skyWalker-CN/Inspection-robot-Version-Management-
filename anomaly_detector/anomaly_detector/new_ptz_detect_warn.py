#!/usr/bin/env python3
"""
智能巡检机器人异常检测节点（CPU性能优化版 + Image话题发布 + 云台控制）

========================================
本文件相对 new_ptz_detect_warn.py 的改动（报警截图带检测框版）
========================================
问题：发布到 /anomaly_detection/alerts 的 screenshot（以及 history_anomaly/ 下的
     截图）原来是直接把原始帧落盘，不带任何检测框，看图时难以定位异常位置。
改法：截图先在帧上绘制"检测框 + 标签"，再落盘 / 转 base64 发布。
- 新增 _ScreenshotAnnotator：把检测分辨率下的 bbox 按比例映射回原图分辨率，
  绘制按异常类型着色的检测框、ASCII 标签（FIRE / SMOKE / NO HELMET ...）、
  突出当前报警目标的角度标记，并在左上角写入时间戳。
- 同一帧产生多条报警时只绘制一次标注底图（缓存复用），且仅在真正需要落盘时才绘制，
  保持原版"报警截图同步保存、频率低、CPU 影响小"的优化思路。
- 新增节点参数：screenshot_annotate(默认 True) / screenshot_annotate_related /
  screenshot_timestamp / screenshot_jpeg_quality；置 screenshot_annotate:=false
  即可完整回退到旧行为。
- 检测、报警冷却、空气质量/热像仪融合、PTZ、话题发布等其余逻辑与原版保持一致。
========================================

优化特性：
- 弃用 PIL 中文绘制，采用 OpenCV 纯英文绘制，消除整图格式转换
- 车牌检测降频至 1.0 秒，大幅降低 HyperLPR CPU 占用
- 推理间隔限制 (infer_interval) 默认 0.05s，平滑 CPU 负载
- 人脸检测异步化 + 可调检测尺寸和间隔
- 报警截图同步保存（频率低，影响小）
- 移除 WebRTC 推流，改为发布原始 Image 话题，降低 CPU 编码开销
- 所有检测结果直接在检测分辨率画面上绘制
- 支持通过 rqt_image_view 或 image_tools 查看处理后的画面
- 新增：云台自动跟踪与变焦（PTZ），优先级：火焰 > 烟雾 > 未知人员 > 未知车牌
- 新增：火焰强制停止机器人，其他类型不停止
- 新增：追踪冷却机制（火焰60秒，其他30秒）
- 新增：居中尝试5次失败后复位预设1+1倍变焦
- 新增：目标丢失宽限期1.5秒后判定失败
- 新增：未戴头盔报警时，若检测到对应人脸为已知人员，则报警信息中带上人员名称
"""

import sys
import os
import json
import math
import time
import threading
import logging
import base64
import random
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append('/home/jetson/anaconda3/envs/test/lib/python3.10/site-packages/')

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import Twist          # 新增：用于控制机器人停止
from cv_bridge import CvBridge

try:
    from ament_index_python.packages import get_package_share_directory
    AMENT_INDEX_AVAILABLE = True
except ImportError:
    AMENT_INDEX_AVAILABLE = False

# ============================================================
# Model imports
# ============================================================
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from hyperlpr3 import LicensePlateCatcher
    HYPERLPR_AVAILABLE = True
except ImportError:
    HYPERLPR_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

# ============================================================
# Bounding box container
# ============================================================
class _BoundingBox:
    __slots__ = ("x1", "y1", "x2", "y2", "cls_id", "conf")
    def __init__(self, x1, y1, x2, y2, cls_id, conf):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls_id = cls_id
        self.conf = conf

# ============================================================
# Fire & Smoke Detector (完整实现)
# ============================================================
class _FireSmokeMiner:
    class_names = ["smoke", "fire"]
    _model_class_order = ["smoke", "fire"]
    iou_thres = 0.45
    cross_iou_thresh = 0.8
    max_det = 150
    _conf_thres_array = np.array([0.6, 0.6], dtype=np.float32)
    _bonus_array = np.array([0.1, 0.1], dtype=np.float32)
    min_box_area = 14 * 14
    min_side = 8
    max_aspect_ratio = 8.0
    smoke_merge_overlap = 0.8
    fire_color_filter_max_conf = 0.45
    color_filter_min_saturation = 0.06

    def __init__(self, model_dir, device="cuda", class_names_override=None):
        self._backend = None
        self._yolo_model = None
        self._onnx_session = None
        self._device = device
        self._class_names_override = class_names_override
        engine_path = model_dir / "fire_smoke_best.engine"
        onnx_path = model_dir / "fire_smoke_best.onnx"
        pt_path = model_dir / "fire_smoke_best.pt"
        if engine_path.exists() and YOLO_AVAILABLE:
            self._yolo_model = YOLO(str(engine_path), task="detect")
            if device == "cuda":
                try:
                    self._yolo_model.to("cuda")
                except Exception:
                    pass
            self._backend = "yolo"
            self._setup_yolo_remap()
            print(f"[FireSmoke] Backend: YOLO TensorRT engine ({engine_path.name})")
        elif onnx_path.exists() and ONNX_AVAILABLE:
            self._setup_onnx(str(onnx_path), device)
        elif pt_path.exists() and YOLO_AVAILABLE:
            self._yolo_model = YOLO(str(pt_path))
            if device == "cuda":
                try:
                    self._yolo_model.to("cuda")
                except Exception:
                    pass
            self._backend = "yolo"
            self._setup_yolo_remap()
            print(f"[FireSmoke] Backend: YOLO PyTorch ({pt_path.name})")
        else:
            raise FileNotFoundError(f"No fire/smoke model found in {model_dir}")

    def _setup_yolo_remap(self):
        model_names = self._yolo_model.names
        num_model_classes = len(model_names)
        if self._class_names_override is not None:
            override = self._class_names_override
            if len(override) != num_model_classes:
                while len(override) < num_model_classes:
                    override.append("smoke")
                override = override[:num_model_classes]
            remap = [self.class_names.index(n) if n in self.class_names else 0 for n in override]
            self.cls_remap = np.array(remap, dtype=np.int32)
            return
        remap = []
        unknown_count = 0
        for idx in sorted(model_names):
            name = str(model_names[idx]).strip().lower()
            if name in self.class_names:
                remap.append(self.class_names.index(name))
            else:
                unknown_count += 1
                remap.append(-1)
        if unknown_count == num_model_classes:
            fallback_order = list(self.class_names)[:num_model_classes]
            remap = [self.class_names.index(n) if n in self.class_names else 0 for n in fallback_order]
        elif unknown_count > 0:
            for i in range(len(remap)):
                if remap[i] == -1:
                    guess = self._model_class_order[i] if i < len(self._model_class_order) else "smoke"
                    remap[i] = self.class_names.index(guess) if guess in self.class_names else 0
        self.cls_remap = np.array(remap, dtype=np.int32)

    def _setup_onnx(self, model_path, device):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 4
        available = ort.get_available_providers()
        if device == "cuda":
            if "TensorrtExecutionProvider" in available:
                providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self._onnx_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.session = self._onnx_session
        model_order = self._read_model_class_order()
        if model_order is None:
            model_order = list(self._model_class_order)
        self.cls_remap = np.array([self.class_names.index(n) for n in model_order], dtype=np.int32)
        self.input_name = self._onnx_session.get_inputs()[0].name
        self.output_names = [o.name for o in self._onnx_session.get_outputs()]
        self.input_shape = self._onnx_session.get_inputs()[0].shape
        self.input_height = self.input_shape[2] if isinstance(self.input_shape[2], int) and self.input_shape[2] > 0 else 1280
        self.input_width = self.input_shape[3] if isinstance(self.input_shape[3], int) and self.input_shape[3] > 0 else 1280
        self._backend = "onnx"

    def _read_model_class_order(self):
        try:
            import ast
            meta = self.session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(meta["names"])
            if isinstance(names, dict):
                order = [str(names[i]) for i in sorted(names)]
            else:
                order = [str(n) for n in names]
            if sorted(order) == sorted(self.class_names):
                return order
        except Exception:
            pass
        return None

    def _letterbox(self, image, new_shape, color=(114,114,114)):
        h, w = image.shape[:2]
        new_w, new_h = new_shape
        ratio = min(new_w / w, new_h / h)
        rw, rh = int(round(w * ratio)), int(round(h * ratio))
        if (rw, rh) != (w, h):
            image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        dw = (new_w - rw) / 2.0
        dh = (new_h - rh) / 2.0
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        return cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=color), ratio, (dw, dh)

    def _preprocess(self, image):
        orig_h, orig_w = image.shape[:2]
        img, ratio, pad = self._letterbox(image, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(img, dtype=np.float32), ratio, pad, (orig_w, orig_h)

    @staticmethod
    def _clip_boxes(boxes, image_size):
        w, h = image_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _xywh_to_xyxy(boxes):
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    @staticmethod
    def _hard_nms(boxes, scores, iou_thresh):
        n = len(boxes)
        if n == 0:
            return np.array([], dtype=np.intp)
        order = np.argsort(-scores)
        keep = []
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a_i = max(0.0, boxes[i, 2] - boxes[i, 0]) * max(0.0, boxes[i, 3] - boxes[i, 1])
            a_r = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (a_i + a_r - inter + 1e-7)
            order = rest[iou <= iou_thresh]
        return np.array(keep, dtype=np.intp)

    def _per_class_nms(self, boxes, scores, cls_ids, iou_thresh):
        if len(boxes) == 0:
            return np.array([], dtype=np.intp)
        all_keep = []
        for c in np.unique(cls_ids):
            mask = cls_ids == c
            indices = np.where(mask)[0]
            keep = self._hard_nms(boxes[mask], scores[mask], iou_thresh)
            all_keep.extend(indices[keep].tolist())
        all_keep.sort()
        return np.array(all_keep, dtype=np.intp)

    @staticmethod
    def _cross_class_dedup(boxes, scores, cls_ids, iou_thresh):
        n = len(boxes)
        if n <= 1:
            return boxes, scores, cls_ids
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        cls_ids = np.asarray(cls_ids, dtype=np.int32)
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        margins = scores - np.array([0.6 if c == 0 else 0.6 for c in cls_ids], dtype=np.float32)
        order = np.lexsort((-areas, -margins))
        suppressed = np.zeros(n, dtype=bool)
        keep = []
        for i in order:
            if suppressed[i]:
                continue
            keep.append(int(i))
            bi = boxes[i]
            xx1 = np.maximum(bi[0], boxes[:, 0])
            yy1 = np.maximum(bi[1], boxes[:, 1])
            xx2 = np.minimum(bi[2], boxes[:, 2])
            yy2 = np.minimum(bi[3], boxes[:, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a_i = max(1e-7, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
            iou = inter / (a_i + areas - inter + 1e-7)
            dup = iou > iou_thresh
            dup[i] = False
            suppressed |= dup
        k = np.array(keep, dtype=np.intp)
        return boxes[k], scores[k], cls_ids[k]

    def _merge_smoke(self, boxes, scores, cls_ids):
        smoke_cls = self.class_names.index("smoke")
        si = np.where(cls_ids == smoke_cls)[0]
        if len(si) <= 1:
            return boxes, scores, cls_ids
        sb = boxes[si].astype(np.float32).tolist()
        ss = scores[si].astype(np.float32).tolist()
        merged = True
        while merged and len(sb) > 1:
            merged = False
            for i in range(len(sb)):
                for j in range(i+1, len(sb)):
                    a, b = sb[i], sb[j]
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    if inter / (min(area_a, area_b) + 1e-7) >= self.smoke_merge_overlap:
                        sb[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        ss[i] = max(ss[i], ss[j])
                        del sb[j]; del ss[j]
                        merged = True
                        break
                if merged:
                    break
        other = cls_ids != smoke_cls
        nb = np.concatenate([boxes[other].astype(np.float32), np.array(sb, dtype=np.float32).reshape(-1, 4)])
        ns = np.concatenate([scores[other].astype(np.float32), np.array(ss, dtype=np.float32)])
        nc = np.concatenate([cls_ids[other].astype(np.int32), np.full(len(sb), smoke_cls, dtype=np.int32)])
        return nb, ns, nc

    def _conf_filter_mask(self, scores, cls_ids):
        if len(scores) == 0:
            return np.zeros(0, dtype=bool)
        thr = self._conf_thres_array[cls_ids]
        keep = scores >= thr
        for c in np.unique(cls_ids):
            b = float(self._bonus_array[c])
            if b <= 0.0:
                continue
            cm = cls_ids == c
            if keep[cm].any():
                continue
            idx = np.where(cm)[0]
            top = int(idx[int(np.argmax(scores[idx]))])
            if scores[top] >= self._conf_thres_array[c] - b:
                keep[top] = True
        return keep

    def _filter_sane(self, boxes, scores, cls_ids, orig_size):
        if len(boxes) == 0:
            return boxes, scores, cls_ids
        orig_w, orig_h = orig_size
        image_area = float(orig_w * orig_h)
        keep = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0 or bw < self.min_side or bh < self.min_side:
                continue
            area = bw * bh
            if area < self.min_box_area or area > 0.95 * image_area:
                continue
            ar = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
            if ar > self.max_aspect_ratio:
                continue
            keep.append(i)
        if not keep:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)
        k = np.array(keep, dtype=np.intp)
        return boxes[k], scores[k], cls_ids[k]

    def _per_view_pipeline(self, boxes, scores, cls_ids):
        if len(boxes) > 1:
            keep = self._per_class_nms(boxes, scores, cls_ids, self.iou_thres)
            boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(scores) > self.max_det:
            top = np.argsort(-scores)[:self.max_det]
            boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]
        if len(boxes) > 1:
            boxes, scores, cls_ids = self._cross_class_dedup(boxes, scores, cls_ids, self.cross_iou_thresh)
        if len(boxes) > 1:
            boxes, scores, cls_ids = self._merge_smoke(boxes, scores, cls_ids)
        return boxes, scores, cls_ids

    @staticmethod
    def _roi_for_box(image, box):
        h, w = image.shape[:2]
        x1 = max(0, int(math.floor(box.x1)))
        y1 = max(0, int(math.floor(box.y1)))
        x2 = min(w, int(math.ceil(box.x2)))
        y2 = min(h, int(math.ceil(box.y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or roi.ndim < 3:
            return None
        return roi

    def _roi_is_grayscale(self, roi):
        if roi.shape[0] < 2 or roi.shape[1] < 2:
            return False
        try:
            mx = roi.max(axis=2).astype(np.float32)
            mn = roi.min(axis=2).astype(np.float32)
            if mx.size == 0 or mn.size == 0:
                return False
            return float(((mx - mn) / (mx + 1e-6)).mean()) < self.color_filter_min_saturation
        except Exception:
            return False

    @staticmethod
    def _passes_fire_color(roi):
        blue = roi[:, :, 0].astype(np.float32)
        green = roi[:, :, 1].astype(np.float32)
        red = roi[:, :, 2].astype(np.float32)
        mean_r = float(np.mean(red))
        max_rgb = float(max(np.max(red), np.max(green), np.max(blue)))
        bright_frac = float(np.mean(np.max(roi, axis=2) >= 150))
        if max_rgb >= 200.0 and bright_frac >= 0.01:
            return True
        warm = (red > green + 10.0) & (red > blue + 10.0)
        warm_frac = float(np.mean(warm))
        r_minus_g = mean_r - float(np.mean(green))
        if warm_frac >= 0.05 and (max_rgb >= 120.0 or mean_r >= 120.0 or warm_frac >= 0.15):
            return True
        if bright_frac >= 0.12 and r_minus_g >= 2.0:
            return True
        return False

    def _filter_color(self, image, results):
        if not results:
            return results
        cls_fire = self.class_names.index("fire")
        out = []
        for box in results:
            try:
                cf = box.cls_id == cls_fire and box.conf <= self.fire_color_filter_max_conf
                if not cf:
                    out.append(box)
                    continue
                roi = self._roi_for_box(image, box)
                if roi is None or self._roi_is_grayscale(roi):
                    out.append(box)
                    continue
                if not self._passes_fire_color(roi):
                    continue
                out.append(box)
            except Exception:
                out.append(box)
        return out

    @staticmethod
    def _build_results(boxes, scores, cls_ids):
        results = []
        for box, conf, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(_BoundingBox(int(math.floor(x1)), int(math.floor(y1)),
                                        int(math.ceil(x2)), int(math.ceil(y2)),
                                        int(cls_id), float(conf)))
        return results

    def _decode_final(self, preds, ratio, pad, orig_size):
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        boxes = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        cls_ids = self.cls_remap[preds[:, 5].astype(np.int32)]
        keep = self._conf_filter_mask(scores, cls_ids)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(boxes) == 0:
            return []
        pw, ph = pad
        boxes[:, [0, 2]] -= pw
        boxes[:, [1, 3]] -= ph
        boxes /= ratio
        boxes = self._clip_boxes(boxes, orig_size)
        boxes, scores, cls_ids = self._filter_sane(boxes, scores, cls_ids, orig_size)
        if len(boxes) == 0:
            return []
        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _decode_raw(self, preds, ratio, pad, orig_size):
        if preds.ndim != 3 or preds.shape[0] != 1:
            return []
        preds = preds[0]
        if preds.shape[0] <= 16 and preds.shape[1] > preds.shape[0]:
            preds = preds.T
        boxes_xywh = preds[:, :4].astype(np.float32)
        cls_part = preds[:, 4:].astype(np.float32)
        if cls_part.shape[1] == 1:
            scores = cls_part[:, 0]
            cls_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            cls_ids = np.argmax(cls_part, axis=1).astype(np.int32)
            scores = cls_part[np.arange(len(cls_part)), cls_ids]
        cls_ids = self.cls_remap[cls_ids]
        keep = self._conf_filter_mask(scores, cls_ids)
        boxes_xywh, scores, cls_ids = boxes_xywh[keep], scores[keep], cls_ids[keep]
        if len(boxes_xywh) == 0:
            return []
        boxes = self._xywh_to_xyxy(boxes_xywh)
        pw, ph = pad
        boxes[:, [0, 2]] -= pw
        boxes[:, [1, 3]] -= ph
        boxes /= ratio
        boxes = self._clip_boxes(boxes, orig_size)
        boxes, scores, cls_ids = self._filter_sane(boxes, scores, cls_ids, orig_size)
        if len(boxes) == 0:
            return []
        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _postprocess(self, output, ratio, pad, orig_size):
        if output.ndim == 2 and output.shape[1] >= 6:
            return self._decode_final(output, ratio, pad, orig_size)
        if output.ndim == 3 and output.shape[0] == 1 and output.shape[2] == 6:
            return self._decode_final(output, ratio, pad, orig_size)
        return self._decode_raw(output, ratio, pad, orig_size)

    def _predict_single(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        inp, ratio, pad, orig_size = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        return self._postprocess(outputs[0], ratio, pad, orig_size)

    def _predict_yolo(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        orig_h, orig_w = image.shape[:2]
        results = self._yolo_model(image, conf=0.30, iou=self.iou_thres, verbose=False)
        all_boxes, all_scores, all_cls = [], [], []
        for res in results:
            if len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = res.boxes.conf.cpu().numpy().astype(np.float32)
            cls_raw = res.boxes.cls.cpu().numpy().astype(np.int32)
            valid_mask = (cls_raw >= 0) & (cls_raw < len(self.cls_remap))
            if not valid_mask.all():
                cls_raw = cls_raw.copy()
                cls_raw[~valid_mask] = len(self.cls_remap) - 1
            cls_mapped = self.cls_remap[cls_raw]
            all_boxes.append(xyxy)
            all_scores.append(confs)
            all_cls.append(cls_mapped)
        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        cls_ids = np.concatenate(all_cls)
        keep = self._conf_filter_mask(scores, cls_ids)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(boxes) == 0:
            return []
        boxes = self._clip_boxes(boxes, (orig_w, orig_h))
        boxes, scores, cls_ids = self._filter_sane(boxes, scores, cls_ids, (orig_w, orig_h))
        if len(boxes) == 0:
            return []
        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def detect(self, image, enable_color_filter=False):
        try:
            if self._backend == "yolo":
                boxes = self._predict_yolo(image)
            else:
                boxes = self._predict_single(image)
            if enable_color_filter:
                boxes = self._filter_color(image, boxes)
        except Exception as e:
            print(f"[FireSmoke] error: {e}")
            boxes = []
        return boxes

    def warmup(self):
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            if self._backend == "yolo":
                self._predict_yolo(dummy)
            else:
                self._predict_single(dummy)
        except Exception:
            pass

# ============================================================
# YOLO model loader
# ============================================================
def _load_yolo_model(model_path, device="cuda", half=True):
    p = Path(model_path)
    engine_path = p.with_suffix(".engine")
    if engine_path.exists():
        model = YOLO(str(engine_path), task="detect")
        print(f"[YOLO] Loaded TensorRT engine: {engine_path.name}")
        return model, True
    if p.exists():
        model = YOLO(str(p))
        if device == "cuda":
            try:
                model.to("cuda")
            except Exception:
                pass
        print(f"[YOLO] Loaded PyTorch weights: {p.name}")
        return model, False
    raise FileNotFoundError(f"Model not found: {model_path}")

# ============================================================
# 相机视场角（1x 光学变倍，写死）
# ============================================================
CAMERA_HFOV = 55.0
CAMERA_VFOV = 33.0

# ============================================================
# PTZ 状态常量
# ============================================================
PTZ_IDLE = "IDLE"
PTZ_CENTERING = "CENTERING"
PTZ_ZOOMING = "ZOOMING"
PTZ_MULTI_ZOOM = "MULTI_ZOOM"
PTZ_SUCCESS = "SUCCESS"
PTZ_FAILED = "FAILED"

# 异常类型优先级（数字越小优先级越高）
PRIORITY_ORDER = ['fire', 'smoke', 'unknown_person', 'unknown_vehicle']

# ============================================================
# 报警截图标注（本文件新增）：截图上先画检测框 + 标签再发布
# ============================================================
# 标签使用 ASCII 文字：cv2.putText(Hershey 字体) 画不出中文，沿用原版
# "纯 OpenCV 英文绘制" 的策略；中文描述仍完整保留在报警 JSON 的 description 中。
ANOMALY_LABEL_EN = {
    'fire': 'FIRE',
    'smoke': 'SMOKE',
    'unknown_person': 'UNKNOWN PERSON',
    'unknown_vehicle': 'UNKNOWN VEHICLE',
    'no_safetyhat': 'NO HELMET',
    'cigarette': 'SMOKING',
    'high_pm25': 'PM2.5 HIGH',
    'high_pm10': 'PM1.0 HIGH',
    'high_co2': 'CO2 HIGH',
    'high_temp': 'TEMP HIGH',
}

# 按异常类型着色（BGR），使报警目标在截图上一眼可辨
ANOMALY_BOX_COLORS = {
    'fire': (0, 0, 255),              # 红
    'smoke': (0, 200, 255),           # 橙
    'unknown_person': (0, 165, 255),  # 深橙
    'unknown_vehicle': (255, 0, 255), # 紫
    'no_safetyhat': (255, 255, 0),    # 青
    'cigarette': (255, 0, 0),         # 蓝
}
DEFAULT_BOX_COLOR = (0, 0, 255)
DEFAULT_LABEL_TEXT = 'ANOMALY'

_NON_PRINTABLE_RE = re.compile(r'[^ -~]')


def ascii_only(text):
    """仅保留可打印 ASCII 字符（车牌中的汉字等无法绘制，剔除后不影响可读性）"""
    if text is None:
        return ''
    return _NON_PRINTABLE_RE.sub('', str(text)).strip()


class _ScreenshotAnnotator:
    """在报警截图上绘制检测框与标签。

    用法（每条报警在落盘前调用一次 render）：
        annotator = _ScreenshotAnnotator(node, cv_image, anomalies, ratio=1.0 / scale)
        img = annotator.render(a)      # 返回带框副本，原始帧不会被改写

    - anomaly['bbox'] 是检测分辨率(small_frame)下的坐标，用 ratio 映射回截图分辨率；
    - 同一帧只绘制一次"标注底图"，各条报警在底图副本上再加粗突出自己的目标框，
      兼顾画面可读性与 CPU 开销（保持原版"截图同步保存、频率低"的优化思路）；
    - node.screenshot_annotate=False 时直接返回原帧，行为与改造前完全一致。
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    _CACHE_LIMIT = 4

    def __init__(self, node, frame, anomalies, ratio=1.0):
        self.node = node
        self.frame = frame
        self.anomalies = list(anomalies) if anomalies else []
        try:
            self.ratio = float(ratio)
        except (TypeError, ValueError):
            self.ratio = 1.0
        if not self.ratio or self.ratio <= 0 or math.isnan(self.ratio) or math.isinf(self.ratio):
            self.ratio = 1.0
        self._base = None
        self._cache = {}
        self._stamp = None

    # ---------------- 开关与时间戳 ----------------
    @property
    def enabled(self):
        return self.frame is not None and bool(getattr(self.node, 'screenshot_annotate', True))

    @property
    def stamp(self):
        if self._stamp is None:
            self._stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self._stamp

    # ---------------- 坐标/文字工具 ----------------
    def _scaled_bbox(self, bbox):
        """检测分辨率坐标 -> 截图坐标，并裁剪到画面内；无效则返回 None"""
        if self.frame is None or bbox is None or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) * self.ratio for v in list(bbox)[:4]]
        except (TypeError, ValueError):
            return None
        h, w = self.frame.shape[:2]
        x1 = max(0, min(w - 1, int(round(x1))))
        x2 = max(0, min(w - 1, int(round(x2))))
        y1 = max(0, min(h - 1, int(round(y1))))
        y2 = max(0, min(h - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _label(self, anomaly):
        atype = (anomaly or {}).get('type') or DEFAULT_LABEL_TEXT
        parts = [ANOMALY_LABEL_EN.get(atype) or ascii_only(atype).replace('_', ' ').upper() or DEFAULT_LABEL_TEXT]
        plate = ascii_only(anomaly.get('plate_number', '')) if anomaly.get('plate_number') else ''
        name = ascii_only(anomaly.get('person_name', '')) if anomaly.get('person_name') not in (None, '', 'unknown') else ''
        if plate:
            parts.append(plate)
        elif name:
            parts.append(name)
        conf = anomaly.get('confidence')
        if isinstance(conf, (int, float)) and conf > 0:
            parts.append("%.2f" % float(conf))
        return ' '.join(parts)[:64]

    @staticmethod
    def _text_color(color):
        b, g, r = color[:3]
        return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)

    # ---------------- 绘制 ----------------
    def _draw_label_bg(self, img, box, text, color, font_scale):
        thickness = max(1, int(round(font_scale * 1.8)))
        (tw, th), baseline = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        h, w = img.shape[:2]
        x1, y1, _, _ = box
        bg_h = th + baseline + 6
        top = y1 - bg_h                       # 默认画在框上方
        if top < 0:                           # 贴顶则画进框内，避免标签被裁掉
            top = min(max(0, y1 + 2), max(0, h - bg_h - 2))
        p1 = (min(x1, w - 1), top)
        p2 = (min(w - 1, x1 + tw + 6), min(h - 1, top + bg_h))
        cv2.rectangle(img, p1, p2, color, -1)
        cv2.putText(img, text, (p1[0] + 3, top + th + 2), self.FONT, font_scale,
                    self._text_color(color), thickness, cv2.LINE_AA)

    def _draw_corners(self, img, box, color, thickness):
        """在当前报警目标的四角画角度标记，进一步强调异常位置"""
        x1, y1, x2, y2 = box
        ln = max(8, int(min(x2 - x1, y2 - y1) * 0.25))
        t = max(2, thickness * 2)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * ln, cy), color, t, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * ln), color, t, cv2.LINE_AA)

    def _draw_box(self, img, anomaly, highlight=False):
        box = self._scaled_bbox((anomaly or {}).get('bbox'))
        if box is None:
            return None
        x1, y1, x2, y2 = box
        h = img.shape[0]
        color = ANOMALY_BOX_COLORS.get((anomaly or {}).get('type'), DEFAULT_BOX_COLOR)
        line = max(2, int(round(h / (260.0 if highlight else 420.0))))
        font_scale = max(0.45, min(1.6, h / 1300.0)) * (1.15 if highlight else 1.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line, cv2.LINE_AA)
        if highlight:
            self._draw_corners(img, box, color, line)
        self._draw_label_bg(img, box, self._label(anomaly), color, font_scale)
        return box

    def _draw_timestamp(self, img):
        h = img.shape[0]
        text = self.stamp
        font_scale = max(0.45, min(1.4, h / 1400.0))
        thickness = max(1, int(round(font_scale * 1.8)))
        (tw, th), baseline = cv2.getTextSize(text, self.FONT, font_scale, thickness)
        pad = max(2, int(th * 0.3))
        cv2.rectangle(img, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (0, 0, 0), -1)
        cv2.putText(img, text, (pad, pad + th), self.FONT, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    def _build_base(self):
        """本帧的标注底图：画出该帧全部异常框（同帧多条报警共用一份）"""
        img = self.frame.copy()
        if getattr(self.node, 'screenshot_annotate_related', True):
            for a in self.anomalies:
                self._draw_box(img, a, highlight=False)
        return img

    def _cache_key(self, anomaly):
        if anomaly is None:
            return None
        bbox = anomaly.get('bbox') or []
        return (anomaly.get('type'), tuple(int(v) for v in bbox[:4] if isinstance(v, (int, float))))

    def render(self, anomaly=None):
        """返回带检测框与标签的截图副本；关闭标注时退回原始帧（与改造前一致）"""
        if self.frame is None:
            return None
        if not self.enabled:
            return self.frame
        key = self._cache_key(anomaly)
        if key in self._cache:
            return self._cache[key]
        if self._base is None:
            self._base = self._build_base()
        img = self._base.copy()
        if anomaly is not None:
            self._draw_box(img, anomaly, highlight=True)
        if getattr(self.node, 'screenshot_timestamp', True):
            self._draw_timestamp(img)
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = img
        return img

    def encode_jpeg(self, anomaly=None, quality=90):
        """直接拿到带检测框的 jpg 字节（可用于不落盘的 base64 发布）"""
        img = self.render(anomaly)
        if img is None:
            return None
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None


# ============================================================
# ROS2 Node (无 WebRTC，发布 Image 话题 + PTZ)
# ============================================================
class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__("anomaly_detector_node")

        # ---- Parameters ----
        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("detect_width", 640)
        self.declare_parameter("detect_height", 320)
        self.declare_parameter("show_window", False)
        self.declare_parameter("display_scale", 1.5)
        self.declare_parameter("processed_image_topic", "/anomaly_detection/processed")
        self.declare_parameter("plate_detect_interval", 2.0)
        self.declare_parameter("infer_interval", 0.05)
        self.declare_parameter("enable_color_filter", False)

        # ---- 报警截图标注参数（本文件新增：截图带检测框与标签后再发布）----
        self.declare_parameter("screenshot_annotate", True)           # 截图上是否绘制检测框+标签
        self.declare_parameter("screenshot_annotate_related", True)   # 是否同时画出同帧其他异常（细框）
        self.declare_parameter("screenshot_timestamp", True)          # 截图左上角是否写入时间戳
        self.declare_parameter("screenshot_jpeg_quality", 90)         # 截图 JPEG 质量（影响 base64 体积）

        # ---- 统一阈值 ----
        self.declare_parameter("fire_conf_threshold", 0.8)
        self.declare_parameter("smoke_conf_threshold", 0.8)
        self.declare_parameter("fire_color_filter_max_conf", 0.45)
        self.declare_parameter("safety_conf_threshold", 0.8)
        self.declare_parameter("plate_conf_threshold", 0.65)
        self.declare_parameter("face_similarity_threshold", 0.50)

        # ---- 模型路径 ----
        models_dir = self._find_package_dir("models")
        face_db_dir = self._find_package_dir("face_database")

        self.declare_parameter("fire_smoke_model_dir", str(models_dir))
        self.declare_parameter("fire_smoke_class_names", [])
        self.declare_parameter("helmet_model_path", str(models_dir / "helmet_best.pt"))
        self.declare_parameter("smoking_model_path", str(models_dir / "smoking_best.pt"))
        self.declare_parameter("safety_combined_model_path", "")
        self.declare_parameter("face_database_dir", str(face_db_dir))
        self.declare_parameter("known_plates_file", str(models_dir / "known_plates.txt"))

        self.declare_parameter("enable_fire_smoke", True)
        self.declare_parameter("enable_safety", True)
        self.declare_parameter("enable_plate", True)
        self.declare_parameter("enable_face", True)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("half", True)
        self.declare_parameter("face_model_pack", "buffalo_l")
        self.declare_parameter("face_det_size", 320)
        self.declare_parameter("face_detection_interval", 0.2)

        # ---- 新增：PTZ 参数 ----
        self.declare_parameter("enable_ptz", False)                     # 是否启用云台控制
        self.declare_parameter("fire_cooldown_time", 60.0)              # 火焰冷却秒数
        self.declare_parameter("other_cooldown_time", 30.0)             # 其他异常冷却秒数
        self.declare_parameter("ptz_max_attempts", 5)                   # 居中最大尝试次数
        self.declare_parameter("ptz_lost_grace", 1.5)                  # 目标丢失宽限期（秒）
        self.declare_parameter("ptz_move_p_factor", 0.8)               # 移动比例因子

        # ---- 读取参数 ----
        self.image_topic = self.get_parameter("image_topic").value
        self.detect_width = self.get_parameter("detect_width").value
        self.detect_height = self.get_parameter("detect_height").value
        self.show_window = self.get_parameter("show_window").value
        self.display_scale = self.get_parameter("display_scale").value
        self.processed_image_topic = self.get_parameter("processed_image_topic").value
        self.plate_detect_interval = float(self.get_parameter("plate_detect_interval").value)
        self.infer_interval = float(self.get_parameter("infer_interval").value)
        self.enable_color_filter = self.get_parameter("enable_color_filter").value

        self.fire_conf_threshold = self.get_parameter("fire_conf_threshold").value
        self.smoke_conf_threshold = self.get_parameter("smoke_conf_threshold").value
        self.fire_color_filter_max_conf = self.get_parameter("fire_color_filter_max_conf").value
        self.safety_conf_threshold = self.get_parameter("safety_conf_threshold").value
        self.plate_conf_threshold = self.get_parameter("plate_conf_threshold").value
        self.face_similarity_threshold = self.get_parameter("face_similarity_threshold").value

        self.enable_fire_smoke = self.get_parameter("enable_fire_smoke").value
        self.enable_safety = self.get_parameter("enable_safety").value
        self.enable_plate = self.get_parameter("enable_plate").value
        self.enable_face = self.get_parameter("enable_face").value
        self.device = self.get_parameter("device").value
        self.half = self.get_parameter("half").value
        self.face_pack = self.get_parameter("face_model_pack").value
        self.face_det_size = self.get_parameter("face_det_size").value
        self.face_detection_interval = float(self.get_parameter("face_detection_interval").value)

        # ---- 读取 PTZ 参数 ----
        self.enable_ptz = self.get_parameter("enable_ptz").value
        self.fire_cooldown_time = float(self.get_parameter("fire_cooldown_time").value)
        self.other_cooldown_time = float(self.get_parameter("other_cooldown_time").value)
        self.ptz_max_attempts = int(self.get_parameter("ptz_max_attempts").value)
        self.ptz_lost_grace = float(self.get_parameter("ptz_lost_grace").value)
        self.ptz_move_p_factor = float(self.get_parameter("ptz_move_p_factor").value)

        # ---- 新增：空气质量与温度传感器融合参数 ----
        self.declare_parameter("pm1_0_threshold", 35.0)
        self.declare_parameter("pm2_5_threshold", 35.0)
        self.declare_parameter("co2_threshold", 1000.0)
        self.declare_parameter("temperature_threshold", 80.0)
        self.declare_parameter("air_check_timeout", 5.0)

        self.pm1_0_threshold = self.get_parameter("pm1_0_threshold").value
        self.pm2_5_threshold = self.get_parameter("pm2_5_threshold").value
        self.co2_threshold = self.get_parameter("co2_threshold").value
        self.temperature_threshold = self.get_parameter("temperature_threshold").value
        self.air_check_timeout = self.get_parameter("air_check_timeout").value

        # 空气质量和温度缓存
        self.latest_pm1_0 = 0.0
        self.latest_pm2_5 = 0.0
        self.latest_co2 = 0.0
        self.latest_air_time = 0.0
        self.latest_temp_max = 0.0
        self.latest_temp_time = 0.0

        # ---- Face database state ----
        self._face_db_names = []
        self._face_db_matrix = None
        self._face_executor = None
        self._face_future = None
        self._last_face_results = []
        self._last_plate_results = []
        self._last_plate_detect_time = 0.0
        self._label_font_size = 36

        # ---- 报警截图标注参数读取（本文件新增）----
        self.screenshot_annotate = bool(self.get_parameter("screenshot_annotate").value)
        self.screenshot_annotate_related = bool(self.get_parameter("screenshot_annotate_related").value)
        self.screenshot_timestamp = bool(self.get_parameter("screenshot_timestamp").value)
        try:
            self.screenshot_jpeg_quality = min(100, max(30, int(self.get_parameter("screenshot_jpeg_quality").value)))
        except (TypeError, ValueError):
            self.screenshot_jpeg_quality = 90

        # ---- Alert system ----
        self.alert_publisher = self.create_publisher(String, '/anomaly_detection/alerts', 10)
        self.last_alert_time = {}
        self.alert_cooldowns = {
            'fire': 10.0,
            'smoke': 10.0,
            'unknown_person': 5.0,
            'unknown_vehicle': 10.0,
            'no_safetyhat': 5.0,
            'cigarette': 5.0
        }
        self.screenshot_interval = 20.0
        self.last_screenshot_time = {}
        self.history_folder = "history_anomaly"
        os.makedirs(self.history_folder, exist_ok=True)

        # ---- 加载已知车牌 ----
        self.known_plates = self._load_known_plates()

        # ====== PTZ 相关变量 ======
        # 原始图像尺寸（用于云台控制）
        self.frame_shape = (1080, 1920)   # 默认，将在回调中更新

        # PTZ 状态
        self.ptz_state = PTZ_IDLE
        self.ptz_target_bbox = None
        self.ptz_target_type = None
        self.ptz_multi_target = False
        self.ptz_active_tracking_type = None
        self.ptz_current_zoom = 1.0
        self.ptz_lock = threading.Lock()
        self.ptz_attempts = 0
        self.ptz_target_lost_time = 0.0
        self.ptz_state_enter_time = time.time()

        # 冷却字典
        self.ptz_cooldowns = {}

        # 机器人运动状态
        self.robot_is_moving = False
        self.last_cmd_vel_time = 0.0
        self.is_stopping_robot = False
        self.last_stop_publish_time = 0.0
        self.stop_publish_rate = 0.05

        # ---- PTZ 发布者与订阅者 ----
        self.ptz_pub = self.create_publisher(String, '/controlCam', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # ---- 新增：空气质量传感器订阅 ----
        self.pm1_0_sub = self.create_subscription(Float32, "/air/pm1_0", self._pm1_0_callback, 10)
        self.pm2_5_sub = self.create_subscription(Float32, "/air/pm2_5", self._pm2_5_callback, 10)
        self.co2_sub = self.create_subscription(Float32, "/air/co2", self._co2_callback, 10)

        # ---- 新增：热像仪温度订阅 ----
        self.hotcam_sub = self.create_subscription(String, "hotCam", self._hotcam_callback, 10)

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Anomaly Detector Node (CPU Optimized + Image Pub + PTZ)")
        self.get_logger().info(f"  Device: {self.device}, FP16: {self.half}")
        self.get_logger().info(f"  Detect size: {self.detect_width}x{self.detect_height}")
        self.get_logger().info(f"  Face Detection Interval: {self.face_detection_interval}s (async)")
        self.get_logger().info(f"  Plate Detection Interval: {self.plate_detect_interval}s")
        self.get_logger().info(f"  Infer Interval: {self.infer_interval}s")
        self.get_logger().info(f"  Color Filter: {'ON' if self.enable_color_filter else 'OFF'}")
        self.get_logger().info(
            "  Alert Screenshot: "
            f"{'annotate ON (box+label)' if self.screenshot_annotate else 'annotate OFF (raw frame)'}, "
            f"related={'ON' if self.screenshot_annotate_related else 'OFF'}, "
            f"timestamp={'ON' if self.screenshot_timestamp else 'OFF'}, jpeg_q={self.screenshot_jpeg_quality}")
        self.get_logger().info(f"  Processed image topic: {self.processed_image_topic}")
        self.get_logger().info(f"  PTZ Enabled: {self.enable_ptz}")
        if self.enable_ptz:
            self.get_logger().info(f"    Fire cooldown: {self.fire_cooldown_time}s, Other: {self.other_cooldown_time}s")
            self.get_logger().info(f"    Max attempts: {self.ptz_max_attempts}, Lost grace: {self.ptz_lost_grace}s")
        self.get_logger().info("=" * 60)

        # ---- Init detectors ----
        self._init_detectors()

        # ---- 图像订阅 ----
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            RosImage, self.image_topic, self.image_callback, QoSProfile(depth=1))
        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")

        # ---- 处理后的图像发布者 ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.processed_pub = self.create_publisher(RosImage, self.processed_image_topic, 10)
        self.get_logger().info(f"Publishing processed images to: {self.processed_image_topic}")

        # ---- 窗口 ----
        if self.show_window:
            cv2.namedWindow("Anomaly Detection", cv2.WINDOW_NORMAL)
            init_w = int(self.detect_width * self.display_scale)
            init_h = int(self.detect_height * self.display_scale)
            cv2.resizeWindow("Anomaly Detection", init_w, init_h)

        # ---- Frame state ----
        self.latest_processed_frame = None
        self.processed_frame_lock = threading.Lock()
        self.frame_counter = 0
        self._last_face_detect_time = 0.0

        self._latest_frame_lock = threading.Lock()
        self._latest_cv_frame = None
        self._worker_stop = threading.Event()

        # ---- 推理线程 ----
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

        # ---- 启动 PTZ 控制线程（如果启用） ----
        if self.enable_ptz:
            self.ptz_thread = threading.Thread(target=self.ptz_control_loop, daemon=True)
            self.ptz_thread.start()
            self.get_logger().info("PTZ control thread started")

        self.get_logger().info("Node ready (Image Publishing Mode + PTZ)")

    # ------------------------------------------------------------------
    # Package directory resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _find_package_dir(subdir):
        if AMENT_INDEX_AVAILABLE:
            try:
                share = Path(get_package_share_directory("anomaly_detector"))
                candidate = share / subdir
                if candidate.is_dir():
                    return candidate
            except Exception:
                pass
        src_candidate = Path.home() / "ros2_ws" / "src" / "anomaly_detector" / subdir
        if src_candidate.is_dir():
            return src_candidate
        pkg_root = Path(__file__).resolve().parents[1]
        install_candidate = pkg_root / subdir
        if install_candidate.is_dir():
            return install_candidate
        return install_candidate

    # ------------------------------------------------------------------
    # Detector initialization
    # ------------------------------------------------------------------
    def _init_detectors(self):
        # 1. Fire & Smoke
        self._fs_miner = None
        if self.enable_fire_smoke and (ONNX_AVAILABLE or YOLO_AVAILABLE):
            try:
                model_dir = Path(self.get_parameter("fire_smoke_model_dir").value)
                fs_class_names = self.get_parameter("fire_smoke_class_names").value
                override = fs_class_names if fs_class_names else None
                self._fs_miner = _FireSmokeMiner(model_dir, device=self.device, class_names_override=override)
                self._fs_miner._conf_thres_array = np.array(
                    [self.smoke_conf_threshold, self.fire_conf_threshold], dtype=np.float32)
                self._fs_miner.fire_color_filter_max_conf = self.fire_color_filter_max_conf
                self._fs_miner.warmup()
                self.get_logger().info("[OK] Fire & Smoke detector loaded")
            except Exception as e:
                self.get_logger().error(f"Fire & Smoke init failed: {e}")

        # 2. Safety (默认禁用)
        self._safety_model = None
        self._safety_helmet = None
        self._safety_smoking = None
        self._safety_combined = False
        if self.enable_safety and YOLO_AVAILABLE:
            try:
                combined_path = self.get_parameter("safety_combined_model_path").value
                if combined_path and Path(combined_path).exists():
                    self._safety_model, _ = _load_yolo_model(combined_path, device=self.device, half=self.half)
                    self._safety_combined = True
                    self.get_logger().info("[OK] Combined safety model loaded")
                else:
                    hp = self.get_parameter("helmet_model_path").value
                    sp = self.get_parameter("smoking_model_path").value
                    self._safety_helmet, _ = _load_yolo_model(hp, device=self.device, half=self.half)
                    self._safety_smoking, _ = _load_yolo_model(sp, device=self.device, half=self.half)
                    self.get_logger().info("[OK] Safety (helmet+smoking) detectors loaded")
            except Exception as e:
                self.get_logger().error(f"Safety init failed: {e}")

        # 3. Plate
        self._plate_catcher = None
        if self.enable_plate and HYPERLPR_AVAILABLE:
            try:
                self._plate_catcher = LicensePlateCatcher()
                self.get_logger().info("[OK] Plate detector loaded")
            except Exception as e:
                self.get_logger().error(f"Plate init failed: {e}")

        # 4. Face (默认禁用)
        self._face_app = None
        self._face_use_gpu = False
        if self.enable_face and INSIGHTFACE_AVAILABLE:
            try:
                available = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    self._face_use_gpu = True
                elif 'TensorrtExecutionProvider' in available:
                    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    self._face_use_gpu = True
                else:
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1
                try:
                    self._face_app = FaceAnalysis(
                        name=self.face_pack, providers=providers,
                        allowed_modules=['detection', 'recognition'])
                except TypeError:
                    self._face_app = FaceAnalysis(name=self.face_pack, providers=providers)
                self._face_app.prepare(ctx_id=ctx_id, det_size=(self.face_det_size,) * 2)
                self._face_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face")
                device_str = "GPU (CUDA)" if self._face_use_gpu else "CPU"
                self.get_logger().info(f"[OK] Face detector loaded (pack={self.face_pack}, device={device_str})")
                self._init_face_database()
            except Exception as e:
                self.get_logger().error(f"Face init failed: {e}")

    def _init_face_database(self):
        db_dir = self.get_parameter("face_database_dir").value
        if not os.path.isdir(db_dir):
            self.get_logger().warn(f"Face database directory not found: {db_dir}")
            return
        embeddings = []
        for name in sorted(os.listdir(db_dir)):
            person_dir = os.path.join(db_dir, name)
            if not os.path.isdir(person_dir):
                continue
            person_path = Path(person_dir)
            feature_files = sorted(person_path.glob("feature_*.npy"))
            if not feature_files:
                legacy = os.path.join(person_dir, "feature.npy")
                if os.path.exists(legacy):
                    feature_files = [Path(legacy)]
            for feat_path in feature_files:
                try:
                    emb = np.load(str(feat_path)).astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    embeddings.append(emb)
                    self._face_db_names.append(name)
                except Exception as e:
                    self.get_logger().warn(f"Failed to load {feat_path.name}: {e}")
        if not embeddings:
            self.get_logger().warn("Face database is empty")
            return
        self._face_db_matrix = np.stack(embeddings).astype(np.float32)
        self.get_logger().info(f"Face database: {len(embeddings)} features from {len(set(self._face_db_names))} people")

    # ------------------------------------------------------------------
    # Known plates loader
    # ------------------------------------------------------------------
    _PLATE_RE = re.compile(r'[^\u4e00-\u9fa5A-Z0-9]')

    def _load_known_plates(self):
        plate_file = self.get_parameter("known_plates_file").value
        known = set()
        if not plate_file or not os.path.exists(plate_file):
            self.get_logger().warn(f"Known plates file not found: {plate_file}, all plates will be treated as unknown")
            return known
        try:
            with open(plate_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        known.add(line.upper())
            self.get_logger().info(f"Loaded {len(known)} known plates")
        except Exception as e:
            self.get_logger().error(f"Error loading known plates: {e}")
        return known

    # ------------------------------------------------------------------
    # cmd_vel callback (用于检测机器人是否运动)
    # ------------------------------------------------------------------
    def cmd_vel_callback(self, msg: Twist):
        self.robot_is_moving = (abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01)
        self.last_cmd_vel_time = time.time()

    def publish_stop_high_rate(self, now):
        if now - self.last_stop_publish_time >= self.stop_publish_rate:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            self.last_stop_publish_time = now

    # ------------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------------
    def image_callback(self, msg: RosImage):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 更新原始图像尺寸
            self.frame_shape = cv_image.shape[:2]
        except Exception as e:
            self.get_logger().warn(f"Bridge error: {e}")
            return

        self.frame_counter += 1
        with self._latest_frame_lock:
            self._latest_cv_frame = cv_image

    # ------------------------------------------------------------------
    # Inference worker
    # ------------------------------------------------------------------
    def _inference_loop(self):
        while not self._worker_stop.is_set():
            with self._latest_frame_lock:
                cv_image = self._latest_cv_frame
                self._latest_cv_frame = None
            if cv_image is None:
                time.sleep(0.002)
                continue
            t0 = time.time()
            try:
                self._process_frame(cv_image)
            except Exception as e:
                self.get_logger().warn(f"Process frame error: {e}")
            if self.infer_interval > 0.0:
                wait = self.infer_interval - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)

    def _process_frame(self, cv_image):
        h, w = cv_image.shape[:2]
        # 自适应缩放至检测尺寸
        if w > self.detect_width:
            scale = self.detect_width / w
            new_w = self.detect_width
            new_h = int(h * scale)
            small_frame = cv2.resize(cv_image, (new_w, new_h))
        else:
            scale = 1.0
            small_frame = cv_image

        draw_ratio = small_frame.shape[1] / float(w) if w > 0 else 1.0
        self._label_font_size = max(14, int(round(36 * draw_ratio)))

        anomalies, all_detections = self.detect_anomalies(small_frame)

        # ---- 如果启用 PTZ，将 bbox 缩放到原始图像坐标 ----
        orig_anomalies = []
        if self.enable_ptz:
            for a in anomalies:
                bbox = a.get('bbox', [])
                if len(bbox) == 4:
                    orig_bbox = [int(bbox[0] / scale), int(bbox[1] / scale),
                                 int(bbox[2] / scale), int(bbox[3] / scale)]
                    a_copy = a.copy()
                    a_copy['bbox'] = orig_bbox
                    orig_anomalies.append(a_copy)
            # 更新 PTZ 目标
            self.update_ptz_target(orig_anomalies)

        # 同步保存截图（频率低，影响小）
        # 截图用的是原始分辨率帧 cv_image，而 anomaly['bbox'] 是检测分辨率 small_frame
        # 下的坐标，故用 ratio = 1/scale 把框映射回原图后再绘制，保证框与目标对齐。
        ratio = (1.0 / scale) if scale else 1.0
        screenshot_annotator = (
            _ScreenshotAnnotator(self, cv_image, anomalies, ratio) if anomalies else None)
        for a in anomalies:
            img_path = self.save_anomaly_screenshot(cv_image, a['type'],
                                                    annotator=screenshot_annotator, anomaly=a)
            if img_path:
                a['image_path'] = img_path

        if anomalies:
            self.publish_alerts(anomalies, cv_image, annotator=screenshot_annotator)

        display_frame = small_frame.copy()
        display_frame = self.draw_detections(display_frame, anomalies, all_detections)

        # 显示信息
        info_scale = 1.5 * draw_ratio
        info_thick = max(1, int(round(3 * draw_ratio)))
        info_y1 = max(20, int(round(40 * draw_ratio)))
        info_y2 = max(40, int(round(80 * draw_ratio)))
        cv2.putText(display_frame, f"Frame: {self.frame_counter}", (10, info_y1),
                    cv2.FONT_HERSHEY_SIMPLEX, info_scale, (0, 255, 0), info_thick)
        cv2.putText(display_frame, f"Anomalies: {len(anomalies)}", (10, info_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, info_scale, (0, 255, 0), info_thick)

        # 添加 PTZ 状态信息
        if self.enable_ptz:
            with self.ptz_lock:
                state_str = f"PTZ: {self.ptz_state} att={self.ptz_attempts}/{self.ptz_max_attempts}"
            cv2.putText(display_frame, state_str, (10, info_y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, info_scale*0.8, (0, 255, 255), info_thick)

        # ---- 发布处理后的图像 ----
        ros_msg = self.bridge.cv2_to_imgmsg(display_frame, encoding="bgr8")
        self.processed_pub.publish(ros_msg)

        # ---- 更新内部缓存 ----
        self.update_processed_frame(display_frame)

        # ---- 窗口显示 ----
        if self.show_window:
            show_frame = cv2.resize(display_frame, (self.detect_width, self.detect_height))
            cv2.imshow('Anomaly Detection', show_frame)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Detection functions
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 空气质量传感器回调
    # ------------------------------------------------------------------
    def _pm1_0_callback(self, msg: Float32):
        self.latest_pm1_0 = msg.data
        self.latest_air_time = time.time()

    def _pm2_5_callback(self, msg: Float32):
        self.latest_pm2_5 = msg.data
        self.latest_air_time = time.time()

    def _co2_callback(self, msg: Float32):
        self.latest_co2 = msg.data
        self.latest_air_time = time.time()

    # ------------------------------------------------------------------
    # 热像仪温度回调
    # ------------------------------------------------------------------
    def _hotcam_callback(self, msg: String):
        """解析 hotCam 话题，格式: "{t_min:.1f} {t_max_real:.1f}" """
        try:
            parts = msg.data.split()
            if len(parts) >= 2:
                self.latest_temp_max = float(parts[1])
                self.latest_temp_time = time.time()
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------
    # 验证火焰/烟雾检测结果
    # ------------------------------------------------------------------
    def _verify_fire_smoke(self, anomaly_type):
        """
        用传感器数据验证检测结果，减少误报：
        - smoke → 检查 PM1.0/PM2.5/CO2 是否超标，确认后保持 smoke 类型
        - fire  → 检查热像仪温度是否 > 阈值，确认后保持 fire 类型
        返回 True 表示确认异常，False 表示误报
        """
        now = time.time()

        if anomaly_type == "smoke":
            if now - self.latest_air_time > self.air_check_timeout:
                self.get_logger().warn(f"空气质量数据过期 ({now - self.latest_air_time:.1f}s > {self.air_check_timeout}s)，跳过烟雾验证")
                return False
            pm1_ok = self.latest_pm1_0 >= self.pm1_0_threshold
            pm25_ok = self.latest_pm2_5 >= self.pm2_5_threshold
            co2_ok = self.latest_co2 >= self.co2_threshold
            if pm1_ok or pm25_ok or co2_ok:
                self.get_logger().info(
                    f"烟雾验证通过: PM1.0={self.latest_pm1_0}(>={self.pm1_0_threshold}?{pm1_ok}), "
                    f"PM2.5={self.latest_pm2_5}(>={self.pm2_5_threshold}?{pm25_ok}), "
                    f"CO2={self.latest_co2}(>={self.co2_threshold}?{co2_ok})")
                return True
            else:
                self.get_logger().info(
                    f"烟雾验证未通过(误报): PM1.0={self.latest_pm1_0}, PM2.5={self.latest_pm2_5}, CO2={self.latest_co2}")
                return False

        elif anomaly_type == "fire":
            if now - self.latest_temp_time > self.air_check_timeout:
                self.get_logger().warn(f"温度数据过期 ({now - self.latest_temp_time:.1f}s > {self.air_check_timeout}s)，跳过火焰验证")
                return False
            if self.latest_temp_max >= self.temperature_threshold:
                self.get_logger().info(f"火焰验证通过: 最高温={self.latest_temp_max}°C >= {self.temperature_threshold}°C")
                return True
            else:
                self.get_logger().info(f"火焰验证未通过(误报): 最高温={self.latest_temp_max}°C < {self.temperature_threshold}°C")
                return False

        return False

    def detect_anomalies(self, frame):
        anomalies = []
        all_detections = []
        now = time.time()

        # Fire & Smoke（增加传感器验证，减少误报）
        if self._fs_miner is not None:
            boxes = self._fs_miner.detect(frame, enable_color_filter=self.enable_color_filter)
            for b in boxes:
                anomaly_type = _FireSmokeMiner.class_names[b.cls_id]
                # 用传感器数据验证，通过后才绘制和报警
                if self._verify_fire_smoke(anomaly_type):
                    anomalies.append({
                        'type': anomaly_type,
                        'confidence': b.conf,
                        'bbox': [b.x1, b.y1, b.x2, b.y2],
                    })
                else:
                    self.get_logger().debug(f"传感器验证未通过，忽略 {anomaly_type} 检测结果")

        # Safety
        if self._safety_model is not None or self._safety_helmet is not None:
            safety_results = self.detect_safety(frame)
            ANOMALY_SAFETY_CLASSES = {'no_safetyhat', 'cigarette'}
            for det in safety_results:
                cls_name = det.get('class_name', '').lower()
                if cls_name in ANOMALY_SAFETY_CLASSES:
                    anomalies.append({
                        'type': cls_name,
                        'confidence': det['conf'],
                        'bbox': [det['x1'], det['y1'], det['x2'], det['y2']],
                        'description': f"安全违规: {cls_name}"
                    })
                else:
                    all_detections.append(det)

        # Plate
        if self._plate_catcher is not None:
            if (now - self._last_plate_detect_time) >= self.plate_detect_interval:
                self._last_plate_results = self.detect_plate(frame)
                self._last_plate_detect_time = now
            for plate_info in self._last_plate_results:
                plate_number = plate_info.get('plate_number', '')
                if plate_number and plate_number not in ("size_error", "no_result", "low_confidence", "ocr_error"):
                    clean_plate = self._normalize_plate(plate_number)
                    if clean_plate and clean_plate in self.known_plates:
                        all_detections.append(plate_info)
                    elif clean_plate:
                        anomalies.append({
                            'type': 'unknown_vehicle',
                            'confidence': plate_info.get('conf', 0.0),
                            'bbox': [plate_info['x1'], plate_info['y1'],
                                     plate_info['x2'], plate_info['y2']],
                            'plate_number': clean_plate,
                            'description': f"未知车辆: {clean_plate}"
                        })
                    else:
                        all_detections.append(plate_info)
                else:
                    all_detections.append(plate_info)

        # Face
        if self._face_app is not None:
            if self._face_future is not None and self._face_future.done():
                try:
                    res = self._face_future.result()
                    if res is not None:
                        self._last_face_results = res
                except Exception:
                    pass
                self._face_future = None
            if (self._face_future is None and
                    (now - self._last_face_detect_time) >= self.face_detection_interval):
                self._last_face_detect_time = now
                self._face_future = self._face_executor.submit(self.detect_face, frame.copy())
            for f in self._last_face_results:
                if f.get('name') == 'unknown':
                    anomalies.append({
                        'type': 'unknown_person',
                        'confidence': f['conf'],
                        'bbox': [f['x1'], f['y1'], f['x2'], f['y2']],
                        'person_name': 'unknown',
                    })
                else:
                    all_detections.append(f)

        return anomalies, all_detections

    def detect_safety(self, frame):
        results = []
        conf = self.safety_conf_threshold
        if self._safety_combined and self._safety_model is not None:
            try:
                r = self._safety_model(frame, conf=conf, verbose=False, device=self.device)
                for res in r:
                    for box in res.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        cls_name = self._safety_model.names.get(int(box.cls[0]), 'unknown')
                        results.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'conf': float(box.conf[0]),
                            'class_name': cls_name.lower(),
                            'source': 'safety'
                        })
            except Exception as e:
                self.get_logger().debug(f"Safety detect error: {e}")
        else:
            for model in (self._safety_helmet, self._safety_smoking):
                if model is None:
                    continue
                try:
                    r = model(frame, conf=conf, verbose=False, device=self.device)
                    for res in r:
                        for box in res.boxes:
                            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                            cls_name = model.names.get(int(box.cls[0]), 'unknown')
                            results.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'conf': float(box.conf[0]),
                                'class_name': cls_name.lower(),
                                'source': 'safety'
                            })
                except Exception as e:
                    self.get_logger().debug(f"Safety detect error: {e}")
        return results

    def detect_plate(self, frame):
        results = []
        try:
            plates = self._plate_catcher(frame)
            for plate in plates:
                plate_number = plate[0]
                score = float(plate[1])
                if score < self.plate_conf_threshold:
                    continue
                x1,y1,x2,y2 = plate[3]
                results.append({
                    'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                    'plate_number': plate_number,
                    'conf': score,
                    'source': 'plate'
                })
        except Exception as e:
            self.get_logger().debug(f"Plate detect error: {e}")
        return results

    def detect_face(self, frame):
        results = []
        try:
            faces = self._face_app.get(frame)
            max_faces = 3
            faces = sorted(faces, key=lambda x: x.det_score, reverse=True)[:max_faces]
            for face in faces:
                box = face.bbox.astype(int)
                x1,y1,x2,y2 = box.tolist()
                det_score = float(face.det_score)
                result = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': det_score,
                    'source': 'face',
                    'name': 'unknown',
                    'similarity': 0.0
                }
                if self._face_db_matrix is not None and hasattr(face, 'embedding') and face.embedding is not None:
                    emb = face.embedding.astype(np.float32)
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                    sims = self._face_db_matrix @ emb_norm
                    best_idx = np.argmax(sims)
                    best_sim = float(sims[best_idx])
                    result['similarity'] = best_sim
                    if best_sim >= self.face_similarity_threshold:
                        result['name'] = self._face_db_names[best_idx]
                results.append(result)
        except Exception as e:
            self.get_logger().debug(f"Face detect error: {e}")
        return results

    @staticmethod
    def _normalize_plate(plate_text):
        return AnomalyDetectorNode._PLATE_RE.sub('', plate_text.upper())

    # ------------------------------------------------------------------
    # Drawing (纯OpenCV)
    # ------------------------------------------------------------------
    def draw_detections(self, frame, anomalies, all_detections=None):
        drawn_boxes = set()
        font_scale = 0.5
        thickness = 1

        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                key = (x1, y1, x2, y2)
                if key in drawn_boxes:
                    continue
                drawn_boxes.add(key)
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if anomaly['type'] in ('no_safetyhat', 'cigarette'):
                    label = f"Violation:{anomaly['type']} {anomaly['confidence']:.2f}"
                elif anomaly['type'] == 'unknown_vehicle':
                    label = f"Unknown:{anomaly.get('plate_number','')}"
                else:
                    label = f"{anomaly.get('type','')} {anomaly.get('confidence',0):.2f}"
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)

        if all_detections:
            for det in all_detections:
                bbox = [det.get('x1'), det.get('y1'), det.get('x2'), det.get('y2')]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    key = (x1, y1, x2, y2)
                    if key in drawn_boxes:
                        continue
                    drawn_boxes.add(key)
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = det.get('class_name') or det.get('plate_number') or det.get('name') or 'detection'
                    if det.get('source') == 'plate' and 'plate_number' in det:
                        label = f"{det['plate_number']}" + (" (known)" if det['plate_number'] in self.known_plates else "")
                    cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

        return frame

    # ------------------------------------------------------------------
    # 辅助方法：计算 IoU
    # ------------------------------------------------------------------
    def _compute_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1b, y1b, x2b, y2b = bbox2
        inter_x1 = max(x1, x1b)
        inter_y1 = max(y1, y1b)
        inter_x2 = min(x2, x2b)
        inter_y2 = min(y2, y2b)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)
        return inter_area / (area1 + area2 - inter_area + 1e-6)

    # ------------------------------------------------------------------
    # Alert publishing (增加未戴头盔匹配已知人脸 + 截图带检测框)
    # ------------------------------------------------------------------
    def publish_alerts(self, anomalies, frame=None, annotator=None):
        """发布报警到 /anomaly_detection/alerts。

        annotator: _ScreenshotAnnotator 实例。若本帧缺截图需要现场补拍，
        用它保证补拍出来的 screenshot 同样带检测框与标签。
        """
        current_time = time.time()
        custom_descriptions = {
            'fire': '已检测到火焰，建议立即处理',
            'smoke': '检测到烟雾，可能存在火灾隐患',
            'unknown_person': '发现未登记人员，请前往核实',
            'unknown_vehicle': '发现未登记车辆，请前往查看',
            'no_safetyhat': '未佩戴安全帽，存在安全隐患',
            'cigarette': '检测到吸烟行为，请及时制止'
        }

        for a in anomalies:
            atype = a['type']
            if atype in self.alert_cooldowns:
                if current_time - self.last_alert_time.get(atype, 0) < self.alert_cooldowns[atype]:
                    continue
                self.last_alert_time[atype] = current_time
            else:
                if current_time - self.last_alert_time.get(atype, 0) < 5.0:
                    continue
                self.last_alert_time[atype] = current_time

            # 特殊处理 no_safetyhat: 尝试匹配已知人脸
            description = custom_descriptions.get(atype, a.get('description', ''))
            if atype == 'no_safetyhat':
                bbox = a.get('bbox', [])
                if len(bbox) == 4:
                    best_name = None
                    best_iou = 0.0
                    for face in self._last_face_results:
                        face_bbox = [face['x1'], face['y1'], face['x2'], face['y2']]
                        iou = self._compute_iou(bbox, face_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_name = face.get('name', 'unknown')
                    if best_iou > 0.3 and best_name and best_name != 'unknown':
                        description = f"{best_name}未佩戴安全帽"

            lat = round(random.uniform(39.9000, 39.9100), 6)
            lon = round(random.uniform(116.3000, 116.4000), 6)

            img_path = a.get('image_path', '')
            if frame is not None and not (img_path and os.path.exists(img_path)):
                # 报警已确定发布：缺截图时绕过节流补拍当帧（报警冷却≥5s，写入频率可控）
                # 传 annotator 使补拍的截图同样带检测框与标签
                img_path = self.save_anomaly_screenshot(frame, atype, force=True,
                                                       annotator=annotator, anomaly=a)
                if img_path:
                    a['image_path'] = img_path
            b64 = ""
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode()
                except Exception:
                    pass

            alert_data = {
                "type": atype,
                "confidence": a.get('confidence', 0.0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": description,
                "position": {"latitude": lat, "longitude": lon},
                "screenshot": b64
            }
            self.alert_publisher.publish(String(data=json.dumps(alert_data, ensure_ascii=False)))
            self.get_logger().info(f"Alert published: {atype} (conf={a.get('confidence',0):.2f}) - {description}")

    def save_anomaly_screenshot(self, frame, anomaly_type, force=False, annotator=None, anomaly=None):
        """保存异常报警截图。

        annotator 不为 None 时，落盘前会先把检测框与标签画到帧上（见 _ScreenshotAnnotator），
        这样 /anomaly_detection/alerts 里的 screenshot 能直接看出异常位置；
        传 None 则与改造前一致，直接落盘传入的帧。
        """
        current_time = time.time()
        if not force and anomaly_type in self.last_screenshot_time:
            if current_time - self.last_screenshot_time[anomaly_type] < self.screenshot_interval:
                return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.history_folder, f"{anomaly_type}_{ts}.jpg")
        # 仅在真正要写文件时才绘制，避免被节流拦下的帧白白消耗 CPU
        out_frame = annotator.render(anomaly) if annotator is not None else frame
        if out_frame is None:
            out_frame = frame
        try:
            ok = cv2.imwrite(path, out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.screenshot_jpeg_quality])
            if not ok:
                raise IOError("cv2.imwrite returned False")
        except Exception as e:
            # 写入失败也要更新时间戳，否则下一帧会立刻重试造成刷屏
            self.last_screenshot_time[anomaly_type] = current_time
            self.get_logger().warn(f"Screenshot save failed: {e}")
            return None
        self.last_screenshot_time[anomaly_type] = current_time
        return path

    # ------------------------------------------------------------------
    # Processed frame management
    # ------------------------------------------------------------------
    def update_processed_frame(self, frame):
        with self.processed_frame_lock:
            self.latest_processed_frame = frame

    def get_latest_processed_frame(self):
        with self.processed_frame_lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame
            return np.zeros((self.detect_height, self.detect_width, 3), dtype=np.uint8)

    # ==================================================================
    # 新增：PTZ 控制相关方法（完全仿照第二段代码）
    # ==================================================================

    # ---------- 目标选择 ----------
    def update_ptz_target(self, anomalies):
        """从当前帧的异常列表中，按照优先级选择目标"""
        now = time.time()
        by_type = {}
        for a in anomalies:
            by_type.setdefault(a['type'], []).append(a)

        selected_type = None
        selected_list = None
        for t in PRIORITY_ORDER:
            if t in by_type and by_type[t]:
                if t in self.ptz_cooldowns and now < self.ptz_cooldowns[t]:
                    continue
                selected_type = t
                selected_list = by_type[t]
                break

        with self.ptz_lock:
            # 处于追踪过程中（居中/变焦）
            if self.ptz_state in (PTZ_CENTERING, PTZ_ZOOMING, PTZ_MULTI_ZOOM) and self.ptz_active_tracking_type is not None:
                # 1. 检测是否出现更高优先级目标
                if selected_type is not None and selected_type != self.ptz_active_tracking_type:
                    current_prio = PRIORITY_ORDER.index(self.ptz_active_tracking_type) if self.ptz_active_tracking_type in PRIORITY_ORDER else 99
                    new_prio = PRIORITY_ORDER.index(selected_type) if selected_type in PRIORITY_ORDER else 99
                    if new_prio < current_prio:
                        self.get_logger().info(f"[PTZ] 发现更高优先级目标 {selected_type}，中断当前 {self.ptz_active_tracking_type} 追踪！")
                        self.ptz_state = PTZ_IDLE
                        self.ptz_attempts = 0
                        self.ptz_active_tracking_type = None
                        self.is_stopping_robot = False
                    else:
                        return
                # 2. 如果是同种目标
                if selected_type == self.ptz_active_tracking_type:
                    if self.ptz_state == PTZ_CENTERING:
                        if selected_list:
                            self.ptz_target_bbox = selected_list[0].get('bbox', None)
                        return
                    if self.ptz_state in (PTZ_ZOOMING, PTZ_MULTI_ZOOM):
                        return

            # 正常 IDLE 状态或被高优先级打断后的逻辑
            if selected_type is None:
                self.ptz_target_type = None
                return

            multi = len(selected_list) > 1
            self.ptz_target_type = selected_type
            self.ptz_multi_target = multi
            if multi:
                self.ptz_target_bbox = None
            else:
                self.ptz_target_bbox = selected_list[0].get('bbox', None)

    # ---------- PTZ 控制主循环 ----------
    def ptz_control_loop(self):
        self.get_logger().info("[PTZ] 控制循环启动")
        while not self._worker_stop.is_set() and rclpy.ok():
            try:
                now = time.time()
                with self.ptz_lock:
                    bbox = self.ptz_target_bbox
                    target_type = self.ptz_target_type
                    multi = self.ptz_multi_target

                # ===== 火焰强制停止机器人 =====
                if self.ptz_active_tracking_type == 'fire' and self.ptz_state != PTZ_IDLE:
                    self.is_stopping_robot = True
                    self.publish_stop_high_rate(now)
                else:
                    if self.is_stopping_robot and self.ptz_state == PTZ_IDLE:
                        self.is_stopping_robot = False

                # ===== 处理目标丢失（宽限期）=====
                if bbox is None and not multi:
                    if self.ptz_state in (PTZ_CENTERING, PTZ_ZOOMING):
                        if self.ptz_target_lost_time == 0:
                            self.ptz_target_lost_time = now
                            self.get_logger().info(f"[PTZ] 目标短暂丢失，进入 {self.ptz_lost_grace}s 宽限期")
                        elif now - self.ptz_target_lost_time > self.ptz_lost_grace:
                            self.get_logger().info("[PTZ] 宽限期超时，目标丢失 → 直接判定失败")
                            self.ptz_target_lost_time = 0
                            self._goto_failed()
                        time.sleep(0.15)
                        continue
                    elif self.ptz_state == PTZ_MULTI_ZOOM:
                        pass
                    elif self.ptz_state == PTZ_IDLE:
                        time.sleep(0.2)
                        continue
                else:
                    self.ptz_target_lost_time = 0

                # ===== 状态机 =====
                if self.ptz_state == PTZ_IDLE:
                    if target_type is None:
                        time.sleep(0.2)
                        continue

                    self.get_logger().info(f"[PTZ] 检测到目标: type={target_type}, multi={multi}, "
                                           f"robot_moving={self.robot_is_moving}")
                    self.ptz_attempts = 1
                    self.ptz_active_tracking_type = target_type

                    if multi:
                        self.get_logger().info("[PTZ] 多同类目标模式：直接放大7倍")
                        self.ptz_state = PTZ_MULTI_ZOOM
                        self.ptz_state_enter_time = now
                    else:
                        self.get_logger().info("[PTZ] 单目标模式：复位云台到预设1")
                        self.ptz_pub.publish(String(data="9 1"))
                        time.sleep(2.0)
                        self.ptz_current_zoom = 1.0
                        self.ptz_state = PTZ_CENTERING
                        self.ptz_state_enter_time = time.time()

                elif self.ptz_state == PTZ_CENTERING:
                    self._do_centering(bbox)

                elif self.ptz_state == PTZ_ZOOMING:
                    self._do_zooming(bbox)

                elif self.ptz_state == PTZ_MULTI_ZOOM:
                    self._do_multi_zoom()

                elif self.ptz_state == PTZ_SUCCESS:
                    self._do_reset("SUCCESS")

                elif self.ptz_state == PTZ_FAILED:
                    self._do_reset("FAILED")

                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f"[PTZ] 控制循环异常: {e}")
                time.sleep(0.3)

    # ---------- 居中 ----------
    def _do_centering(self, bbox):
        if bbox is None or len(bbox) != 4:
            return
        h, w = self.frame_shape
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        img_cx, img_cy = w // 2, h // 2
        dx = cx - img_cx
        dy = cy - img_cy
        current_hfov = CAMERA_HFOV / max(self.ptz_current_zoom, 1.0)
        current_vfov = CAMERA_VFOV / max(self.ptz_current_zoom, 1.0)

        pan_angle = dx / w * current_hfov
        tilt_angle = -dy / h * current_vfov

        self.get_logger().info(f"[PTZ] CENTERING attempt={self.ptz_attempts}/{self.ptz_max_attempts} "
                               f"原始偏差 pan={pan_angle:+.2f} tilt={tilt_angle:+.2f}")

        if abs(pan_angle) < 5.0 and abs(tilt_angle) < 5.0:
            self.get_logger().info("[PTZ] 已居中（偏差<5°），进入 ZOOMING")
            self.ptz_state = PTZ_ZOOMING
            self.ptz_state_enter_time = time.time()
            return

        if self.ptz_attempts <= self.ptz_max_attempts:
            move_pan = pan_angle * self.ptz_move_p_factor
            move_tilt = tilt_angle * self.ptz_move_p_factor

            # 防止移动过小
            if abs(move_pan) > 0.1 and abs(move_pan) < 1.0:
                move_pan = 1.0 if move_pan > 0 else -1.0
            if abs(move_tilt) > 0.1 and abs(move_tilt) < 1.0:
                move_tilt = 1.0 if move_tilt > 0 else -1.0

            self.get_logger().info(f"[PTZ] 发送移动指令 实际移动 pan={move_pan:+.2f} tilt={move_tilt:+.2f}，"
                                   f"消耗 1 次尝试 ({self.ptz_attempts}/{self.ptz_max_attempts})")
            self.send_move(move_pan, move_tilt)
            self.ptz_attempts += 1
            time.sleep(1.5)
        else:
            self.get_logger().info(f"[PTZ] {self.ptz_max_attempts}次尝试均失败 → FAILED")
            self._goto_failed()

    # ---------- 放大（单目标）----------
    def _do_zooming(self, bbox):
        if int(self.ptz_current_zoom) != 7:
            self.get_logger().info("[PTZ] 发送变焦指令 4 7（7倍）")
            self.ptz_pub.publish(String(data="4 7"))
            self.ptz_current_zoom = 7.0
            time.sleep(1.2)
        self.get_logger().info("[PTZ] 追踪至中心并放大7倍成功！→ SUCCESS")
        self.ptz_state = PTZ_SUCCESS
        self.ptz_state_enter_time = time.time()

    # ---------- 多目标放大 ----------
    def _do_multi_zoom(self):
        if int(self.ptz_current_zoom) != 7:
            self.get_logger().info("[PTZ] MULTI_ZOOM: 发送变焦指令 4 7")
            self.ptz_pub.publish(String(data="4 7"))
            self.ptz_current_zoom = 7.0
            time.sleep(1.5)
        self.get_logger().info("[PTZ] MULTI_ZOOM 完成 → SUCCESS")
        self.ptz_state = PTZ_SUCCESS
        self.ptz_state_enter_time = time.time()

    # ---------- 复位 ----------
    def _do_reset(self, reason):
        self.get_logger().info(f"[PTZ] {reason}: 复位到预设1（9 1）+ 1倍变焦（5 32）")
        if reason == "SUCCESS":
            time.sleep(1.5)
        self.ptz_pub.publish(String(data="9 1"))
        time.sleep(0.6)
        self.ptz_pub.publish(String(data="5 32"))
        self.ptz_current_zoom = 1.0
        time.sleep(1.5)

        tracked_type = self.ptz_active_tracking_type
        if tracked_type is not None:
            now = time.time()
            if tracked_type == 'fire':
                self.ptz_cooldowns[tracked_type] = now + self.fire_cooldown_time
                self.get_logger().info(f"[PTZ] {tracked_type} 进入冷却期 {self.fire_cooldown_time}s")
            else:
                self.ptz_cooldowns[tracked_type] = now + self.other_cooldown_time
                self.get_logger().info(f"[PTZ] {tracked_type} 进入冷却期 {self.other_cooldown_time}s")

        self.ptz_state = PTZ_IDLE
        self.ptz_attempts = 0
        self.ptz_target_lost_time = 0
        self.ptz_active_tracking_type = None
        self.is_stopping_robot = False
        self.get_logger().info("[PTZ] 状态切换: IDLE")

    def _goto_failed(self):
        self.ptz_state = PTZ_FAILED
        self.ptz_state_enter_time = time.time()

    # ---------- 发送移动指令 ----------
    def send_move(self, pan, tilt):
        if abs(pan) > 0.1:
            cmd = f"0 {pan:.2f}" if pan > 0 else f"1 {abs(pan):.2f}"
            self.ptz_pub.publish(String(data=cmd))
        if abs(tilt) > 0.1:
            cmd = f"2 {tilt:.2f}" if tilt > 0 else f"3 {abs(tilt):.2f}"
            self.ptz_pub.publish(String(data=cmd))

    # ==================================================================
    # 结束 PTZ 相关方法
    # ==================================================================

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self._worker_stop.set()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        if self.enable_ptz and hasattr(self, 'ptz_thread') and self.ptz_thread.is_alive():
            self.ptz_thread.join(timeout=3.0)
        if self._face_executor is not None:
            self._face_executor.shutdown(wait=False)
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()

# ============================================================
# Main
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
