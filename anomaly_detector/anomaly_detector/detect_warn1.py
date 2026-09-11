#!/usr/bin/env python3
"""
智能巡检机器人异常检测节点（性能优化版：异步推理 + 最新帧策略 + 批量中文绘制）

功能与原版完全一致：
- 检测目标：火焰、烟雾、车牌、人脸、抽烟行为、安全帽
- 异常（红色框）：火焰、烟雾、未知人员、未知车牌、未戴安全帽、吸烟
- 正常（绿色框）：已戴安全帽、已知人员、已知车牌
- 报警发布：仅对红色框异常发送警报（含冷却期 + 全分辨率截图）
- WebRTC 实时推流、中文标签 PIL 绘制

性能优化点（不改变检测语义与输出内容）：
1. 订阅回调只缓存最新帧，推理在独立工作线程执行：
   相机帧率高于推理速度时自动丢弃中间帧，不再堆积队列
   （消除延迟累积造成的画面卡顿与 CPU/GPU 持续满载）。
2. 人脸检测异步化：buffalo_l(CPU) 推理放到独立线程，不再每 5 秒阻塞主流水线
   （消除周期性卡顿）；仅加载 detection + recognition 模块
   （去掉 2D/3D 关键点、性别年龄模型，降低 CPU 占用）；检测结果缓存并持续绘制。
3. 车牌检测降频（plate_detect_interval 参数，默认 0.2s），结果缓存。
4. 中文标签由“每个检测框一次整帧 PIL 转换”改为“每帧一次批量转换”，并缓存字体。
   原先 N 个框 = N 次全图 BGR<->RGB 转换 + 整帧内存拷贝，是主要 CPU 消耗之一。
5. 检测框直接绘制在检测分辨率（960x540）画面上，字体按比例缩放，视觉效果一致；
   报警截图仍使用原始全分辨率图像。
6. 去除 processed frame 的双重整帧拷贝（update/get 各一次 .copy()）。
7. 图像订阅 QoS 队列深度 10 -> 1，避免 DDS 层帧积压。
8. 新增 infer_interval 参数（默认 0.0 = 全速）：如需进一步降低占用，
   可设为 0.05~0.1 限制每秒推理次数。
9. 修复 _face_db_names 未初始化导致人脸库加载失败、已知人员识别静默失效的 bug。
"""

import sys
import os
import json
import math
import time
import threading
import asyncio
import logging
import base64
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

sys.path.append('/home/jetson/anaconda3/envs/test/lib/python3.10/site-packages/')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
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

try:
    import websockets
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("[WARN] WebRTC dependencies not installed. Install: pip install aiortc av websockets")

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
    _conf_thres_array = np.array([0.6, 0.6], dtype=np.float32)  # 外部更新
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

    def detect(self, image):
        try:
            if self._backend == "yolo":
                boxes = self._predict_yolo(image)
            else:
                boxes = self._predict_single(image)
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
# WebRTC Video Track
# ============================================================
class ProcessedFrameTrack(VideoStreamTrack):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.frame_count = 0
        self._last_frame = None
        self.push_width = 480
        self.push_height = 270

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        try:
            frame_bgr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.node.thread_pool, self.node.get_latest_processed_frame
                ),
                timeout=0.1
            )
            self._last_frame = frame_bgr
        except Exception:
            frame_bgr = self._last_frame if self._last_frame is not None else np.zeros((540, 960, 3), dtype=np.uint8)
        frame_bgr = cv2.resize(frame_bgr, (self.push_width, self.push_height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        self.frame_count += 1
        return video_frame

# ============================================================
# 辅助函数：使用PIL绘制中文文本（字体缓存 + 批量绘制）
# ============================================================
_FONT_PATHS = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei-bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyhbd.ttf",
    "C:/Windows/Fonts/simhei.ttf"
]
_FONT_CACHE = {}
_FONT_LOCK = threading.Lock()


def _get_chinese_font(font_size):
    """加载并缓存指定字号的中文字体（避免每帧重复解析字体文件）"""
    font = _FONT_CACHE.get(font_size)
    if font is not None:
        return font
    with _FONT_LOCK:
        font = _FONT_CACHE.get(font_size)
        if font is not None:
            return font
        for fp in _FONT_PATHS:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[font_size] = font
        return font


def put_chinese_texts(img, items):
    """
    批量绘制中文文本：整帧只做一次 BGR->PIL->BGR 转换。
    :param img: OpenCV BGR图像
    :param items: [(text, (x, y), font_size, color(B,G,R), bg_color or None), ...]
    """
    if not items:
        return
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for text, position, font_size, color, bg_color in items:
            font = _get_chinese_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x, y = position
            if bg_color is not None:
                bg_rgb = (bg_color[2], bg_color[1], bg_color[0]) if len(bg_color) == 3 else bg_color
                draw.rectangle(
                    [(x - 2, y - text_height - 2), (x + text_width + 2, y + 2)],
                    fill=bg_rgb
                )
            rgb_color = (color[2], color[1], color[0]) if len(color) == 3 else color
            draw.text((x, y - text_height), text, font=font, fill=rgb_color)
        img[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # 回退到 cv2.putText (仅英文)
        for text, position, _fs, _c, _bg in items:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def put_chinese_text(img, text, position, font_size=24, color=(255, 255, 255), bg_color=None):
    """兼容原接口：单条文本绘制"""
    put_chinese_texts(img, [(text, position, font_size, color, bg_color)])

# ============================================================
# ROS2 Node
# ============================================================
class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__("anomaly_detector_node")

        # ---- Parameters ----
        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("detect_width", 960)
        self.declare_parameter("detect_height", 540)
        self.declare_parameter("push_width", 480)
        self.declare_parameter("push_height", 270)
        self.declare_parameter("show_window", False)
        self.declare_parameter("display_scale", 1.5)
        self.declare_parameter("enable_webrtc", True)
        self.declare_parameter("webrtc_port", 8080)
        self.declare_parameter("face_detection_interval", 0.2)
        # 新增性能参数
        self.declare_parameter("plate_detect_interval", 0.2)   # 车牌检测最小间隔(秒)，0=每帧
        self.declare_parameter("infer_interval", 0.0)          # 主流水线最小间隔(秒)，0=全速

        # ---- Unified confidence thresholds ----
        self.declare_parameter("fire_conf_threshold", 0.8)
        self.declare_parameter("smoke_conf_threshold", 0.8)
        self.declare_parameter("fire_color_filter_max_conf", 0.45)
        self.declare_parameter("safety_conf_threshold", 0.8)
        self.declare_parameter("plate_conf_threshold", 0.65)
        self.declare_parameter("face_similarity_threshold", 0.50)

        # ---- Models directories and whitelists ----
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
        self.declare_parameter("enable_safety", False)
        self.declare_parameter("enable_plate", True)
        self.declare_parameter("enable_face", False)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("half", True)
        self.declare_parameter("face_model_pack", "buffalo_l")
        self.declare_parameter("face_det_size", 320)

        # ---- Read params ----
        self.image_topic = self.get_parameter("image_topic").value
        self.detect_width = self.get_parameter("detect_width").value
        self.detect_height = self.get_parameter("detect_height").value
        self.push_width = self.get_parameter("push_width").value
        self.push_height = self.get_parameter("push_height").value
        self.show_window = self.get_parameter("show_window").value
        self.display_scale = self.get_parameter("display_scale").value
        self.enable_webrtc = self.get_parameter("enable_webrtc").value
        self.webrtc_port = self.get_parameter("webrtc_port").value
        self.face_detection_interval = self.get_parameter("face_detection_interval").value
        self.plate_detect_interval = float(self.get_parameter("plate_detect_interval").value)
        self.infer_interval = float(self.get_parameter("infer_interval").value)

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

        # ---- Face database state（修复：原版本未初始化导致人脸库加载失败）----
        self._face_db_names = []
        self._face_db_matrix = None
        # 人脸异步检测状态
        self._face_executor = None
        self._face_future = None
        self._last_face_results = []
        # 车牌降频缓存
        self._last_plate_results = []
        self._last_plate_detect_time = 0.0
        # 标签字号（按绘制分辨率自适应）
        self._label_font_size = 36

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

        # ---- Load known plates whitelist ----
        self.known_plates = self._load_known_plates()

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Anomaly Detector Node (Optimized: Async + Latest-Frame)")
        self.get_logger().info(f"  Device: {self.device}, FP16: {self.half}")
        self.get_logger().info(f"  Detect size: {self.detect_width}x{self.detect_height}")
        self.get_logger().info(f"  Face Detection Interval: {self.face_detection_interval}s (async)")
        self.get_logger().info(f"  Plate Detection Interval: {self.plate_detect_interval}s")
        self.get_logger().info(f"  Infer Interval: {self.infer_interval}s (0=full speed)")
        self.get_logger().info(f"  Confidence thresholds:")
        self.get_logger().info(f"    fire: {self.fire_conf_threshold}, smoke: {self.smoke_conf_threshold}")
        self.get_logger().info(f"    fire_color_filter_max: {self.fire_color_filter_max_conf}")
        self.get_logger().info(f"    safety: {self.safety_conf_threshold}")
        self.get_logger().info(f"    plate: {self.plate_conf_threshold}")
        self.get_logger().info(f"    face similarity: {self.face_similarity_threshold}")
        self.get_logger().info("=" * 60)

        # ---- Init detectors ----
        self._init_detectors()

        # ---- Image subscription（QoS 深度 1：只保留最新帧，避免积压）----
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            RosImage, self.image_topic, self.image_callback, QoSProfile(depth=1))
        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")

        # ---- Window ----
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

        # 最新帧缓存（订阅回调写入，推理线程读取）
        self._latest_frame_lock = threading.Lock()
        self._latest_cv_frame = None
        self._worker_stop = threading.Event()

        # ---- WebRTC server ----
        self.thread_pool = ThreadPoolExecutor(max_workers=2) if WEBRTC_AVAILABLE else None
        self.webrtc_loop = None
        self.webrtc_running = False
        if self.enable_webrtc and WEBRTC_AVAILABLE:
            self.webrtc_running = True
            self.webrtc_thread = threading.Thread(target=self._run_webrtc_server, daemon=True)
            self.webrtc_thread.start()
            self.get_logger().info(f"WebRTC server started on port {self.webrtc_port}")

        # ---- Inference worker thread ----
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

        self.get_logger().info("Node ready (Async Latest-Frame Mode)")

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

        # 2. Safety (helmet + smoking)
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

        # 4. Face（仅加载 detection + recognition 模块，自动检测 GPU 可用性）
        self._face_app = None
        self._face_use_gpu = False
        if self.enable_face and INSIGHTFACE_AVAILABLE:
            try:
                # 自动检测 ONNX Runtime 是否支持 CUDA / TensorRT
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
    # Image callback：只做格式转换并缓存最新帧，立即返回
    # ------------------------------------------------------------------
    def image_callback(self, msg: RosImage):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Bridge error: {e}")
            return

        self.frame_counter += 1
        with self._latest_frame_lock:
            # 只保留最新帧；推理线程来不及处理的中间帧直接丢弃，避免延迟累积
            self._latest_cv_frame = cv_image

    # ------------------------------------------------------------------
    # Inference worker：从缓存取最新帧执行完整检测流水线
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
            # infer_interval > 0 时限制推理频率，进一步释放 CPU/GPU
            if self.infer_interval > 0.0:
                wait = self.infer_interval - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)

    def _process_frame(self, cv_image):
        h, w = cv_image.shape[:2]
        if w > self.detect_width:
            scale = self.detect_width / w
            new_w = self.detect_width
            new_h = int(h * scale)
            small_frame = cv2.resize(cv_image, (new_w, new_h))
        else:
            small_frame = cv_image

        # 绘制分辨率相对原始分辨率的比例，用于字号自适应
        draw_ratio = small_frame.shape[1] / float(w)
        self._label_font_size = max(14, int(round(36 * draw_ratio)))

        anomalies, all_detections = self.detect_anomalies(small_frame)

        # 报警截图仍使用原始全分辨率图像
        for a in anomalies:
            img_path = self.save_anomaly_screenshot(cv_image, a['type'])
            a['image_path'] = img_path or ''
        if anomalies:
            self.publish_alerts(anomalies)

        # 直接在检测分辨率画面上绘制（WebRTC 推流与窗口显示最终都会缩放到该级别，
        # 在全分辨率原图上绘制没有收益，反而让 PIL 转换代价放大 4 倍）
        display_frame = small_frame.copy()
        display_frame = self.draw_detections(display_frame, anomalies, all_detections)

        info_scale = 1.5 * draw_ratio
        info_thick = max(1, int(round(3 * draw_ratio)))
        info_y1 = max(20, int(round(40 * draw_ratio)))
        info_y2 = max(40, int(round(80 * draw_ratio)))
        cv2.putText(display_frame, f"Frame: {self.frame_counter}", (10, info_y1),
                    cv2.FONT_HERSHEY_SIMPLEX, info_scale, (0, 255, 0), info_thick)
        cv2.putText(display_frame, f"Anomalies: {len(anomalies)}", (10, info_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, info_scale, (0, 255, 0), info_thick)

        self.update_processed_frame(display_frame)

        if self.show_window:
            show_frame = cv2.resize(display_frame, (self.detect_width, self.detect_height))
            cv2.imshow('Anomaly Detection', show_frame)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Detection functions
    # ------------------------------------------------------------------
    def detect_anomalies(self, frame):
        anomalies = []
        all_detections = []
        now = time.time()

        # ---- Fire & Smoke（每个处理帧都检测）----
        if self._fs_miner is not None:
            self._fs_miner._conf_thres_array = np.array(
                [self.smoke_conf_threshold, self.fire_conf_threshold], dtype=np.float32)
            self._fs_miner.fire_color_filter_max_conf = self.fire_color_filter_max_conf
            boxes = self._fs_miner.detect(frame)
            for b in boxes:
                anomaly_type = _FireSmokeMiner.class_names[b.cls_id]
                anomalies.append({
                    'type': anomaly_type,
                    'confidence': b.conf,
                    'position': [(b.x1+b.x2)//2, (b.y1+b.y2)//2],
                    'bbox': [b.x1, b.y1, b.x2, b.y2],
                })

        # ---- Safety（每个处理帧都检测）----
        if self._safety_model is not None or self._safety_helmet is not None:
            safety_results = self.detect_safety(frame)
            # 异常类别：未戴安全帽、吸烟（标签已经转为小写）
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

        # ---- Plate（降频检测 + 结果缓存）----
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

        # ---- Face（异步线程 + 间隔控制 + 结果缓存，不阻塞主流水线）----
        if self._face_app is not None:
            # 收割已完成的人脸检测结果
            if self._face_future is not None and self._face_future.done():
                try:
                    res = self._face_future.result()
                    if res is not None:
                        self._last_face_results = res
                except Exception:
                    pass
                self._face_future = None
            # 到间隔且上一次检测已结束，则提交新的异步检测任务
            if (self._face_future is None and
                    (now - self._last_face_detect_time) >= self.face_detection_interval):
                self._last_face_detect_time = now
                self._face_future = self._face_executor.submit(self.detect_face, frame)
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
        import re
        return re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', plate_text.upper())

    # ------------------------------------------------------------------
    # Drawing（每帧仅一次 PIL 整帧转换，批量绘制所有中文标签）
    # ------------------------------------------------------------------
    def draw_detections(self, frame, anomalies, all_detections=None):
        drawn_boxes = set()
        text_items = []
        font_size = self._label_font_size

        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1,y1,x2,y2 = bbox
                box_key = f"{x1}_{y1}_{x2}_{y2}"
                if box_key in drawn_boxes:
                    continue
                drawn_boxes.add(box_key)
                color = (0, 0, 255)  # 红色代表异常
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                # 构建标签文本
                if anomaly['type'] in ('no_safetyhat', 'cigarette'):
                    label = f"安全违规: {anomaly['type']} {anomaly['confidence']:.2f}"
                elif anomaly['type'] == 'unknown_vehicle':
                    label = f"未知车辆: {anomaly.get('plate_number','')}"
                else:
                    label = f"{anomaly.get('type','')} {anomaly.get('confidence',0):.2f}"
                text_items.append((label, (x1, y1-5), font_size, (0, 0, 255), None))

        if all_detections:
            for det in all_detections:
                bbox = [det.get('x1'), det.get('y1'), det.get('x2'), det.get('y2')]
                if len(bbox) == 4:
                    x1,y1,x2,y2 = bbox
                    box_key = f"{x1}_{y1}_{x2}_{y2}"
                    if box_key in drawn_boxes:
                        continue
                    drawn_boxes.add(box_key)
                    color = (0, 255, 0)  # 绿色代表正常
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    label = det.get('class_name') or det.get('plate_number') or det.get('name') or 'detection'
                    if det.get('source') == 'plate' and 'plate_number' in det:
                        label = f"{det['plate_number']} (已知)" if det['plate_number'] in self.known_plates else f"{det['plate_number']}"
                    text_items.append((label, (x1, y1-5), font_size, (0, 255, 0), None))

        # 整帧只做一次 BGR<->RGB 转换，批量绘制全部标签
        put_chinese_texts(frame, text_items)
        return frame

    # ------------------------------------------------------------------
    # Alert publishing
    # ------------------------------------------------------------------
    def publish_alerts(self, anomalies):
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

            lat = round(random.uniform(39.9000, 39.9100), 6)
            lon = round(random.uniform(116.3000, 116.4000), 6)

            img_path = a.get('image_path', '')
            b64 = ""
            if img_path and os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()

            alert_data = {
                "type": atype,
                "confidence": a.get('confidence', 0.0),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": custom_descriptions.get(atype, a.get('description', '')),
                "position": {"latitude": lat, "longitude": lon},
                "screenshot": b64
            }
            self.alert_publisher.publish(String(data=json.dumps(alert_data, ensure_ascii=False)))
            self.get_logger().info(f"Alert published: {atype} (conf={a.get('confidence',0):.2f})")

    def save_anomaly_screenshot(self, frame, anomaly_type):
        current_time = time.time()
        if anomaly_type in self.last_screenshot_time:
            if current_time - self.last_screenshot_time[anomaly_type] < self.screenshot_interval:
                return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.history_folder, f"{anomaly_type}_{ts}.jpg")
        cv2.imwrite(path, frame)
        self.last_screenshot_time[anomaly_type] = current_time
        return path

    # ------------------------------------------------------------------
    # Processed frame management（不再做整帧拷贝：存储与读取均为引用，
    # 每帧由推理线程新建对象，写入后不再修改，线程安全）
    # ------------------------------------------------------------------
    def update_processed_frame(self, frame):
        with self.processed_frame_lock:
            self.latest_processed_frame = frame

    def get_latest_processed_frame(self):
        with self.processed_frame_lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame
            return np.zeros((self.detect_height, self.detect_width, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # WebRTC server
    # ------------------------------------------------------------------
    def _run_webrtc_server(self):
        try:
            self.webrtc_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.webrtc_loop)
            async def start_server():
                self.websocket_server = await websockets.serve(
                    self._browser_handler, "0.0.0.0", self.webrtc_port
                )
            self.webrtc_loop.run_until_complete(start_server())
            while self.webrtc_running:
                self.webrtc_loop.run_until_complete(asyncio.sleep(0.5))
            for t in asyncio.all_tasks(self.webrtc_loop):
                t.cancel()
            self.webrtc_loop.run_until_complete(asyncio.sleep(0.1))
            self.webrtc_loop.close()
        except Exception as e:
            self.get_logger().error(f"WebRTC server error: {e}")

    async def _browser_handler(self, websocket):
        pc = None
        try:
            pc = RTCPeerConnection()
            track = ProcessedFrameTrack(self)
            pc.addTransceiver(track, direction="sendonly")

            @pc.on("icecandidate")
            async def on_icecandidate(candidate):
                if candidate:
                    await websocket.send(json.dumps({
                        "type": "candidate",
                        "candidate": {
                            "candidate": candidate.candidate,
                            "sdpMid": candidate.sdpMid,
                            "sdpMLineIndex": candidate.sdpMLineIndex
                        }
                    }))

            @pc.on("iceconnectionstatechange")
            async def on_ice_state():
                if pc.iceConnectionState in ["failed", "closed"]:
                    await pc.close()

            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
                    answer = await pc.createAnswer()
                    sdp_lines = answer.sdp.split('\n')
                    new_lines = []
                    for line in sdp_lines:
                        new_lines.append(line)
                        if line.startswith('m=video'):
                            new_lines.append('b=AS:1000')
                    answer.sdp = '\n'.join(new_lines)
                    await pc.setLocalDescription(answer)
                    await websocket.send(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    }))
                elif msg_type == "candidate":
                    cand = self._create_ice_candidate(data["candidate"])
                    if cand:
                        try:
                            await pc.addIceCandidate(cand)
                        except Exception:
                            pass
                elif msg_type == "bye":
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            pass
        finally:
            if pc is not None:
                await pc.close()

    def _create_ice_candidate(self, cand):
        candidate_str = cand.get("candidate", "")
        sdpMid = cand.get("sdpMid")
        sdpMLineIndex = cand.get("sdpMLineIndex")
        if not candidate_str:
            return None
        try:
            candidate = RTCIceCandidate.from_sdp(candidate_str)
            candidate.sdpMid = sdpMid
            candidate.sdpMLineIndex = sdpMLineIndex
            return candidate
        except AttributeError:
            pass
        parts = candidate_str.split()
        if len(parts) < 8:
            return None
        foundation = parts[0].split(':', 1)[1] if ':' in parts[0] else parts[0]
        component = int(parts[1])
        protocol = parts[2]
        priority = int(parts[3])
        ip = parts[4]
        port = int(parts[5])
        typ = parts[7]
        tcp_type = None
        if protocol.upper() == "TCP" and len(parts) > 9 and parts[8] == "tcptype":
            tcp_type = parts[9]
        return SimpleNamespace(
            component=component, foundation=foundation, protocol=protocol, priority=priority,
            ip=ip, port=port, type=typ, tcpType=tcp_type, relatedAddress=None, relatedPort=None,
            candidate=candidate_str, sdpMid=sdpMid, sdpMLineIndex=sdpMLineIndex
        )

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self._worker_stop.set()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        if self._face_executor is not None:
            self._face_executor.shutdown(wait=False)
        self.webrtc_running = False
        if self.webrtc_loop is not None and self.webrtc_loop.is_running():
            self.webrtc_loop.call_soon_threadsafe(self.webrtc_loop.stop)
        if hasattr(self, 'webrtc_thread') and self.webrtc_thread.is_alive():
            self.webrtc_thread.join(timeout=3.0)
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()

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
