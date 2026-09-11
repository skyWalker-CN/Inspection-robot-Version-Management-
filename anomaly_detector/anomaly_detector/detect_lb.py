#!/usr/bin/env python3
"""
智能巡检机器人异常检测节点（同步检测版）
- 同步模式：画面与检测框严格对齐，彻底解决目标移动时框滞后、不跟随的问题
- 人脸检测降频至 5秒/次，避免高频阻塞导致整体卡顿
- 使用 MultiThreadedExecutor 防止图像回调阻塞 ROS2 其他通信
- 所有模型置信度统一为 ROS2 参数，方便动态调整
"""
import sys
import os
import json
import math
import time
import threading
import asyncio
import logging
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
# Fire & Smoke Detector
# ============================================================
class _FireSmokeMiner:
    class_names = ["smoke", "fire"]
    _model_class_order = ["smoke", "fire"]
    iou_thres = 0.45
    cross_iou_thresh = 0.8
    max_det = 150
    _conf_thres_array = np.array([0.6, 0.6], dtype=np.float32)  # will be updated externally
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
            results.append(_BoundingBox(int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2)), int(cls_id), float(conf)))
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
        self.push_width = 640
        self.push_height = 480

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
# ROS2 Node (同步检测架构 + 人脸降频优化)
# ============================================================
class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__("anomaly_detector_node")

        # ---- Parameters ----
        self.declare_parameter("image_topic", "/hik_camera/image_raw")
        self.declare_parameter("detect_width", 960)
        self.declare_parameter("detect_height", 540)
        self.declare_parameter("push_width", 640)
        self.declare_parameter("push_height", 360)
        self.declare_parameter("show_window", False)
        self.declare_parameter("display_scale", 1.5)
        self.declare_parameter("enable_webrtc", True)
        self.declare_parameter("webrtc_port", 8080)
        self.declare_parameter("face_detection_interval", 2.0)

        # ---- Unified confidence thresholds ----
        self.declare_parameter("fire_conf_threshold", 0.6)
        self.declare_parameter("smoke_conf_threshold", 0.8)
        self.declare_parameter("fire_color_filter_max_conf", 0.45)
        self.declare_parameter("safety_conf_threshold", 0.6)
        self.declare_parameter("plate_conf_threshold", 0.5)
        self.declare_parameter("face_similarity_threshold", 0.50)

        # ---- Models directories ----
        models_dir = self._find_package_dir("models")
        face_db_dir = self._find_package_dir("face_database")

        self.declare_parameter("fire_smoke_model_dir", str(models_dir))
        self.declare_parameter("fire_smoke_class_names", [])
        self.declare_parameter("helmet_model_path", str(models_dir / "helmet_best.pt"))
        self.declare_parameter("smoking_model_path", str(models_dir / "smoking_best.pt"))
        self.declare_parameter("safety_combined_model_path", "")
        self.declare_parameter("face_database_dir", str(face_db_dir))

        self.declare_parameter("enable_fire_smoke", True)
        self.declare_parameter("enable_safety", True)
        self.declare_parameter("enable_plate", True)
        self.declare_parameter("enable_face", True)
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

        # Confidence thresholds
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

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Anomaly Detector Node (Sync Detection)")
        self.get_logger().info(f"  Device: {self.device}, FP16: {self.half}")
        self.get_logger().info(f"  Detect size: {self.detect_width}x{self.detect_height}")
        self.get_logger().info(f"  Face Detection Interval: {self.face_detection_interval}s")
        self.get_logger().info(f"  Confidence thresholds:")
        self.get_logger().info(f"    fire: {self.fire_conf_threshold}, smoke: {self.smoke_conf_threshold}")
        self.get_logger().info(f"    fire_color_filter_max: {self.fire_color_filter_max_conf}")
        self.get_logger().info(f"    safety: {self.safety_conf_threshold}")
        self.get_logger().info(f"    plate: {self.plate_conf_threshold}")
        self.get_logger().info(f"    face similarity: {self.face_similarity_threshold}")
        self.get_logger().info("=" * 60)

        # ---- Init detectors ----
        self._init_detectors()

        # ---- Image subscription ----
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            RosImage, self.image_topic, self.image_callback, 10)
        self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")

        # ---- Publishers ----
        self.pub_fire_smoke = self.create_publisher(String, "/anomaly/fire_smoke", 10)
        self.pub_safety = self.create_publisher(String, "/anomaly/safety", 10)
        self.pub_plate = self.create_publisher(String, "/anomaly/plate", 10)
        self.pub_face = self.create_publisher(String, "/anomaly/face", 10)

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
        
        # ---- Face detection interval control ----
        self._last_face_detect_time = 0.0

        # ---- WebRTC server ----
        self.thread_pool = ThreadPoolExecutor(max_workers=2) if WEBRTC_AVAILABLE else None
        self.webrtc_loop = None
        self.webrtc_running = False
        if self.enable_webrtc and WEBRTC_AVAILABLE:
            self.webrtc_running = True
            self.webrtc_thread = threading.Thread(target=self._run_webrtc_server, daemon=True)
            self.webrtc_thread.start()
            self.get_logger().info(f"WebRTC server started on port {self.webrtc_port}")
        elif self.enable_webrtc and not WEBRTC_AVAILABLE:
            self.get_logger().warn("WebRTC dependencies not installed; disabling WebRTC")

        self.get_logger().info("Node ready (Sync Mode)")

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
                # Apply initial thresholds
                self._fs_miner._conf_thres_array = np.array(
                    [self.smoke_conf_threshold, self.fire_conf_threshold], dtype=np.float32)
                self._fs_miner.fire_color_filter_max_conf = self.fire_color_filter_max_conf
                self._fs_miner.warmup()
                self.get_logger().info("[OK] Fire & Smoke detector loaded")
            except Exception as e:
                self.get_logger().error(f"Fire & Smoke init failed: {e}")
        elif self.enable_fire_smoke and not (ONNX_AVAILABLE or YOLO_AVAILABLE):
            self.get_logger().warn("onnxruntime or ultralytics not installed, fire/smoke disabled")

        # 2. Safety
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
        elif self.enable_safety and not YOLO_AVAILABLE:
            self.get_logger().warn("ultralytics not installed, safety disabled")

        # 3. Plate
        self._plate_catcher = None
        if self.enable_plate and HYPERLPR_AVAILABLE:
            try:
                self._plate_catcher = LicensePlateCatcher()
                self.get_logger().info("[OK] Plate detector loaded")
            except Exception as e:
                self.get_logger().error(f"Plate init failed: {e}")
        elif self.enable_plate and not HYPERLPR_AVAILABLE:
            self.get_logger().warn("hyperlpr3 not installed, plate detection disabled")

        # 4. Face
        self._face_app = None
        if self.enable_face and INSIGHTFACE_AVAILABLE:
            try:
                providers = ['CPUExecutionProvider']
                self._face_app = FaceAnalysis(name=self.face_pack, providers=providers)
                self._face_app.prepare(ctx_id=-1, det_size=(self.face_det_size,) * 2)
                self.get_logger().info(f"[OK] Face detector loaded (pack={self.face_pack}, det_size={self.face_det_size})")
                self._init_face_database()
            except Exception as e:
                self.get_logger().error(f"Face init failed: {e}")
        elif self.enable_face and not INSIGHTFACE_AVAILABLE:
            self.get_logger().warn("insightface not installed, face detection disabled")

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
    # Image callback (Sync Mode: Detect and Draw in the same frame)
    # ------------------------------------------------------------------
    def image_callback(self, msg: RosImage):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Bridge error: {e}")
            return

        self.frame_counter += 1

        # ---- Scale frame for detection ----
        h, w = cv_image.shape[:2]
        if w > self.detect_width:
            scale = self.detect_width / w
            new_w = self.detect_width
            new_h = int(h * scale)
            small_frame = cv2.resize(cv_image, (new_w, new_h))
        else:
            scale = 1.0
            small_frame = cv_image

        # ---- Sync Detection (Blocking) ----
        anomalies, all_detections = self.detect_anomalies(small_frame, cv_image)

        # ---- Draw on current frame ----
        display_frame = cv_image.copy()
        display_frame = self.draw_detections(display_frame, anomalies, all_detections)

        # ---- Overlay info ----
        cv2.putText(display_frame, f"Frame: {self.frame_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(display_frame, f"Anomalies: {len(anomalies)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        # ---- Store for WebRTC ----
        self.update_processed_frame(display_frame)

        # ---- Local display ----
        if self.show_window:
            show_frame = cv2.resize(display_frame, (self.detect_width, self.detect_height))
            cv2.imshow('Anomaly Detection', show_frame)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Detection functions
    # ------------------------------------------------------------------
    def detect_anomalies(self, frame, original_frame=None):
        if original_frame is None:
            original_frame = frame

        anomalies = []
        all_detections = []

        # ---- Fire & Smoke ----
        if self._fs_miner is not None:
            # Update thresholds from parameters (allows runtime adjustment)
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

        # ---- Safety ----
        if self._safety_model is not None or self._safety_helmet is not None:
            safety_results = self.detect_safety(frame)
            all_detections.extend(safety_results)

        # ---- Plate ----
        if self._plate_catcher is not None:
            plate_results = self.detect_plate(frame)
            all_detections.extend(plate_results)

        # ---- Face (Interval control) ----
        if self._face_app is not None:
            current_time = time.time()
            if (current_time - self._last_face_detect_time) >= self.face_detection_interval:
                face_results = self.detect_face(frame)
                self._last_face_detect_time = current_time
                
                for f in face_results:
                    if f.get('name') == 'unknown':
                        anomalies.append({
                            'type': 'unknown_person',
                            'confidence': f['conf'],
                            'bbox': [f['x1'], f['y1'], f['x2'], f['y2']],
                            'person_name': 'unknown',
                        })
                    else:
                        all_detections.append(f)

        # Scale bboxes to original frame coordinates
        oh, ow = original_frame.shape[:2]
        sh, sw = frame.shape[:2]
        scale_x = ow / sw
        scale_y = oh / sh

        for a in anomalies:
            bbox = a.get('bbox')
            if bbox:
                a['bbox'] = [int(bbox[0]*scale_x), int(bbox[1]*scale_y), int(bbox[2]*scale_x), int(bbox[3]*scale_y)]

        for d in all_detections:
            bbox = [d.get('x1'), d.get('y1'), d.get('x2'), d.get('y2')]
            if bbox:
                d['x1'] = int(bbox[0]*scale_x)
                d['y1'] = int(bbox[1]*scale_y)
                d['x2'] = int(bbox[2]*scale_x)
                d['y2'] = int(bbox[3]*scale_y)

        return anomalies, all_detections

    def detect_safety(self, frame):
        results = []
        conf = self.safety_conf_threshold  # use parameter
        if self._safety_combined and self._safety_model is not None:
            try:
                r = self._safety_model(frame, conf=conf, verbose=False, device=self.device)
                for res in r:
                    for box in res.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                        results.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'conf': float(box.conf[0]),
                            'class_name': self._safety_model.names.get(int(box.cls[0]), 'unknown'),
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
                            results.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'conf': float(box.conf[0]),
                                'class_name': model.names.get(int(box.cls[0]), 'unknown'),
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
                # Confidence filtering
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
                    if best_sim >= self.face_similarity_threshold:  # use parameter
                        result['name'] = self._face_db_names[best_idx]
                results.append(result)
        except Exception as e:
            self.get_logger().debug(f"Face detect error: {e}")
        return results

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_detections(self, frame, anomalies, all_detections=None):
        drawn_boxes = set()
        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1,y1,x2,y2 = bbox
                box_key = f"{x1}_{y1}_{x2}_{y2}"
                if box_key in drawn_boxes:
                    continue
                drawn_boxes.add(box_key)
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                label = f"{anomaly.get('type','')} {anomaly.get('confidence',0):.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if all_detections:
            for det in all_detections:
                bbox = [det.get('x1'), det.get('y1'), det.get('x2'), det.get('y2')]
                if len(bbox) == 4:
                    x1,y1,x2,y2 = bbox
                    box_key = f"{x1}_{y1}_{x2}_{y2}"
                    if box_key in drawn_boxes:
                        continue
                    drawn_boxes.add(box_key)
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    label = det.get('class_name') or det.get('plate_number') or det.get('name') or 'detection'
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish_anomalies(self, anomalies):
        for a in anomalies:
            if a['type'] == 'fire':
                self._publish('fire_smoke', [{'class': 'fire', 'conf': a['confidence'], 'box': a['bbox']}])
            elif a['type'] == 'smoke':
                self._publish('fire_smoke', [{'class': 'smoke', 'conf': a['confidence'], 'box': a['bbox']}])
            elif a['type'] == 'unknown_person':
                self._publish('face', [{'name': 'unknown', 'conf': a['confidence'], 'box': a['bbox']}])

    def _publish(self, topic_key, data):
        if not rclpy.ok():
            return
        msg = String()
        msg.data = json.dumps(data)
        pub = getattr(self, f"pub_{topic_key}", None)
        if pub is not None:
            try:
                pub.publish(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Processed frame management
    # ------------------------------------------------------------------
    def update_processed_frame(self, frame):
        with self.processed_frame_lock:
            self.latest_processed_frame = frame.copy()

    def get_latest_processed_frame(self):
        with self.processed_frame_lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
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

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
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
    
    # Use MultiThreadedExecutor to prevent image callback from blocking other ROS communications
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
