#!/usr/bin/env python3
"""
Integrated Anomaly Detector Node — single-file implementation.

Combines four detection capabilities in one ROS2 node:
  1. Fire & Smoke Detection         (ONNX Runtime)
  2. Safety Helmet & Smoking        (YOLO / ultralytics)
  3. License Plate Detection        (HyperLPR3)
  4. Face Detection & Recognition   (InsightFace)

All results are rendered on a single preview window and published to
separate ROS2 topics as JSON strings.
"""

import sys

# ============================================================
# 添加 Conda 虚拟环境路径（必须在其他 import 之前）
# 使主环境能使用各虚拟环境中的依赖
# ============================================================

# 火焰烟雾检测依赖 (onnxruntime, opencv-python, numpy 等)
sys.path.insert(
    0,
    '/home/jetson/miniforge3/envs/anomaly_detect/lib/python3.10/site-packages'
)


# ============================================================
# 其他标准库和第三方库导入
# ============================================================

import json
import math
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ============================================================
# 1. Fire & Smoke Detection (ONNX Runtime)
# ============================================================

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class _BoundingBox:
    __slots__ = ("x1", "y1", "x2", "y2", "cls_id", "conf")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, cls_id: int, conf: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls_id = cls_id
        self.conf = conf


class _FireSmokeMiner:
    """ONNX Runtime miner for fire / smoke / fire_extinguisher detection."""

    class_names = ["fire", "smoke", "fire extinguisher"]
    _model_class_order = ["fire", "fire extinguisher", "smoke"]
    iou_thres = 0.45
    cross_iou_thresh = 0.8
    max_det = 150
    _conf_thres_array = np.array([0.42, 0.2, 0.15], dtype=np.float32)
    _bonus_array = np.array([0.28, 0.05, 0.05], dtype=np.float32)
    min_box_area = 14 * 14
    min_side = 8
    max_aspect_ratio = 8.0
    smoke_merge_overlap = 0.8
    fire_color_filter_max_conf = 0.45
    fire_ext_color_filter_max_conf = 0.40
    color_filter_min_saturation = 0.06
    use_edge_filter = False
    edge_filter_max_conf = 0.0
    edge_tol = 2.0
    use_tta = False

    def __init__(self, model_dir: Path):
        model_path = model_dir / "weights.onnx"
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 4

        available_providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available_providers:
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options,
            providers=providers,
        )
        model_class_order = self._read_model_class_order()
        if model_class_order is None:
            model_class_order = list(self._model_class_order)
        self.cls_remap = np.array(
            [self.class_names.index(n) for n in model_class_order], dtype=np.int32)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self._safe_dim(self.input_shape[2], 1280)
        self.input_width = self._safe_dim(self.input_shape[3], 1280)

    @staticmethod
    def _safe_dim(value, default):
        return value if isinstance(value, int) and value > 0 else default

    def _read_model_class_order(self):
        try:
            import ast
            meta = self.session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(meta["names"])
            if isinstance(names, dict):
                order = [str(names[i]) for i in sorted(names)]
            else:
                order = [str(n) for n in names]
        except Exception:
            return None
        if sorted(order) != sorted(self.class_names):
            return None
        return order

    def _letterbox(self, image, new_shape, color=(114, 114, 114)):
        h, w = image.shape[:2]
        new_w, new_h = new_shape
        ratio = min(new_w / w, new_h / h)
        rw, rh = int(round(w * ratio)), int(round(h * ratio))
        if (rw, rh) != (w, h):
            interp = cv2.INTER_CUBIC if ratio > 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (rw, rh), interpolation=interp)
        dw = (new_w - rw) / 2.0
        dh = (new_h - rh) / 2.0
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        return cv2.copyMakeBorder(image, top, bottom, left, right,
                                  borderType=cv2.BORDER_CONSTANT, value=color), ratio, (dw, dh)

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
            a_r = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * \
                  np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
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

    def _cross_class_dedup(self, boxes, scores, cls_ids, iou_thresh):
        n = len(boxes)
        if n <= 1:
            return boxes, scores, cls_ids
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        cls_ids = np.asarray(cls_ids, dtype=np.int32)
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * \
                np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        margins = scores - self._conf_thres_array[cls_ids]
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
                for j in range(i + 1, len(sb)):
                    a, b = sb[i], sb[j]
                    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    if inter / (min(area_a, area_b) + 1e-7) >= self.smoke_merge_overlap:
                        sb[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                 max(a[2], b[2]), max(a[3], b[3])]
                        ss[i] = max(ss[i], ss[j])
                        del sb[j]; del ss[j]
                        merged = True; break
                if merged:
                    break
        other = cls_ids != smoke_cls
        nb = np.concatenate([boxes[other].astype(np.float32),
                             np.array(sb, dtype=np.float32).reshape(-1, 4)])
        ns = np.concatenate([scores[other].astype(np.float32), np.array(ss, dtype=np.float32)])
        nc = np.concatenate([cls_ids[other].astype(np.int32),
                             np.full(len(sb), smoke_cls, dtype=np.int32)])
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
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), \
                   np.empty((0,), dtype=np.int32)
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
            boxes, scores, cls_ids = self._cross_class_dedup(
                boxes, scores, cls_ids, self.cross_iou_thresh)
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
        return roi if roi.size else None

    def _roi_is_grayscale(self, roi):
        mx = roi.max(axis=2).astype(np.float32)
        mn = roi.min(axis=2).astype(np.float32)
        return float(((mx - mn) / (mx + 1e-6)).mean()) < self.color_filter_min_saturation

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

    @staticmethod
    def _passes_ext_color(roi):
        blue = roi[:, :, 0].astype(np.float32)
        green = roi[:, :, 1].astype(np.float32)
        red = roi[:, :, 2].astype(np.float32)
        if float(np.mean((red > green + 10.0) & (red > blue + 10.0))) >= 0.03:
            return True
        if (float(np.mean(red)) - float(np.mean(green))) >= 0.0 and float(np.mean(red)) >= 50.0:
            return True
        return False

    def _filter_color(self, image, results):
        if not results:
            return results
        cls_fire = self.class_names.index("fire")
        cls_ext = self.class_names.index("fire extinguisher")
        out = []
        for box in results:
            cf = box.cls_id == cls_fire and box.conf <= self.fire_color_filter_max_conf
            ce = box.cls_id == cls_ext and box.conf <= self.fire_ext_color_filter_max_conf
            if not cf and not ce:
                out.append(box); continue
            roi = self._roi_for_box(image, box)
            if roi is None or self._roi_is_grayscale(roi):
                out.append(box); continue
            if cf and not self._passes_fire_color(roi):
                continue
            if ce and not self._passes_ext_color(roi):
                continue
            out.append(box)
        return out

    @staticmethod
    def _build_results(boxes, scores, cls_ids):
        results = []
        for box, conf, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(_BoundingBox(
                int(math.floor(x1)), int(math.floor(y1)),
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
        boxes[:, [0, 2]] -= pw; boxes[:, [1, 3]] -= ph
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
        boxes[:, [0, 2]] -= pw; boxes[:, [1, 3]] -= ph
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

    def _predict_tta(self, image):
        boxes_orig = self._predict_single(image)
        flipped = cv2.flip(image, 1)
        boxes_flip = self._predict_single(flipped)
        w = image.shape[1]
        boxes_flip = [_BoundingBox(x1=w - b.x2, y1=b.y1, x2=w - b.x1, y2=b.y2,
                                   cls_id=b.cls_id, conf=b.conf) for b in boxes_flip]
        all_boxes = boxes_orig + boxes_flip
        if not all_boxes:
            return []
        coords = np.array([[b.x1, b.y1, b.x2, b.y2] for b in all_boxes], dtype=np.float32)
        scores = np.array([b.conf for b in all_boxes], dtype=np.float32)
        cls_ids = np.array([b.cls_id for b in all_boxes], dtype=np.int32)
        hard_keep = self._per_class_nms(coords, scores, cls_ids, self.iou_thres)
        if len(hard_keep) == 0:
            return []
        if len(hard_keep) > self.max_det:
            top = np.argsort(-scores[hard_keep])[:self.max_det]
            hard_keep = hard_keep[top]
        kept_coords, kept_cls, kept_scores = coords[hard_keep], cls_ids[hard_keep], scores[hard_keep]
        if len(kept_coords) > 1:
            kept_coords, kept_scores, kept_cls = self._cross_class_dedup(
                kept_coords, kept_scores, kept_cls, self.cross_iou_thresh)
        if len(kept_coords) > 1:
            kept_coords, kept_scores, kept_cls = self._merge_smoke(
                kept_coords, kept_scores, kept_cls)
        return [_BoundingBox(
            int(math.floor(kept_coords[j, 0])), int(math.floor(kept_coords[j, 1])),
            int(math.ceil(kept_coords[j, 2])), int(math.ceil(kept_coords[j, 3])),
            int(kept_cls[j]), float(kept_scores[j])) for j in range(len(kept_coords))]

    def detect(self, image):
        try:
            boxes = self._predict_tta(image) if self.use_tta else self._predict_single(image)
            boxes = self._filter_color(image, boxes)
        except Exception as e:
            print(f"FireSmokeMiner error: {e}")
            boxes = []
        return boxes


# ============================================================
# 2. Safety Detector (YOLO — helmet + smoking)
# ============================================================

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ============================================================
# 3. Plate Detector (HyperLPR3)
# ============================================================

try:
    from hyperlpr3 import LicensePlateCatcher
    HYPERLPR_AVAILABLE = True
except ImportError:
    HYPERLPR_AVAILABLE = False


# ============================================================
# 4. Face Detector (InsightFace)
# ============================================================

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


# ============================================================
# ROS2 Node
# ============================================================

SOURCE_COLORS = {
    "fire_smoke": (0, 0, 255),
    "safety":     (255, 0, 0),
    "plate":      (0, 255, 255),
    "face":       (0, 255, 0),
}


class AnomalyDetectorNode(Node):

    def __init__(self):
        super().__init__("anomaly_detector_node")

        # ---- Parameters ----
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("fps", 15)
        self.declare_parameter("show_window", True)

        pkg_root = Path(__file__).resolve().parents[1]
        models_dir = pkg_root / "models"
        self.declare_parameter("fire_smoke_model_dir", str(models_dir))
        self.declare_parameter("helmet_model_path", str(models_dir / "helmet_best.pt"))
        self.declare_parameter("smoking_model_path", str(models_dir / "smoking_best.pt"))
        self.declare_parameter("face_database_dir", str(pkg_root / "face_database"))

        self.declare_parameter("enable_fire_smoke", True)
        self.declare_parameter("enable_safety", True)
        self.declare_parameter("enable_plate", True)
        self.declare_parameter("enable_face", True)

        self.declare_parameter("skip_frames", 2)
        self.declare_parameter("parallel_detection", True)

        # ---- Read parameters ----
        camera_id = self.get_parameter("camera_id").value
        self.width = self.get_parameter("frame_width").value
        self.height = self.get_parameter("frame_height").value
        self.fps = self.get_parameter("fps").value
        self.show_window = self.get_parameter("show_window").value

        self.enable_fire_smoke = self.get_parameter("enable_fire_smoke").value
        self.enable_safety = self.get_parameter("enable_safety").value
        self.enable_plate = self.get_parameter("enable_plate").value
        self.enable_face = self.get_parameter("enable_face").value

        self.skip_frames = self.get_parameter("skip_frames").value
        self.parallel_detection = self.get_parameter("parallel_detection").value
        self._frame_counter = 0

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Integrated Anomaly Detector Node Starting")
        self.get_logger().info("=" * 60)

        # ---- Init all detectors ----
        self._init_detectors()

        # ---- Camera ----
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"Camera {camera_id} open failed!")
            raise RuntimeError("Camera open failed")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.get_logger().info(f"Camera opened (ID={camera_id}, {self.width}x{self.height})")

        # ---- Publishers ----
        self.pub_fire_smoke = self.create_publisher(String, "/anomaly/fire_smoke", 10)
        self.pub_safety = self.create_publisher(String, "/anomaly/safety", 10)
        self.pub_plate = self.create_publisher(String, "/anomaly/plate", 10)
        self.pub_face = self.create_publisher(String, "/anomaly/face", 10)

        # ---- Window ----
        if self.show_window:
            cv2.namedWindow("Anomaly Detector", cv2.WINDOW_NORMAL)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / max(self.fps, 1), self.timer_callback)

        # ---- Stats ----
        self._frame_count = 0
        self._last_log = time.time()

        # ---- Thread pool for parallel detection ----
        if self.parallel_detection:
            self._executor = ThreadPoolExecutor(max_workers=4)
        else:
            self._executor = None

        # ---- Cached results for skip frames ----
        self._cached_results = []

        self.get_logger().info("Anomaly Detector Node ready")

    # ------------------------------------------------------------------
    # Detector initialisation
    # ------------------------------------------------------------------

    def _init_detectors(self):
        # 1. Fire & Smoke (ONNX)
        self._fs_miner = None
        if self.enable_fire_smoke and ONNX_AVAILABLE:
            try:
                model_dir = Path(self.get_parameter("fire_smoke_model_dir").value)
                self._fs_miner = _FireSmokeMiner(model_dir)
                self.get_logger().info("Fire & Smoke detector loaded")
            except Exception as e:
                self.get_logger().error(f"Fire & Smoke init failed: {e}")
        elif self.enable_fire_smoke and not ONNX_AVAILABLE:
            self.get_logger().warn("onnxruntime not installed, fire/smoke disabled")

        # 2. Safety (YOLO)
        self._safety_helmet = None
        self._safety_smoking = None
        if self.enable_safety and YOLO_AVAILABLE:
            try:
                hp = self.get_parameter("helmet_model_path").value
                sp = self.get_parameter("smoking_model_path").value
                self._safety_helmet = YOLO(hp)
                self._safety_smoking = YOLO(sp)
                self.get_logger().info("Safety (helmet+smoking) detector loaded")
            except Exception as e:
                self.get_logger().error(f"Safety init failed: {e}")
        elif self.enable_safety and not YOLO_AVAILABLE:
            self.get_logger().warn("ultralytics not installed, safety disabled")

        # 3. Plate (HyperLPR3)
        self._plate_catcher = None
        self._plate_font = None
        if self.enable_plate and HYPERLPR_AVAILABLE:
            try:
                self._plate_catcher = LicensePlateCatcher()
                font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
                if not os.path.exists(font_path):
                    font_path = "/usr/share/fonts/truetype/arphic/uming.ttc"
                try:
                    self._plate_font = ImageFont.truetype(font_path, 24)
                except Exception:
                    self._plate_font = ImageFont.load_default()
                self.get_logger().info("Plate detector loaded")
            except Exception as e:
                self.get_logger().error(f"Plate init failed: {e}")
        elif self.enable_plate and not HYPERLPR_AVAILABLE:
            self.get_logger().warn("hyperlpr3 not installed, plate detection disabled")

        # 4. Face (InsightFace)
        self._face_app = None
        self._face_database = {}
        self._face_threshold = 0.50
        if self.enable_face and INSIGHTFACE_AVAILABLE:
            try:
                self._face_app = FaceAnalysis(
                    name='buffalo_l', providers=['CPUExecutionProvider'])
                self._face_app.prepare(ctx_id=-1, det_size=(640, 640))
                db_dir = self.get_parameter("face_database_dir").value
                if os.path.isdir(db_dir):
                    for name in os.listdir(db_dir):
                        feat_path = os.path.join(db_dir, name, "feature.npy")
                        if os.path.exists(feat_path):
                            self._face_database[name] = np.load(feat_path)
                            self.get_logger().info(f"Face loaded: {name}")
                self.get_logger().info(
                    f"Face detector loaded ({len(self._face_database)} faces)")
            except Exception as e:
                self.get_logger().error(f"Face init failed: {e}")
        elif self.enable_face and not INSIGHTFACE_AVAILABLE:
            self.get_logger().warn("insightface not installed, face detection disabled")

    # ------------------------------------------------------------------
    # Timer callback — run all detectors on each frame
    # ------------------------------------------------------------------

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self._frame_counter += 1
        use_cached = (self._frame_counter % (self.skip_frames + 1)) != 0

        if use_cached:
            all_results = self._cached_results
        else:
            if self.parallel_detection and self._executor is not None:
                all_results = self._run_parallel_detection(frame)
            else:
                all_results = self._run_sequential_detection(frame)
            self._cached_results = all_results

        if self.show_window:
            display = self._draw_results(frame.copy(), all_results)
            cv2.imshow("Anomaly Detector", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("User pressed 'q', shutting down")
                rclpy.shutdown()
                return

        self._frame_count += 1
        now = time.time()
        if now - self._last_log >= 5.0:
            self.get_logger().info(
                f"FPS: {self._frame_count / (now - self._last_log):.1f}, "
                f"detections: {len(all_results)}")
            self._frame_count = 0
            self._last_log = now

    def _run_sequential_detection(self, frame):
        all_results = []

        if self._fs_miner is not None:
            boxes = self._fs_miner.detect(frame)
            for b in boxes:
                all_results.append({
                    "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                    "conf": b.conf,
                    "class_name": _FireSmokeMiner.class_names[b.cls_id],
                    "source": "fire_smoke",
                })
            if boxes:
                self._publish("fire_smoke", [{
                    "class": _FireSmokeMiner.class_names[b.cls_id],
                    "conf": b.conf, "box": [b.x1, b.y1, b.x2, b.y2]
                } for b in boxes])

        if self._safety_helmet is not None:
            safety_results = self._run_safety(frame)
            all_results.extend(safety_results)
            if safety_results:
                self._publish("safety", [{
                    "class": r["class_name"], "conf": r["conf"],
                    "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in safety_results])

        if self._plate_catcher is not None:
            plate_results = self._run_plate(frame)
            all_results.extend(plate_results)
            if plate_results:
                self._publish("plate", [{
                    "plate": r["plate_number"], "conf": r["conf"],
                    "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in plate_results])

        if self._face_app is not None:
            face_results = self._run_face(frame)
            all_results.extend(face_results)
            if face_results:
                self._publish("face", [{
                    "name": r.get("name", "unknown"),
                    "similarity": r.get("similarity", 0.0),
                    "conf": r["conf"], "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in face_results])

        return all_results

    def _run_parallel_detection(self, frame):
        futures = {}

        if self._fs_miner is not None:
            futures['fire_smoke'] = self._executor.submit(self._fs_miner.detect, frame)

        if self._safety_helmet is not None:
            futures['safety'] = self._executor.submit(self._run_safety, frame)

        if self._plate_catcher is not None:
            futures['plate'] = self._executor.submit(self._run_plate, frame)

        if self._face_app is not None:
            futures['face'] = self._executor.submit(self._run_face, frame)

        all_results = []

        if 'fire_smoke' in futures:
            boxes = futures['fire_smoke'].result()
            for b in boxes:
                all_results.append({
                    "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                    "conf": b.conf,
                    "class_name": _FireSmokeMiner.class_names[b.cls_id],
                    "source": "fire_smoke",
                })
            if boxes:
                self._publish("fire_smoke", [{
                    "class": _FireSmokeMiner.class_names[b.cls_id],
                    "conf": b.conf, "box": [b.x1, b.y1, b.x2, b.y2]
                } for b in boxes])

        if 'safety' in futures:
            safety_results = futures['safety'].result()
            all_results.extend(safety_results)
            if safety_results:
                self._publish("safety", [{
                    "class": r["class_name"], "conf": r["conf"],
                    "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in safety_results])

        if 'plate' in futures:
            plate_results = futures['plate'].result()
            all_results.extend(plate_results)
            if plate_results:
                self._publish("plate", [{
                    "plate": r["plate_number"], "conf": r["conf"],
                    "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in plate_results])

        if 'face' in futures:
            face_results = futures['face'].result()
            all_results.extend(face_results)
            if face_results:
                self._publish("face", [{
                    "name": r.get("name", "unknown"),
                    "similarity": r.get("similarity", 0.0),
                    "conf": r["conf"], "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in face_results])

        return all_results

    # ------------------------------------------------------------------
    # Individual detector runners
    # ------------------------------------------------------------------

    def _run_safety(self, frame):
        results = []
        conf = 0.9
        # Helmet
        try:
            for r in self._safety_helmet(frame, conf=conf, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    results.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(box.conf[0]),
                        "class_name": self._safety_helmet.names.get(int(box.cls[0]), "unknown"),
                        "source": "safety",
                    })
        except Exception as e:
            self.get_logger().debug(f"Helmet detect error: {e}")
        # Smoking
        try:
            for r in self._safety_smoking(frame, conf=conf, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    results.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": float(box.conf[0]),
                        "class_name": self._safety_smoking.names.get(int(box.cls[0]), "unknown"),
                        "source": "safety",
                    })
        except Exception as e:
            self.get_logger().debug(f"Smoking detect error: {e}")
        return results

    def _run_plate(self, frame):
        results = []
        try:
            plates = self._plate_catcher(frame)
            for plate in plates:
                plate_number = plate[0]
                score = float(plate[1])
                x1, y1, x2, y2 = plate[3]
                results.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "plate_number": plate_number, "conf": score, "source": "plate",
                })
        except Exception as e:
            self.get_logger().debug(f"Plate detect error: {e}")
        return results

    def _run_face(self, frame):
        results = []
        try:
            faces = self._face_app.get(frame)
            for face in faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box.tolist()
                det_score = float(face.det_score)
                result = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": det_score, "source": "face",
                    "name": "unknown", "similarity": 0.0,
                }
                if hasattr(face, 'embedding') and face.embedding is not None:
                    emb = face.embedding
                    best_name, best_sim = "unknown", 0.0
                    for name, db_emb in self._face_database.items():
                        sim = float(np.dot(emb, db_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(db_emb) + 1e-8))
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name
                    result["similarity"] = best_sim
                    if best_sim >= self._face_threshold:
                        result["name"] = best_name
                results.append(result)
        except Exception as e:
            self.get_logger().debug(f"Face detect error: {e}")
        return results

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_results(self, frame, results):
        for r in results:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            source = r.get("source", "unknown")
            color = SOURCE_COLORS.get(source, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if source == "fire_smoke":
                label = f"{r['class_name']} {r['conf']:.2f}"
            elif source == "safety":
                label = f"{r['class_name']} {r['conf']:.2f}"
            elif source == "plate":
                label = f"{r['plate_number']} {r['conf']:.2f}"
            elif source == "face":
                label = f"{r.get('name', 'unknown')} {r.get('similarity', 0):.2f}"
            else:
                label = f"{r.get('class_name', '')} {r.get('conf', 0):.2f}"

            text_y = y1 - 10 if y1 - 10 > 15 else y1 + 20

            if source == "plate" and self._plate_font is not None:
                frame = self._draw_chinese_text(frame, label, (x1, text_y), color)
            else:
                cv2.putText(frame, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _draw_chinese_text(self, img, text, position, text_color=(0, 255, 0)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=self._plate_font,
                  fill=(text_color[2], text_color[1], text_color[0]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish(self, topic_key, data):
        msg = String()
        msg.data = json.dumps(data)
        pub = getattr(self, f"pub_{topic_key}", None)
        if pub is not None:
            pub.publish(msg)

    # ------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down AnomalyDetectorNode")
        if self._executor is not None:
            self._executor.shutdown(wait=False)
        if self.cap.isOpened():
            self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
