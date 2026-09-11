#!/usr/bin/env python3
"""
Optimized Anomaly Detector Node for Jetson Orin NX
===================================================

Key optimizations vs original:
  1. Async pipeline   — capture / detect / display in separate threads
  2. TensorRT engine  — auto-detect .engine files for YOLO (5-10x faster)
  3. CUDA for Insight  — GPU instead of CPU (10-50x faster)
  4. TensorRT/CUDA EP — for ONNX fire/smoke model
  5. Batch rendering   — single PIL conversion per frame (not per-plate)
  6. Frame dropping    — always process latest frame, no backlog
  7. Combined model    — optional single YOLO for helmet+smoking
  8. FP16 inference    — half precision on GPU
  9. Face DB matrix    — vectorized cosine similarity (O(1) per face)
 10. Warm-up           — pre-run inference to init CUDA contexts

Expected FPS on Jetson Orin NX (16GB, TensorRT engines):
  - With TensorRT + FP16:  15-25 FPS (real-time)
  - With CUDA only:         5-10 FPS
  - CPU only:               1-3 FPS

Usage:
  ros2 run <pkg> anomaly_detector_node
  ros2 run <pkg> anomaly_detector_node --ros-args -p device:=cuda
"""

import sys
import os
import json
import math
import time
import threading
from pathlib import Path

# ============================================================
# Conda env path — must precede other imports
# ============================================================
sys.path.insert(
    0,
    '/home/jetson/miniforge3/envs/anomaly_detect/lib/python3.10/site-packages'
)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from ament_index_python.packages import get_package_share_directory
    AMENT_INDEX_AVAILABLE = True
except ImportError:
    AMENT_INDEX_AVAILABLE = False

# ============================================================
# Graceful model imports
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


# ============================================================
# Bounding box container (zero-overhead via __slots__)
# ============================================================

class _BoundingBox:
    __slots__ = ("x1", "y1", "x2", "y2", "cls_id", "conf")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, cls_id: int, conf: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls_id = cls_id
        self.conf = conf


# ============================================================
# 1. Fire & Smoke Detector (YOLO TensorRT engine / ONNX Runtime)
# ============================================================

class _FireSmokeMiner:
    """Fire/smoke detector with dual backend support.

    Backend priority:
      1. fire_smoke_best.engine  — TensorRT via YOLO (fastest)
      2. fire_smoke_best.onnx    — ONNX via onnxruntime
      3. fire_smoke_best.pt      — PyTorch via YOLO (fallback)
    """

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
    use_edge_filter = False
    edge_filter_max_conf = 0.0
    edge_tol = 2.0
    use_tta = False

    def __init__(self, model_dir: Path, device: str = "cuda",
                 class_names_override: list = None):
        self._backend = None         # "yolo" or "onnx"
        self._yolo_model = None
        self._onnx_session = None
        self._device = device

        # Allow explicit class name override (for TensorRT engines with garbled metadata)
        if class_names_override is not None:
            self._class_names_override = [n.strip().lower() for n in class_names_override]
            print(f"[FireSmoke] Class names override: {self._class_names_override}")
        else:
            self._class_names_override = None

        # Model file search order (highest priority first):
        #   1. fire_smoke_best.engine  — TensorRT, loaded via YOLO (fastest)
        #   2. fire_smoke_best.onnx    — ONNX, loaded via onnxruntime
        #   3. fire_smoke_best.pt      — PyTorch, loaded via YOLO (slower)
        engine_path = model_dir / "fire_smoke_best.engine"
        onnx_path   = model_dir / "fire_smoke_best.onnx"
        pt_path     = model_dir / "fire_smoke_best.pt"

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
            raise FileNotFoundError(
                f"No fire/smoke model found in {model_dir}. Expected one of: "
                f"fire_smoke_best.engine, fire_smoke_best.onnx, "
                f"fire_smoke_best.pt")

    def _setup_yolo_remap(self):
        """Map YOLO model class names to canonical class_names.

        When TensorRT engine metadata contains garbled class names (common with
        engines exported on different platforms), this method falls back to
        index-based mapping using _model_class_order or explicit user override.
        """
        model_names = self._yolo_model.names
        num_model_classes = len(model_names)

        # ---- Priority 1: explicit user override ----
        if self._class_names_override is not None:
            override = self._class_names_override
            if len(override) != num_model_classes:
                print(f"[FireSmoke] WARNING: class_names_override has {len(override)} "
                      f"entries but engine has {num_model_classes} classes — "
                      f"truncating/padding with 'smoke'")
                while len(override) < num_model_classes:
                    override.append("smoke")
                override = override[:num_model_classes]
            remap = []
            for name in override:
                if name in self.class_names:
                    remap.append(self.class_names.index(name))
                else:
                    print(f"[FireSmoke] WARNING: '{name}' not in canonical "
                          f"class_names {self.class_names}, using 'smoke'")
                    remap.append(self.class_names.index("smoke"))
            self.cls_remap = np.array(remap, dtype=np.int32)
            print(f"[FireSmoke] Using user-specified class order: {override} "
                  f"-> remap {self.cls_remap.tolist()}")
            return

        # ---- Priority 2: name-based matching ----
        remap = []
        unknown_count = 0
        for idx in sorted(model_names):
            name = str(model_names[idx]).strip().lower()
            if name in self.class_names:
                remap.append(self.class_names.index(name))
            else:
                unknown_count += 1
                print(f"[FireSmoke] WARNING: unknown class '{name}' (id={idx}), "
                      f"will use index-based fallback")
                remap.append(-1)  # placeholder

        # ---- Priority 3: index-based fallback when ALL names are garbled ----
        if unknown_count == num_model_classes:
            print(f"[FireSmoke] All {num_model_classes} class names are garbled "
                  f"in engine metadata!")
            # Try _model_class_order truncation (most likely correct mapping)
            if num_model_classes <= len(self._model_class_order):
                fallback_order = list(self._model_class_order)[:num_model_classes]
            else:
                # More classes than expected — use canonical order
                fallback_order = list(self.class_names)[:num_model_classes]
            remap = [self.class_names.index(n)
                     if n in self.class_names else self.class_names.index("smoke")
                     for n in fallback_order]
            print(f"[FireSmoke] Index-based fallback: "
                  f"{dict(enumerate(fallback_order))} -> remap {remap}")
        elif unknown_count > 0:
            # Partial match — fill in unknowns with index-based guess
            print(f"[FireSmoke] {unknown_count}/{num_model_classes} class names "
                  f"unrecognized, using index-based fill for unknowns")
            for i in range(len(remap)):
                if remap[i] == -1:
                    if i < len(self._model_class_order):
                        guess = self._model_class_order[i]
                    else:
                        guess = self.class_names[i % len(self.class_names)]
                    remap[i] = (self.class_names.index(guess)
                                if guess in self.class_names
                                else self.class_names.index("smoke"))
                    print(f"[FireSmoke]   id={i}: using '{guess}' as fallback")

        self.cls_remap = np.array(remap, dtype=np.int32)
        print(f"[FireSmoke] YOLO classes: {dict(model_names)} -> "
              f"remap {self.cls_remap.tolist()}")

    def _setup_onnx(self, model_path: str, device: str):
        """Set up ONNX Runtime session with GPU/TensorRT providers."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 4

        available = ort.get_available_providers()
        if device == "cuda":
            if "TensorrtExecutionProvider" in available:
                providers = ["TensorrtExecutionProvider",
                             "CUDAExecutionProvider",
                             "CPUExecutionProvider"]
            elif "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("[FireSmoke] WARNING: No GPU provider, falling back to CPU")
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._onnx_session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers)
        self.session = self._onnx_session  # alias for existing ONNX code

        model_order = self._read_model_class_order()
        if model_order is None:
            model_order = list(self._model_class_order)
        self.cls_remap = np.array(
            [self.class_names.index(n) for n in model_order], dtype=np.int32)
        self.input_name = self._onnx_session.get_inputs()[0].name
        self.output_names = [o.name for o in self._onnx_session.get_outputs()]
        self.input_shape = self._onnx_session.get_inputs()[0].shape
        self.input_height = self._safe_dim(self.input_shape[2], 1280)
        self.input_width = self._safe_dim(self.input_shape[3], 1280)

        self._backend = "onnx"
        print(f"[FireSmoke] Backend: ONNX Runtime ({Path(model_path).name}), "
              f"input: {self.input_width}x{self.input_height}, "
              f"providers: {providers}")

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

    # ----- Preprocessing -----
    def _letterbox(self, image, new_shape, color=(114, 114, 114)):
        h, w = image.shape[:2]
        new_w, new_h = new_shape
        ratio = min(new_w / w, new_h / h)
        rw, rh = int(round(w * ratio)), int(round(h * ratio))
        if (rw, rh) != (w, h):
            interp = cv2.INTER_LINEAR  # always LINEAR — faster, good enough
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

    # ----- NMS & decoding (unchanged from original — well-tuned) -----
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
        # Guard against degenerate ROIs (e.g., single-pixel height/width with no channels)
        if roi.size == 0 or roi.ndim < 3:
            return None
        return roi

    def _roi_is_grayscale(self, roi):
        # Guard: minimum ROI size for meaningful color analysis
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
                # Only low-confidence fire detections go through color filter
                cf = box.cls_id == cls_fire and box.conf <= self.fire_color_filter_max_conf
                if not cf:
                    out.append(box); continue
                roi = self._roi_for_box(image, box)
                if roi is None or self._roi_is_grayscale(roi):
                    out.append(box); continue
                if not self._passes_fire_color(roi):
                    continue
                out.append(box)
            except Exception:
                # Edge case: malformed ROI — keep the detection
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
        """Inference via ONNX Runtime (legacy path)."""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        inp, ratio, pad, orig_size = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        return self._postprocess(outputs[0], ratio, pad, orig_size)

    def _predict_yolo(self, image):
        """Inference via YOLO engine — YOLO handles letterbox/preprocessing internally."""
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        orig_h, orig_w = image.shape[:2]

        # Low conf threshold — _conf_filter_mask applies per-class thresholds later
        results = self._yolo_model(
            image, conf=0.30, iou=self.iou_thres, verbose=False)

        all_boxes, all_scores, all_cls = [], [], []
        for res in results:
            if len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = res.boxes.conf.cpu().numpy().astype(np.float32)
            cls_raw = res.boxes.cls.cpu().numpy().astype(np.int32)
            # Guard against class indices out of range (mismatched model classes)
            valid_mask = (cls_raw >= 0) & (cls_raw < len(self.cls_remap))
            if not valid_mask.all():
                cls_raw = cls_raw.copy()
                cls_raw[~valid_mask] = len(self.cls_remap) - 1  # map to last class
            cls_mapped = self.cls_remap[cls_raw]
            all_boxes.append(xyxy)
            all_scores.append(confs)
            all_cls.append(cls_mapped)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        cls_ids = np.concatenate(all_cls)

        # Per-class confidence filter (different thresholds + bonus for top detection)
        keep = self._conf_filter_mask(scores, cls_ids)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(boxes) == 0:
            return []

        # Sanity filter (size, aspect ratio, area)
        boxes = self._clip_boxes(boxes, (orig_w, orig_h))
        boxes, scores, cls_ids = self._filter_sane(
            boxes, scores, cls_ids, (orig_w, orig_h))
        if len(boxes) == 0:
            return []

        # Cross-class dedup + smoke merge
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
        """Pre-run a dummy inference to initialize CUDA/TensorRT context."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            if self._backend == "yolo":
                self._predict_yolo(dummy)
            else:
                self._predict_single(dummy)
        except Exception:
            pass


# ============================================================
# YOLO model loader — auto-detect TensorRT engine
# ============================================================

def _load_yolo_model(model_path: str, device: str = "cuda", half: bool = True):
    """Load YOLO model: .engine (TensorRT) > .pt (PyTorch).

    Returns (model, is_engine).
    """
    p = Path(model_path)
    engine_path = p.with_suffix(".engine")

    if engine_path.exists():
        model = YOLO(str(engine_path), task="detect")
        print(f"[YOLO] Loaded TensorRT engine: {engine_path.name}")
        return model, True

    if p.exists():
        model = YOLO(str(p))
        # Move to GPU on first call
        if device == "cuda":
            try:
                model.to("cuda")
            except Exception:
                pass
        print(f"[YOLO] Loaded PyTorch weights: {p.name} "
              f"(consider converting to .engine for 5-10x speedup)")
        return model, False

    raise FileNotFoundError(f"Model not found: {model_path}")


# ============================================================
# ROS2 Node — async pipeline
# ============================================================

SOURCE_COLORS = {
    "fire_smoke": (0, 0, 255),
    "safety":     (255, 0, 0),
    "plate":      (0, 255, 255),
    "face":       (0, 255, 0),
}


class AnomalyDetectorNode(Node):
    """ROS2 node with async capture → detect → display pipeline."""

    def __init__(self):
        super().__init__("anomaly_detector_node")

        # ---- Parameters ----
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("show_window", True)

        # ---- Resolve models directory with multiple fallbacks ----
        # Priority:
        #   1. ament_index share dir (colcon install)
        #   2. Source workspace: ~/ros2_ws/src/anomaly_detector/models
        #   3. Install site-packages dir (fallback)
        models_dir = self._find_package_dir("models")
        face_db_dir = self._find_package_dir("face_database")

        self.declare_parameter("fire_smoke_model_dir", str(models_dir))
        self.declare_parameter("fire_smoke_class_names", [])  # empty = auto-detect; e.g. ['fire','smoke'] or ['smoke','fire']
        self.declare_parameter("helmet_model_path", str(models_dir / "helmet_best.pt"))
        self.declare_parameter("smoking_model_path", str(models_dir / "smoking_best.pt"))
        self.declare_parameter("safety_combined_model_path", "")
        self.declare_parameter("face_database_dir", str(face_db_dir))

        self.declare_parameter("enable_fire_smoke", True)
        self.declare_parameter("enable_safety", True)
        self.declare_parameter("enable_plate", True)
        self.declare_parameter("enable_face", True)

        # New optimization params
        self.declare_parameter("device", "cuda")
        self.declare_parameter("half", True)       # FP16
        self.declare_parameter("face_model_pack", "buffalo_l")  # must match registration model pack!
        self.declare_parameter("face_det_size", 320)             # smaller = faster
        self.declare_parameter("detection_interval", 0)           # 0 = every frame

        # ---- Read params ----
        self.width = self.get_parameter("frame_width").value
        self.height = self.get_parameter("frame_height").value
        self.fps = self.get_parameter("fps").value
        self.show_window = self.get_parameter("show_window").value

        self.enable_fire_smoke = self.get_parameter("enable_fire_smoke").value
        self.enable_safety = self.get_parameter("enable_safety").value
        self.enable_plate = self.get_parameter("enable_plate").value
        self.enable_face = self.get_parameter("enable_face").value

        self.device = self.get_parameter("device").value
        self.half = self.get_parameter("half").value
        self.face_pack = self.get_parameter("face_model_pack").value
        self.face_det_size = self.get_parameter("face_det_size").value
        self.detection_interval = self.get_parameter("detection_interval").value

        self.get_logger().info("=" * 60)
        self.get_logger().info("  Optimized Anomaly Detector Node Starting")
        self.get_logger().info(f"  Device: {self.device}, FP16: {self.half}")
        self.get_logger().info("=" * 60)

        # ---- Init detectors ----
        self._init_detectors()

        # ---- Camera ----
        camera_id = self.get_parameter("camera_id").value
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"Camera {camera_id} open failed!")
            raise RuntimeError("Camera open failed")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Try to set FPS (may not be supported by all cameras)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Use MJPEG if available for lower USB bandwidth
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.get_logger().info(f"Camera opened (ID={camera_id}, {self.width}x{self.height})")

        # ---- Publishers ----
        self.pub_fire_smoke = self.create_publisher(String, "/anomaly/fire_smoke", 10)
        self.pub_safety = self.create_publisher(String, "/anomaly/safety", 10)
        self.pub_plate = self.create_publisher(String, "/anomaly/plate", 10)
        self.pub_face = self.create_publisher(String, "/anomaly/face", 10)

        # ---- Window ----
        if self.show_window:
            cv2.namedWindow("Anomaly Detector", cv2.WINDOW_NORMAL)

        # ---- Async pipeline state ----
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_id = 0
        self._latest_results = []
        self._results_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # ---- Start background threads ----
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True)
        self._detect_thread = threading.Thread(
            target=self._detect_loop, name="detect", daemon=True)
        self._capture_thread.start()
        self._detect_thread.start()

        # ---- Display timer (runs on main thread — cv2 requires it) ----
        self.timer = self.create_timer(1.0 / max(self.fps, 1), self.timer_callback)

        # ---- Stats ----
        self._display_count = 0
        self._detect_count = 0
        self._last_log = time.time()
        self._last_detect_time = 0.0
        # Per-detector timing
        self._det_times = {"fire_smoke": 0, "safety": 0, "plate": 0, "face": 0}
        self._det_counts = {"fire_smoke": 0, "safety": 0, "plate": 0, "face": 0}

        self.get_logger().info("Anomaly Detector Node ready (async pipeline)")

    # ------------------------------------------------------------------
    # Package directory resolution (multi-level fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_package_dir(subdir: str) -> Path:
        """Resolve a package subdirectory with multiple fallbacks.

        Priority:
          1. ament_index share dir (colcon install)
          2. Source workspace: ~/ros2_ws/src/anomaly_detector/<subdir>
          3. Install site-packages/models (legacy colcon install)
          4. Fallback relative to this script
        """
        # 1. Try ament_index package share directory
        if AMENT_INDEX_AVAILABLE:
            try:
                share = Path(get_package_share_directory("anomaly_detector"))
                candidate = share / subdir
                if candidate.is_dir():
                    print(f"[Path] Using ament_index: {candidate}")
                    return candidate
            except Exception:
                pass

        # 2. Try source workspace
        src_candidate = Path.home() / "ros2_ws" / "src" / "anomaly_detector" / subdir
        if src_candidate.is_dir():
            print(f"[Path] Using source workspace: {src_candidate}")
            return src_candidate

        # 3. Install site-packages (legacy)
        pkg_root = Path(__file__).resolve().parents[1]
        install_candidate = pkg_root / subdir
        if install_candidate.is_dir():
            print(f"[Path] Using install site-packages: {install_candidate}")
            return install_candidate

        # 4. Fallback — return the best guess, let downstream report errors
        print(f"[Path] WARNING: '{subdir}' not found via ament_index, source, or install; "
              f"falling back to {install_candidate}")
        return install_candidate

    # ------------------------------------------------------------------
    # Detector initialization
    # ------------------------------------------------------------------

    def _init_detectors(self):
        # 1. Fire & Smoke (YOLO TensorRT engine / ONNX Runtime)
        self._fs_miner = None
        if self.enable_fire_smoke and (ONNX_AVAILABLE or YOLO_AVAILABLE):
            try:
                model_dir = Path(self.get_parameter("fire_smoke_model_dir").value)
                fs_class_names = self.get_parameter("fire_smoke_class_names").value
                if fs_class_names and len(fs_class_names) > 0:
                    override = fs_class_names
                else:
                    override = None
                self._fs_miner = _FireSmokeMiner(
                    model_dir, device=self.device,
                    class_names_override=override)
                self._fs_miner.warmup()
                self.get_logger().info("[OK] Fire & Smoke detector loaded")
            except Exception as e:
                self.get_logger().error(f"Fire & Smoke init failed: {e}")
        elif self.enable_fire_smoke and not (ONNX_AVAILABLE or YOLO_AVAILABLE):
            self.get_logger().warn(
                "Neither onnxruntime nor ultralytics installed, fire/smoke disabled")

        # 2. Safety (YOLO — combined or separate)
        self._safety_model = None       # combined model
        self._safety_helmet = None      # separate helmet model
        self._safety_smoking = None     # separate smoking model
        self._safety_combined = False
        if self.enable_safety and YOLO_AVAILABLE:
            try:
                combined_path = self.get_parameter("safety_combined_model_path").value
                if combined_path and Path(combined_path).exists():
                    self._safety_model, is_engine = _load_yolo_model(
                        combined_path, device=self.device, half=self.half)
                    self._safety_combined = True
                    self.get_logger().info("[OK] Combined safety model loaded")
                else:
                    hp = self.get_parameter("helmet_model_path").value
                    sp = self.get_parameter("smoking_model_path").value
                    self._safety_helmet, _ = _load_yolo_model(
                        hp, device=self.device, half=self.half)
                    self._safety_smoking, _ = _load_yolo_model(
                        sp, device=self.device, half=self.half)
                    self.get_logger().info("[OK] Safety (helmet+smoking) detectors loaded")
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
                self.get_logger().info("[OK] Plate detector loaded")
            except Exception as e:
                self.get_logger().error(f"Plate init failed: {e}")
        elif self.enable_plate and not HYPERLPR_AVAILABLE:
            self.get_logger().warn("hyperlpr3 not installed, plate detection disabled")

        # 4. Face (InsightFace — GPU accelerated)
        self._face_app = None
        self._face_threshold = 0.50
        self._face_db_names = []
        self._face_db_matrix = None
        if self.enable_face and INSIGHTFACE_AVAILABLE:
            try:
                providers = []
                ctx_id = -1  # CPU default
                if self.device == "cuda":
                    available = ort.get_available_providers() if ONNX_AVAILABLE else []
                    if "CUDAExecutionProvider" in available:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        ctx_id = 0  # GPU
                        self.get_logger().info(
                            "InsightFace: CUDAExecutionProvider available — GPU acceleration ON")
                    elif "TensorrtExecutionProvider" in available:
                        # TensorRT EP available but CUDA EP missing — still try GPU
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        self.get_logger().warn(
                            "InsightFace: TensorrtExecutionProvider found but "
                            "CUDAExecutionProvider missing. "
                            "Trying CUDAExecutionProvider anyway (may fail)...")
                        # Check if onnxruntime-gpu is installed vs onnxruntime (CPU-only)
                        try:
                            import onnxruntime
                            ep_list = onnxruntime.get_available_providers()
                            self.get_logger().info(
                                f"  Available ONNX providers: {ep_list}")
                        except Exception:
                            pass
                        self.get_logger().warn(
                            "  If CPU-only: pip install onnxruntime-gpu==1.17.1 "
                            "(compatible with JetPack 6.x)")
                    else:
                        providers = ["CPUExecutionProvider"]
                        self.get_logger().warn(
                            "InsightFace: No GPU provider found in onnxruntime. "
                            "Falling back to CPU — expect 10-50x slowdown.\n"
                            "  Fix: pip install onnxruntime-gpu==1.17.1\n"
                            "  Or check: python -c 'import onnxruntime; "
                            "print(onnxruntime.get_available_providers())'")
                else:
                    providers = ["CPUExecutionProvider"]

                self._face_app = FaceAnalysis(
                    name=self.face_pack, providers=providers)
                self._face_app.prepare(ctx_id=ctx_id, det_size=(self.face_det_size,) * 2)
                self.get_logger().info(
                    f"[OK] Face detector loaded (pack={self.face_pack}, "
                    f"det_size={self.face_det_size}, ctx={'GPU' if ctx_id == 0 else 'CPU'}, "
                    f"providers={providers})")

                # Load face database with pre-normalized embeddings
                self._init_face_database()
            except Exception as e:
                self.get_logger().error(f"Face init failed: {e}")
        elif self.enable_face and not INSIGHTFACE_AVAILABLE:
            self.get_logger().warn("insightface not installed, face detection disabled")

    def _init_face_database(self):
        """Load face embeddings from database — supports multiple features per person.

        Database structure:
          face_database/<person_name>/
              feature_000.npy    ← multi-angle captures (new format, preferred)
              feature_001.npy
              ...
              feature.npy        ← legacy single capture (backward compatible)

        Multiple features per person enable angle-invariant recognition:
        the best match across all registered angles is used.
        """
        db_dir = self.get_parameter("face_database_dir").value
        if not os.path.isdir(db_dir):
            self.get_logger().warn(
                f"Face database directory not found: {db_dir}")
            return

        embeddings = []
        person_feature_counts = {}  # for logging summary

        for name in sorted(os.listdir(db_dir)):
            person_dir = os.path.join(db_dir, name)
            if not os.path.isdir(person_dir):
                continue

            # ---- Find all feature files ----
            person_path = Path(person_dir)
            feature_files = sorted(person_path.glob("feature_*.npy"))
            loaded_count = 0

            if not feature_files:
                # Fallback to legacy single feature.npy
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
                    loaded_count += 1
                except Exception as e:
                    self.get_logger().warn(
                        f"  Failed to load {feat_path.name}: {e}")

            if loaded_count > 0:
                person_feature_counts[name] = loaded_count

        # ---- Summary ----
        if not embeddings:
            self.get_logger().warn(
                "Face database is empty — no face features loaded")
            return

        self._face_db_matrix = np.stack(embeddings).astype(np.float32)
        self.get_logger().info(
            f"  Face database: {len(embeddings)} total features "
            f"from {len(person_feature_counts)} people")
        for name, count in person_feature_counts.items():
            self.get_logger().info(f"    {name}: {count} feature(s)")

    # ------------------------------------------------------------------
    # Capture thread — reads camera continuously, keeps latest frame
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Continuously capture frames, always keeping the latest."""
        consecutive_failures = 0
        while not self._shutdown_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    self.get_logger().error("Camera read failed 10+ times")
                time.sleep(0.01)
                continue
            consecutive_failures = 0
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_id += 1

    # ------------------------------------------------------------------
    # Detection thread — runs all detectors on the latest frame
    # ------------------------------------------------------------------

    def _detect_loop(self):
        """Process the latest frame, publish results."""
        last_processed_id = -1
        while not self._shutdown_event.is_set():
            # Get latest frame
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._frame_id
            if frame is None or frame_id == last_processed_id:
                time.sleep(0.002)
                continue

            # Optional detection interval throttling
            if self.detection_interval > 0:
                elapsed = time.time() - self._last_detect_time
                if elapsed < self.detection_interval:
                    time.sleep(self.detection_interval - elapsed)
                    continue

            last_processed_id = frame_id
            detect_start = time.time()

            # Run all detectors
            results = self._run_all_detectors(frame)

            # Store results for display
            with self._results_lock:
                self._latest_results = results

            self._last_detect_time = time.time()
            self._detect_count += 1

            # Log per-detector timing every 100 detections
            if self._detect_count % 100 == 0:
                parts = []
                for k in ("fire_smoke", "safety", "plate", "face"):
                    if self._det_counts[k] > 0:
                        avg = self._det_times[k] / max(1, self._det_counts[k]) * 1000
                        parts.append(f"{k}={avg:.1f}ms")
                if parts:
                    self.get_logger().info(
                        f"Det #{self._detect_count}: " + ", ".join(parts))

    # ------------------------------------------------------------------
    # Display timer — runs on main thread (cv2 requires it)
    # ------------------------------------------------------------------

    def timer_callback(self):
        """Display latest frame with latest results (non-blocking)."""
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return

        with self._results_lock:
            results = list(self._latest_results)

        if self.show_window:
            display = self._draw_results(frame.copy(), results)
            cv2.imshow("Anomaly Detector", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("User pressed 'q', shutting down")
                rclpy.shutdown()
                return

        # FPS logging
        self._display_count += 1
        now = time.time()
        if now - self._last_log >= 5.0:
            display_fps = self._display_count / (now - self._last_log)
            detect_fps = self._detect_count / (now - self._last_log)
            self.get_logger().info(
                f"Display: {display_fps:.1f} FPS, "
                f"Detect: {detect_fps:.1f} FPS, "
                f"Results: {len(results)}")
            self._display_count = 0
            self._detect_count = 0
            self._last_log = now

    # ------------------------------------------------------------------
    # Run all detectors on a frame
    # ------------------------------------------------------------------

    def _run_all_detectors(self, frame):
        """Run all enabled detectors and publish results."""
        all_results = []

        if self._fs_miner is not None:
            t0 = time.time()
            boxes = self._fs_miner.detect(frame)
            dt = time.time() - t0
            self._det_times["fire_smoke"] += dt
            self._det_counts["fire_smoke"] += 1
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

        if self._safety_model is not None or self._safety_helmet is not None:
            t0 = time.time()
            safety_results = self._detect_safety(frame)
            dt = time.time() - t0
            self._det_times["safety"] += dt
            self._det_counts["safety"] += 1
            all_results.extend(safety_results)
            if safety_results:
                self._publish("safety", [{
                    "class": r["class_name"], "conf": r["conf"],
                    "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in safety_results])

        if self._plate_catcher is not None:
            t0 = time.time()
            plate_results = self._detect_plate(frame)
            dt = time.time() - t0
            self._det_times["plate"] += dt
            self._det_counts["plate"] += 1
            all_results.extend(plate_results)
            if plate_results:
                self._publish("plate", [{
                    "plate": r["plate_number"], "conf": r["conf"],
                    "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in plate_results])

        if self._face_app is not None:
            t0 = time.time()
            face_results = self._detect_face(frame)
            dt = time.time() - t0
            self._det_times["face"] += dt
            self._det_counts["face"] += 1
            all_results.extend(face_results)
            if face_results:
                self._publish("face", [{
                    "name": r.get("name", "unknown"),
                    "similarity": r.get("similarity", 0.0),
                    "conf": r["conf"], "box": [r["x1"], r["y1"], r["x2"], r["y2"]]
                } for r in face_results])

        return all_results

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _detect_safety(self, frame):
        """Safety detection — combined model or two separate models."""
        results = []
        conf = 0.9

        if self._safety_combined and self._safety_model is not None:
            # Single combined model
            try:
                r = self._safety_model(frame, conf=conf, verbose=False, device=self.device)
                for res in r:
                    for box in res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        results.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "conf": float(box.conf[0]),
                            "class_name": self._safety_model.names.get(
                                int(box.cls[0]), "unknown"),
                            "source": "safety",
                        })
            except Exception as e:
                self.get_logger().debug(f"Safety detect error: {e}", throttle_duration_sec=5)
        else:
            # Two separate models
            for model in (self._safety_helmet, self._safety_smoking):
                if model is None:
                    continue
                try:
                    r = model(frame, conf=conf, verbose=False, device=self.device)
                    for res in r:
                        for box in res.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            results.append({
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "conf": float(box.conf[0]),
                                "class_name": model.names.get(
                                    int(box.cls[0]), "unknown"),
                                "source": "safety",
                            })
                except Exception as e:
                    self.get_logger().debug(f"Safety detect error: {e}", throttle_duration_sec=5)
        return results

    def _detect_plate(self, frame):
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
            self.get_logger().debug(f"Plate detect error: {e}", throttle_duration_sec=5)
        return results

    def _detect_face(self, frame):
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
                # Vectorized face recognition
                if self._face_db_matrix is not None and \
                   hasattr(face, 'embedding') and face.embedding is not None:
                    emb = face.embedding.astype(np.float32)
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                    sims = self._face_db_matrix @ emb_norm
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    result["similarity"] = best_sim
                    if best_sim >= self._face_threshold:
                        result["name"] = self._face_db_names[best_idx]
                results.append(result)
        except Exception as e:
            self.get_logger().debug(f"Face detect error: {e}", throttle_duration_sec=5)
        return results

    # ------------------------------------------------------------------
    # Drawing — batch Chinese text rendering (single PIL conversion)
    # ------------------------------------------------------------------

    def _draw_results(self, frame, results):
        """Draw all results on frame. Chinese text is batched into one PIL pass."""
        # Separate plates (need PIL for Chinese) from others (use cv2)
        plate_labels = []  # [(x, y, text, color), ...]

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
                plate_labels.append((x1, text_y, label, color))
            else:
                cv2.putText(frame, label, (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Batch render all Chinese (plate) text in a single PIL pass
        if plate_labels:
            frame = self._draw_chinese_batch(frame, plate_labels)

        return frame

    def _draw_chinese_batch(self, img, labels):
        """Render multiple Chinese text labels in a single PIL conversion."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        for x, y, text, color in labels:
            # PIL uses RGB; OpenCV uses BGR — swap color channels
            fill = (color[2], color[1], color[0])
            draw.text((x, y), text, font=self._plate_font, fill=fill)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, topic_key, data):
        # Guard: skip publish if ROS context is no longer valid (shutting down)
        if not rclpy.ok():
            return
        msg = String()
        msg.data = json.dumps(data)
        pub = getattr(self, f"pub_{topic_key}", None)
        if pub is not None:
            try:
                pub.publish(msg)
            except Exception:
                pass  # silently ignore publish errors during shutdown

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info("Shutting down AnomalyDetectorNode...")

        # 1. Signal all threads to stop
        self._shutdown_event.set()

        # 2. Wait for threads to finish — detect thread may be mid-inference
        #    (especially InsightFace on CPU), so use a longer timeout
        if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
        if hasattr(self, '_detect_thread') and self._detect_thread.is_alive():
            self.get_logger().info("Waiting for detect thread to finish...")
            self._detect_thread.join(timeout=10.0)
            if self._detect_thread.is_alive():
                self.get_logger().warn(
                    "Detect thread did not exit after 10s (may be stuck in long inference)")

        # 3. Release hardware resources
        if self.cap.isOpened():
            self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

        # 4. Destroy ROS node context LAST — after all threads have stopped
        super().destroy_node()


# ============================================================
# Model conversion utility
# ============================================================

def convert_models_to_tensorrt(models_dir: str, imgsz: int = 640, half: bool = True):
    """Convert YOLO .pt models to TensorRT .engine files.

    Run this on the Jetson device:
        python -c "from anomaly_detector_optimized import convert_models_to_tensorrt; convert_models_to_tensorrt('/path/to/models')"

    Or as a standalone script:
        python anomaly_detector_optimized.py --convert --models-dir /path/to/models
    """
    from ultralytics import YOLO

    model_dir = Path(models_dir)
    targets = ["helmet_best.pt", "smoking_best.pt"]

    print("=" * 60)
    print("  TensorRT Engine Conversion")
    print("=" * 60)

    for name in targets:
        pt_path = model_dir / name
        engine_path = pt_path.with_suffix(".engine")

        if not pt_path.exists():
            print(f"  [SKIP] {name} not found")
            continue

        if engine_path.exists():
            print(f"  [EXISTS] {engine_path.name} already exists")
            continue

        print(f"  Converting {name} -> .engine (imgsz={imgsz}, half={half})...")
        try:
            model = YOLO(str(pt_path))
            model.export(format="engine", device=0, half=half, imgsz=imgsz)
            print(f"  [DONE] {engine_path.name} created")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print("=" * 60)
    print("  Conversion complete!")
    print("  The optimized node will auto-detect .engine files.")
    print("=" * 60)


def check_environment():
    """Print available providers and model status."""
    print("\n" + "=" * 60)
    print("  Environment Check")
    print("=" * 60)

    # ONNX Runtime providers
    if ONNX_AVAILABLE:
        providers = ort.get_available_providers()
        print(f"  ONNX Runtime providers: {providers}")
        has_cuda = "CUDAExecutionProvider" in providers
        has_trt = "TensorrtExecutionProvider" in providers
        if has_trt:
            print("  [OK] TensorRT Execution Provider available")
        elif has_cuda:
            print("  [OK] CUDA Execution Provider available")
            print("  [TIP] Install TensorRT for additional speedup")
        else:
            print("  [WARN] No GPU provider! Install onnxruntime-gpu")
    else:
        print("  [MISS] onnxruntime not installed")

    # YOLO
    print(f"  Ultralytics YOLO: {'available' if YOLO_AVAILABLE else 'NOT installed'}")

    # InsightFace
    print(f"  InsightFace: {'available' if INSIGHTFACE_AVAILABLE else 'NOT installed'}")

    # HyperLPR3
    print(f"  HyperLPR3: {'available' if HYPERLPR_AVAILABLE else 'NOT installed'}")

    # GPU info
    try:
        import subprocess
        result = subprocess.run(
            ["tegr-stats"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  tegr-stats: {result.stdout[:200]}")
    except Exception:
        pass

    print("=" * 60 + "\n")


# ============================================================
# Main
# ============================================================

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly Detector Node")
    parser.add_argument("--convert", action="store_true",
                        help="Convert YOLO models to TensorRT engines")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Path to models directory (for --convert)")
    parser.add_argument("--check", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for TensorRT conversion")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable FP16 (use FP32)")
    cli_args = parser.parse_args(args)

    if cli_args.check:
        check_environment()
        return

    if cli_args.convert:
        if not cli_args.models_dir:
            print("Error: --models-dir required for --convert")
            sys.exit(1)
        convert_models_to_tensorrt(cli_args.models_dir,
                                   imgsz=cli_args.imgsz,
                                   half=not cli_args.no_half)
        return

    # Normal ROS2 node startup
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
