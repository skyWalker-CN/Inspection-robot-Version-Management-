#!/usr/bin/env python3
"""
Fire & Smoke Detector using ONNX Runtime.
Ported from fire_smoke_detect package.
"""

import math
import cv2
import numpy as np
import onnxruntime as ort
from numpy import ndarray
from pathlib import Path


class BoundingBox:
    __slots__ = ("x1", "y1", "x2", "y2", "cls_id", "conf")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, cls_id: int, conf: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cls_id = cls_id
        self.conf = conf


class FireSmokeDetector:
    """ONNX Runtime detector for fire / smoke / fire_extinguisher."""

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
    use_tta_view_filter = False
    tta_view_filter_max_conf = 0.0
    tta_view_iou_thresh = 0.5

    def __init__(self, model_dir: Path) -> None:
        model_path = model_dir / "weights.onnx"
        print("ORT version:", ort.__version__)
        try:
            ort.preload_dlls()
            print("onnxruntime.preload_dlls() success")
        except Exception as e:
            print(f"preload_dlls failed: {e}")
        print("ORT available providers:", ort.get_available_providers())
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            print("Created ORT session with CUDA provider list")
        except Exception as e:
            print(f"CUDA session creation failed, falling back to CPU: {e}")
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        print("ORT session providers:", self.session.get_providers())
        model_class_order = self._read_model_class_order()
        if model_class_order is None:
            model_class_order = list(self._model_class_order)
            print(f"cls order: no usable ONNX metadata, FALLBACK {model_class_order}")
        else:
            print(f"cls order: from ONNX metadata {model_class_order}")
        self.cls_remap = np.array(
            [self.class_names.index(n) for n in model_class_order],
            dtype=np.int32,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self._safe_dim(self.input_shape[2], default=1280)
        self.input_width = self._safe_dim(self.input_shape[3], default=1280)
        self.use_tta = True
        print(f"ONNX model loaded from: {model_path}")
        print(f"ONNX providers: {self.session.get_providers()}")

    @staticmethod
    def _safe_dim(value, default: int) -> int:
        return value if isinstance(value, int) and value > 0 else default

    def _read_model_class_order(self) -> list | None:
        try:
            import ast
            meta = self.session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(meta["names"])
            if isinstance(names, dict):
                order = [str(names[i]) for i in sorted(names)]
            else:
                order = [str(n) for n in names]
        except Exception as e:
            print(f"cls order: could not read ONNX names metadata ({e})")
            return None
        if sorted(order) != sorted(self.class_names):
            print(f"cls order: ONNX names {order} do not match expected classes {self.class_names}")
            return None
        return order

    def _letterbox(self, image: ndarray, new_shape: tuple, color=(114, 114, 114)):
        h, w = image.shape[:2]
        new_w, new_h = new_shape
        ratio = min(new_w / w, new_h / h)
        resized_w = int(round(w * ratio))
        resized_h = int(round(h * ratio))
        if (resized_w, resized_h) != (w, h):
            interp = cv2.INTER_CUBIC if ratio > 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (resized_w, resized_h), interpolation=interp)
        dw = (new_w - resized_w) / 2.0
        dh = (new_h - resized_h) / 2.0
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=color)
        return padded, ratio, (dw, dh)

    def _preprocess(self, image: ndarray):
        orig_h, orig_w = image.shape[:2]
        img, ratio, pad = self._letterbox(image, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img, ratio, pad, (orig_w, orig_h)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, image_size: tuple) -> np.ndarray:
        w, h = image_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

    @staticmethod
    def _hard_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
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

    def _per_class_hard_nms(self, boxes, scores, cls_ids, iou_thresh):
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

    def _cross_class_dedup_op(self, boxes, scores, cls_ids, iou_thresh):
        n = len(boxes)
        if n <= 1:
            return boxes, scores, cls_ids
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        cls_ids = np.asarray(cls_ids, dtype=np.int32)
        areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) *
                 np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))
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
        keep_idx = np.array(keep, dtype=np.intp)
        return boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]

    def _merge_smoke_boxes(self, boxes, scores, cls_ids):
        smoke_cls = self.class_names.index("smoke")
        smoke_idx = np.where(cls_ids == smoke_cls)[0]
        if len(smoke_idx) <= 1:
            return boxes, scores, cls_ids
        sb = boxes[smoke_idx].astype(np.float32).tolist()
        ss = scores[smoke_idx].astype(np.float32).tolist()
        merged_any = True
        while merged_any and len(sb) > 1:
            merged_any = False
            for i in range(len(sb)):
                for j in range(i + 1, len(sb)):
                    a, b = sb[i], sb[j]
                    ix1 = max(a[0], b[0])
                    iy1 = max(a[1], b[1])
                    ix2 = min(a[2], b[2])
                    iy2 = min(a[3], b[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                    smaller = min(area_a, area_b)
                    if inter / (smaller + 1e-7) >= self.smoke_merge_overlap:
                        sb[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                 max(a[2], b[2]), max(a[3], b[3])]
                        ss[i] = max(ss[i], ss[j])
                        del sb[j]
                        del ss[j]
                        merged_any = True
                        break
                if merged_any:
                    break
        other = cls_ids != smoke_cls
        new_boxes = np.concatenate([boxes[other].astype(np.float32),
                                    np.array(sb, dtype=np.float32).reshape(-1, 4)])
        new_scores = np.concatenate([scores[other].astype(np.float32),
                                     np.array(ss, dtype=np.float32)])
        new_cls = np.concatenate([cls_ids[other].astype(np.int32),
                                  np.full(len(sb), smoke_cls, dtype=np.int32)])
        return new_boxes, new_scores, new_cls

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

    def _filter_sane_boxes(self, boxes, scores, cls_ids, orig_size):
        if len(boxes) == 0:
            return boxes, scores, cls_ids
        orig_w, orig_h = orig_size
        image_area = float(orig_w * orig_h)
        keep = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if bw < self.min_side or bh < self.min_side:
                continue
            area = bw * bh
            if area < self.min_box_area:
                continue
            if area > 0.95 * image_area:
                continue
            ar = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
            if ar > self.max_aspect_ratio:
                continue
            keep.append(i)
        if not keep:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32))
        k = np.array(keep, dtype=np.intp)
        return boxes[k], scores[k], cls_ids[k]

    def _per_view_pipeline(self, boxes, scores, cls_ids):
        if len(boxes) > 1:
            keep = self._per_class_hard_nms(boxes, scores, cls_ids, self.iou_thres)
            boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(scores) > self.max_det:
            top = np.argsort(-scores)[:self.max_det]
            boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]
        if len(boxes) > 1:
            boxes, scores, cls_ids = self._cross_class_dedup_op(
                boxes, scores, cls_ids, self.cross_iou_thresh)
        if len(boxes) > 1:
            boxes, scores, cls_ids = self._merge_smoke_boxes(boxes, scores, cls_ids)
        return boxes, scores, cls_ids

    @staticmethod
    def _roi_for_box(image: np.ndarray, box: BoundingBox) -> np.ndarray | None:
        h, w = image.shape[:2]
        x1 = max(0, int(math.floor(box.x1)))
        y1 = max(0, int(math.floor(box.y1)))
        x2 = min(w, int(math.ceil(box.x2)))
        y2 = min(h, int(math.ceil(box.y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = image[y1:y2, x1:x2]
        return roi if roi.size else None

    def _roi_is_near_grayscale(self, roi: np.ndarray) -> bool:
        mx = roi.max(axis=2).astype(np.float32)
        mn = roi.min(axis=2).astype(np.float32)
        sat = (mx - mn) / (mx + 1e-6)
        return float(sat.mean()) < self.color_filter_min_saturation

    @staticmethod
    def _passes_fire_color(roi: np.ndarray) -> bool:
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
    def _passes_fire_ext_red_color(roi: np.ndarray) -> bool:
        blue = roi[:, :, 0].astype(np.float32)
        green = roi[:, :, 1].astype(np.float32)
        red = roi[:, :, 2].astype(np.float32)
        red_dom = float(np.mean((red > green + 10.0) & (red > blue + 10.0)))
        if red_dom >= 0.03:
            return True
        if (float(np.mean(red)) - float(np.mean(green))) >= 0.0 and float(np.mean(red)) >= 50.0:
            return True
        return False

    def _remove_edge_low_conf(self, results: list, orig_size: tuple) -> list:
        if not self.use_edge_filter or self.edge_filter_max_conf <= 0.0 or not results:
            return results
        w, h = orig_size
        tol = self.edge_tol
        out = []
        for b in results:
            on_edge = (b.x1 <= tol or b.y1 <= tol or b.x2 >= w - 1 - tol or b.y2 >= h - 1 - tol)
            if on_edge and b.conf <= self.edge_filter_max_conf:
                continue
            out.append(b)
        return out

    def _filter_low_conf_by_color(self, image: np.ndarray, results: list) -> list:
        if not results:
            return results
        cls_fire = self.class_names.index("fire")
        cls_ext = self.class_names.index("fire extinguisher")
        out = []
        for box in results:
            check_fire = box.cls_id == cls_fire and box.conf <= self.fire_color_filter_max_conf
            check_ext = box.cls_id == cls_ext and box.conf <= self.fire_ext_color_filter_max_conf
            if not check_fire and not check_ext:
                out.append(box)
                continue
            roi = self._roi_for_box(image, box)
            if roi is None or self._roi_is_near_grayscale(roi):
                out.append(box)
                continue
            if check_fire and not self._passes_fire_color(roi):
                continue
            if check_ext and not self._passes_fire_ext_red_color(roi):
                continue
            out.append(box)
        return out

    @staticmethod
    def _build_results(boxes, scores, cls_ids) -> list:
        results = []
        for box, conf, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(BoundingBox(
                x1=int(math.floor(x1)), y1=int(math.floor(y1)),
                x2=int(math.ceil(x2)), y2=int(math.ceil(y2)),
                cls_id=int(cls_id), conf=float(conf)))
        return results

    def _decode_final_dets(self, preds, ratio, pad, orig_size):
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.ndim != 2 or preds.shape[1] < 6:
            raise ValueError(f"Unexpected ONNX final-det output shape: {preds.shape}")
        boxes = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        cls_ids = preds[:, 5].astype(np.int32)
        cls_ids = self.cls_remap[cls_ids]
        keep = self._conf_filter_mask(scores, cls_ids)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(boxes) == 0:
            return []
        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes = self._clip_boxes(boxes, orig_size)
        boxes, scores, cls_ids = self._filter_sane_boxes(boxes, scores, cls_ids, orig_size)
        if len(boxes) == 0:
            return []
        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _decode_raw_yolo(self, preds, ratio, pad, orig_size):
        if preds.ndim != 3 or preds.shape[0] != 1:
            raise ValueError(f"Unexpected raw ONNX output shape: {preds.shape}")
        preds = preds[0]
        if preds.shape[0] <= 16 and preds.shape[1] > preds.shape[0]:
            preds = preds.T
        if preds.ndim != 2 or preds.shape[1] < 5:
            raise ValueError(f"Unexpected raw output shape: {preds.shape}")
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
        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes = self._clip_boxes(boxes, orig_size)
        boxes, scores, cls_ids = self._filter_sane_boxes(boxes, scores, cls_ids, orig_size)
        if len(boxes) == 0:
            return []
        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _postprocess(self, output, ratio, pad, orig_size):
        if output.ndim == 2 and output.shape[1] >= 6:
            return self._decode_final_dets(output, ratio, pad, orig_size)
        if output.ndim == 3 and output.shape[0] == 1 and output.shape[2] == 6:
            return self._decode_final_dets(output, ratio, pad, orig_size)
        return self._decode_raw_yolo(output, ratio, pad, orig_size)

    def _predict_single(self, image: np.ndarray) -> list:
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input is not numpy array: {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape={image.shape}")
        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got shape={image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        input_tensor, ratio, pad, orig_size = self._preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self._postprocess(outputs[0], ratio, pad, orig_size)

    def _predict_tta(self, image: np.ndarray) -> list:
        boxes_orig = self._predict_single(image)
        flipped = cv2.flip(image, 1)
        boxes_flip = self._predict_single(flipped)
        w = image.shape[1]
        boxes_flip = [
            BoundingBox(x1=w - b.x2, y1=b.y1, x2=w - b.x1, y2=b.y2,
                        cls_id=b.cls_id, conf=b.conf)
            for b in boxes_flip
        ]
        all_boxes = boxes_orig + boxes_flip
        if not all_boxes:
            return []
        coords = np.array([[b.x1, b.y1, b.x2, b.y2] for b in all_boxes], dtype=np.float32)
        scores = np.array([b.conf for b in all_boxes], dtype=np.float32)
        cls_ids = np.array([b.cls_id for b in all_boxes], dtype=np.int32)
        hard_keep = self._per_class_hard_nms(coords, scores, cls_ids, self.iou_thres)
        if len(hard_keep) == 0:
            return []
        if len(hard_keep) > self.max_det:
            top = np.argsort(-scores[hard_keep])[:self.max_det]
            hard_keep = hard_keep[top]
        kept_coords = coords[hard_keep]
        kept_cls = cls_ids[hard_keep]
        boosted = scores[hard_keep]
        if len(kept_coords) > 1:
            kept_coords, boosted, kept_cls = self._cross_class_dedup_op(
                kept_coords, boosted, kept_cls, self.cross_iou_thresh)
        if len(kept_coords) > 1:
            kept_coords, boosted, kept_cls = self._merge_smoke_boxes(
                kept_coords, boosted, kept_cls)
        return [
            BoundingBox(
                x1=int(math.floor(kept_coords[j, 0])),
                y1=int(math.floor(kept_coords[j, 1])),
                x2=int(math.ceil(kept_coords[j, 2])),
                y2=int(math.ceil(kept_coords[j, 3])),
                cls_id=int(kept_cls[j]),
                conf=float(boosted[j]),
            ) for j in range(len(kept_coords))
        ]

    def detect(self, image: np.ndarray) -> list:
        """Run fire/smoke detection on a single image. Returns list of BoundingBox."""
        try:
            if self.use_tta:
                boxes = self._predict_tta(image)
            else:
                boxes = self._predict_single(image)
            boxes = self._filter_low_conf_by_color(image, boxes)
            boxes = self._remove_edge_low_conf(boxes, (image.shape[1], image.shape[0]))
        except Exception as e:
            print(f"FireSmokeDetector error: {e}")
            boxes = []
        return boxes