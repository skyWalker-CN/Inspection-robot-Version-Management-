#!/usr/bin/env python3
"""
Integrated anomaly detection node for Jetson Orin NX + ROS2 Humble.

Functions:
  - fire / smoke detection with ONNX Runtime
  - face detection / optional recognition with InsightFace
  - license plate detection with HyperLPR3
  - helmet and smoking detection with Ultralytics YOLO

The node reads the camera only once and schedules heavy models at different
intervals so the display and ROS publication stay responsive on edge hardware.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 添加anaconda环境路径以正确导入torch
sys.path.append('/home/jetson/anaconda3/envs/test/lib/python3.10/site-packages/')

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from base_interfaces.msg import AnomalyDetection, AnomalyList
except Exception as exc:  # pragma: no cover - only used before ROS workspace is sourced
    raise RuntimeError(
        "Cannot import base_interfaces. Run `source install/setup.bash` after colcon build."
    ) from exc

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


# 使用绝对路径 ~/ros2_ws/src/example_python/models_new/
MODELS_DIR = Path.home() / "ros2_ws" / "src" / "example_python" / "models_new"
DEFAULT_FIRE_MODEL = MODELS_DIR / "fire_smoke_best.pt"
DEFAULT_HELMET_MODEL = MODELS_DIR / "helmet_best.pt"
DEFAULT_SMOKING_MODEL = MODELS_DIR / "smoking_best.pt"
DEFAULT_FACE_DB = MODELS_DIR.parent / "face_detect" / "face_database"


@dataclass(slots=True)
class Detection:
    kind: str
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    source: str
    extra: dict[str, Any] = field(default_factory=dict)


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def iou_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.intp)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = max(0.0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        area_r = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        iou = inter / (area_i + area_r - inter + 1e-7)
        order = rest[iou <= iou_thresh]
    return np.asarray(keep, dtype=np.intp)


class LatestFrameCamera:
    def __init__(self, camera_id: int, width: int, height: int, fps: int) -> None:
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera open failed: {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stopped = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class FireSmokeYoloDetector:
    """火烟雾检测器，使用 Ultralytics YOLO 加载 .pt 模型"""

    def __init__(
        self,
        model_path: Path,
        device: str,
        imgsz: int,
        half: bool,
        conf: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        if not model_path.exists():
            raise FileNotFoundError(f"火烟雾模型不存在: {model_path}")

        self.device = device
        self.imgsz = imgsz
        self.half = half and device != "cpu"
        self.conf = conf
        self.model = YOLO(str(model_path))
        if device != "cpu":
            self.model.to(device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        result = results[0]
        names = result.names or {}
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            label = str(names.get(cls_id, "fire_smoke"))
            detections.append(
                Detection(
                    "fire_smoke",
                    label,
                    float(box.conf[0].detach().cpu().item()),
                    clip_box((x1, y1, x2, y2), frame.shape[1], frame.shape[0]),
                    "fire_smoke",
                )
            )
        return detections


class SafetyYoloDetector:
    def __init__(
        self,
        helmet_model_path: Path,
        smoking_model_path: Path,
        device: str,
        imgsz: int,
        half: bool,
        helmet_conf: float,
        smoking_conf: float,
    ) -> None:
        from ultralytics import YOLO

        self.device = device
        self.imgsz = imgsz
        self.half = half and device != "cpu"
        self.helmet_conf = helmet_conf
        self.smoking_conf = smoking_conf
        self.helmet_model = YOLO(str(helmet_model_path))
        self.smoking_model = YOLO(str(smoking_model_path))
        if device != "cpu":
            self.helmet_model.to(device)
            self.smoking_model.to(device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for source, model, conf in (
            ("helmet", self.helmet_model, self.helmet_conf),
            ("smoking", self.smoking_model, self.smoking_conf),
        ):
            results = model.predict(
                frame,
                conf=conf,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
            result = results[0]
            names = result.names or {}
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                cls_id = int(box.cls[0].detach().cpu().item())
                label = str(names.get(cls_id, source))
                detections.append(
                    Detection(source, label, float(box.conf[0].detach().cpu().item()), clip_box((x1, y1, x2, y2), frame.shape[1], frame.shape[0]), source)
                )
        return detections


class FaceDetector:
    def __init__(self, database_dir: Path, threshold: float, det_size: int) -> None:
        from insightface.app import FaceAnalysis

        self.threshold = threshold
        self.database = self._load_database(database_dir)
        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        except Exception:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    @staticmethod
    def _load_database(database_dir: Path) -> dict[str, np.ndarray]:
        database: dict[str, np.ndarray] = {}
        if not database_dir.exists():
            return database
        for person_dir in database_dir.iterdir():
            feature_path = person_dir / "feature.npy"
            if person_dir.is_dir() and feature_path.exists():
                database[person_dir.name] = np.load(feature_path)
        return database

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for face in self.app.get(frame):
            x1, y1, x2, y2 = clip_box(tuple(face.bbox.astype(float)), frame.shape[1], frame.shape[0])
            label = "face"
            sim = 0.0
            if self.database and getattr(face, "embedding", None) is not None:
                for name, embedding in self.database.items():
                    score = self._cosine(face.embedding, embedding)
                    if score > sim:
                        sim = score
                        label = name
                if sim < self.threshold:
                    label = "unknown"
            detections.append(
                Detection("face", label, float(getattr(face, "det_score", sim or 1.0)), (x1, y1, x2, y2), "face", {"similarity": sim})
            )
        return detections


class PlateDetector:
    def __init__(self) -> None:
        from hyperlpr3 import LicensePlateCatcher

        self.catcher = LicensePlateCatcher()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for plate in self.catcher(frame):
            number = str(plate[0])
            score = float(plate[1])
            x1, y1, x2, y2 = clip_box(tuple(plate[3]), frame.shape[1], frame.shape[0])
            detections.append(Detection("plate", number, score, (x1, y1, x2, y2), "plate"))
        return detections


class IntegratedAnomalyNode(Node):
    COLORS = {
        "fire": (0, 0, 255),
        "smoke": (0, 180, 255),
        "fire extinguisher": (255, 80, 0),
        "face": (0, 255, 0),
        "plate": (255, 0, 255),
        "helmet": (255, 180, 0),
        "smoking": (0, 80, 255),
    }

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("integrated_anomaly_detector")
        self.args = args
        self.camera = LatestFrameCamera(args.camera_id, args.width, args.height, args.camera_fps)
        self.bridge = CvBridge() if CvBridge is not None else None
        self.anomaly_pub = self.create_publisher(AnomalyList, args.anomaly_topic, 10)
        self.image_pub = self.create_publisher(Image, args.image_topic, 10) if self.bridge else None
        self.detectors: list[tuple[str, int, Any]] = []
        self.cached: dict[str, list[Detection]] = {}
        self.frame_index = 0
        self.last_log = time.time()
        self.last_fps_time = time.time()
        self.loop_fps = 0.0

        self._load_detectors()
        self.timer = self.create_timer(max(0.001, 1.0 / args.loop_fps), self.tick)
        if args.show:
            cv2.namedWindow("Integrated Anomaly Detection", cv2.WINDOW_NORMAL)
        self.get_logger().info("Integrated anomaly detector started.")

    def _load_detectors(self) -> None:
        if self.args.enable_fire:
            self.detectors.append(
                (
                    "fire_smoke",
                    self.args.fire_interval,
                    FireSmokeOnnxDetector(Path(self.args.fire_model), self.args.fire_conf, self.args.iou),
                )
            )
            self.cached["fire_smoke"] = []
        if self.args.enable_safety:
            self.detectors.append(
                (
                    "safety",
                    self.args.safety_interval,
                    SafetyYoloDetector(
                        Path(self.args.helmet_model),
                        Path(self.args.smoking_model),
                        self.args.device,
                        self.args.yolo_imgsz,
                        self.args.yolo_half,
                        self.args.helmet_conf,
                        self.args.smoking_conf,
                    ),
                )
            )
            self.cached["safety"] = []
        if self.args.enable_face:
            self.detectors.append(
                ("face", self.args.face_interval, FaceDetector(Path(self.args.face_db), self.args.face_threshold, self.args.face_det_size))
            )
            self.cached["face"] = []
        if self.args.enable_plate:
            self.detectors.append(("plate", self.args.plate_interval, PlateDetector()))
            self.cached["plate"] = []

    def tick(self) -> None:
        frame = self.camera.read()
        if frame is None:
            return
        self.frame_index += 1
        started = time.time()
        for name, interval, detector in self.detectors:
            if self.frame_index % max(1, interval) != 0:
                continue
            try:
                self.cached[name] = detector.detect(frame)
            except Exception as exc:
                self.get_logger().warning(f"{name} inference failed: {exc}")
                self.cached[name] = []
        all_detections = [det for group in self.cached.values() for det in group]
        self.publish_anomalies(all_detections)
        annotated = self.draw(frame, all_detections)
        self.publish_image(annotated)
        self.show(annotated)
        elapsed = time.time() - started
        self.loop_fps = 0.9 * self.loop_fps + 0.1 * (1.0 / max(elapsed, 1e-6))
        if time.time() - self.last_log > 2.0:
            self.get_logger().info(f"fps={self.loop_fps:.1f}, detections={len(all_detections)}")
            self.last_log = time.time()

    def publish_anomalies(self, detections: list[Detection]) -> None:
        msg = AnomalyList()
        now_msg = self.get_clock().now().to_msg()
        if hasattr(msg, "header"):
            msg.header.stamp = now_msg
            msg.header.frame_id = self.args.frame_id
        msg.total_count = len(detections)
        msg.frame_timestamp = time.time()
        for det in detections:
            x1, y1, x2, y2 = det.box
            item = AnomalyDetection()
            if hasattr(item, "header"):
                item.header.stamp = now_msg
                item.header.frame_id = self.args.frame_id
            item.anomaly_type = det.kind
            item.timestamp = msg.frame_timestamp
            item.confidence = det.confidence
            item.position = Point(x=float((x1 + x2) / 2), y=float((y1 + y2) / 2), z=0.0)
            item.image_path = ""
            item.description = (
                f"source={det.source}; label={det.label}; bbox=({x1},{y1},{x2},{y2}); extra={det.extra}"
            )
            msg.anomalies.append(item)
        self.anomaly_pub.publish(msg)

    def publish_image(self, frame: np.ndarray) -> None:
        if self.image_pub is None or self.bridge is None:
            return
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self.COLORS.get(det.kind, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.label} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(th + 4, y1)
            cv2.rectangle(out, (x1, y_text - th - 6), (min(out.shape[1] - 1, x1 + tw + 4), y_text + 2), color, -1)
            cv2.putText(out, label, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(out, f"FPS {self.loop_fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        return out

    def show(self, frame: np.ndarray) -> None:
        if not self.args.show:
            return
        cv2.imshow("Integrated Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            rclpy.shutdown()

    def destroy_node(self) -> None:
        self.camera.release()
        if self.args.show:
            cv2.destroyAllWindows()
        super().destroy_node()


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Integrated ROS2 anomaly detector")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--loop-fps", type=float, default=15.0)
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anomaly-topic", default="/anomaly_detection")
    parser.add_argument("--image-topic", default="/anomaly_detection/image")

    parser.add_argument("--enable-fire", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fire-model", default=str(DEFAULT_FIRE_MODEL))
    parser.add_argument("--fire-conf", type=float, default=0.25)
    parser.add_argument("--fire-interval", type=int, default=1)
    parser.add_argument("--iou", type=float, default=0.45)

    parser.add_argument("--enable-safety", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--helmet-model", default=str(DEFAULT_HELMET_MODEL))
    parser.add_argument("--smoking-model", default=str(DEFAULT_SMOKING_MODEL))
    parser.add_argument("--helmet-conf", type=float, default=0.9)
    parser.add_argument("--smoking-conf", type=float, default=0.6)
    parser.add_argument("--safety-interval", type=int, default=3)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-half", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--enable-face", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-db", default=str(DEFAULT_FACE_DB))
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--face-det-size", type=int, default=640)
    parser.add_argument("--face-interval", type=int, default=5)

    parser.add_argument("--enable-plate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plate-interval", type=int, default=5)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = IntegratedAnomalyNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
