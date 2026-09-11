#!/usr/bin/env python3
"""
Safety Detector: helmet & smoking detection using YOLO models.
Ported from safety_detect package.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class SafetyDetector:
    """YOLO-based detector for helmet and smoking behavior."""

    def __init__(self, helmet_model_path: str, smoking_model_path: str):
        print(f"Loading helmet model: {helmet_model_path}")
        self.helmet_model = YOLO(helmet_model_path)
        print(f"Loading smoking model: {smoking_model_path}")
        self.smoking_model = YOLO(smoking_model_path)
        self.conf_threshold = 0.9
        print("SafetyDetector initialized")

    def detect(self, image: np.ndarray) -> list:
        """
        Run helmet and smoking detection.
        Returns list of dicts: {x1, y1, x2, y2, conf, class_name}
        """
        results = []

        # Helmet detection
        try:
            helmet_results = self.helmet_model(image, conf=self.conf_threshold, verbose=False)
            for r in helmet_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.helmet_model.names.get(cls_id, f"cls_{cls_id}")
                    results.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "class_name": cls_name, "source": "safety",
                    })
        except Exception as e:
            print(f"Helmet detection error: {e}")

        # Smoking detection
        try:
            smoking_results = self.smoking_model(image, conf=self.conf_threshold, verbose=False)
            for r in smoking_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.smoking_model.names.get(cls_id, f"cls_{cls_id}")
                    results.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "class_name": cls_name, "source": "safety",
                    })
        except Exception as e:
            print(f"Smoking detection error: {e}")

        return results