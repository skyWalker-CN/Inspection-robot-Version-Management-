#!/usr/bin/env python3
"""
Face Detection & Recognition using InsightFace.
Ported from face_detect package.
"""

import os
import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Face detection disabled.")


class FaceDetector:
    """InsightFace-based face detection and recognition."""

    def __init__(self, database_dir: str, threshold: float = 0.50):
        self.database_dir = database_dir
        self.threshold = threshold
        self.face_database = {}

        if not INSIGHTFACE_AVAILABLE:
            print("FaceDetector: InsightFace not available, detector disabled")
            self.app = None
            return

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace model loaded")

        self._load_face_database()

    @property
    def available(self) -> bool:
        return self.app is not None

    def _load_face_database(self):
        if not os.path.exists(self.database_dir):
            print(f"Face database directory not found: {self.database_dir}")
            return

        for person_name in os.listdir(self.database_dir):
            person_dir = os.path.join(self.database_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            feature_path = os.path.join(person_dir, "feature.npy")
            if os.path.exists(feature_path):
                embedding = np.load(feature_path)
                self.face_database[person_name] = embedding
                print(f"Face loaded: {person_name}")

        print(f"Face database: {list(self.face_database.keys())}")

    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        return float(np.dot(feat1, feat2) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8
        ))

    def detect(self, image: np.ndarray) -> list:
        """
        Run face detection and recognition.
        Returns list of dicts: {x1, y1, x2, y2, conf, name, similarity, source}
        """
        if not self.available:
            return []

        results = []
        try:
            faces = self.app.get(image)
            for face in faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box.tolist()
                det_score = float(face.det_score)

                result = {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": det_score, "source": "face",
                    "name": "unknown",
                    "similarity": 0.0,
                }

                # Face recognition
                if hasattr(face, 'embedding') and face.embedding is not None:
                    current_embedding = face.embedding
                    best_name = "unknown"
                    best_sim = 0.0

                    for person_name, db_embedding in self.face_database.items():
                        sim = self._cosine_similarity(current_embedding, db_embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_name = person_name

                    if best_sim >= self.threshold:
                        result["name"] = best_name
                        result["similarity"] = best_sim
                    else:
                        result["similarity"] = best_sim

                results.append(result)
        except Exception as e:
            print(f"FaceDetector error: {e}")

        return results