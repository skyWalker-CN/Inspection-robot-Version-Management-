#!/usr/bin/env python3
"""
License Plate Detector using HyperLPR3.
Ported from plate_detect_pkg package.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from hyperlpr3 import LicensePlateCatcher
    HYPERLPR_AVAILABLE = True
except ImportError:
    HYPERLPR_AVAILABLE = False
    print("Warning: HyperLPR3 not available. License plate detection disabled.")


class PlateDetector:
    """HyperLPR3-based license plate detector."""

    def __init__(self, font_path: str = None):
        if not HYPERLPR_AVAILABLE:
            print("PlateDetector: HyperLPR3 not available, detector disabled")
            self.catcher = None
            self.font = None
            return

        self.catcher = LicensePlateCatcher()
        print("HyperLPR3 initialized")

        # Load Chinese font
        if font_path is None:
            font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            import os
            if not os.path.exists(font_path):
                font_path = "/usr/share/fonts/truetype/arphic/uming.ttc"
        try:
            self.font = ImageFont.truetype(font_path, 24)
            print(f"Chinese font loaded: {font_path}")
        except Exception as e:
            print(f"Font load failed: {e}, using default font")
            self.font = ImageFont.load_default()

    @property
    def available(self) -> bool:
        return self.catcher is not None

    def detect(self, image: np.ndarray) -> list:
        """
        Run license plate detection.
        Returns list of dicts: {x1, y1, x2, y2, plate_number, score, source}
        """
        if not self.available:
            return []

        results = []
        try:
            plates = self.catcher(image)
            for plate in plates:
                plate_number = plate[0]
                score = float(plate[1])
                x1, y1, x2, y2 = plate[3]
                results.append({
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "plate_number": plate_number,
                    "conf": score,
                    "source": "plate",
                })
        except Exception as e:
            print(f"PlateDetector error: {e}")

        return results

    def draw_text(self, image: np.ndarray, text: str, position: tuple,
                  text_color=(0, 255, 0)) -> np.ndarray:
        """Draw Chinese text on OpenCV BGR image."""
        if self.font is None:
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, text_color, 2)
            return image

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=self.font,
                  fill=(text_color[2], text_color[1], text_color[0]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)