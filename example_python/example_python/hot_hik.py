#!/usr/bin/env python3
"""
热成像引导云台变焦节点（最终精确版）
补偿值根据最新校准：PAN_OFFSET = 21.82, TILT_OFFSET = 13.78
点动逻辑：归零后若计算角度 > 1° 则点动一次（水平10°/次，垂直6°/次），然后直接放大7倍。
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading

# ========== 配置参数 ==========
TEMP_THRESHOLD = 50.0
MIN_BLOB_AREA = 5

THERMAL_HFOV = 57.0
THERMAL_VFOV = 42.0
THERMAL_WIDTH = 256
THERMAL_HEIGHT = 192

# ---------- 补偿值（已更新为最新校准结果）----------
PAN_OFFSET = 18.82
TILT_OFFSET = 10.78

PRESET_DELAY = 3.0           # 归零等待
MAX_ZOOM = 7.0               # 放大倍数
ZOOM_SPEED = 6.7
ZOOM_DELAY = 1.5

MOVE_THRESHOLD = 1.0         # 角度偏差超过1°才点动（避免微小抖动）
JOG_INTERVAL = 0.3           # 点动间隔

def pixel_to_angle(u, v):
    cx = THERMAL_WIDTH / 2.0
    cy = THERMAL_HEIGHT / 2.0
    deg_per_pixel_x = THERMAL_HFOV / THERMAL_WIDTH
    deg_per_pixel_y = THERMAL_VFOV / THERMAL_HEIGHT
    pan = (u - cx) * deg_per_pixel_x + PAN_OFFSET
    tilt = (cy - v) * deg_per_pixel_y + TILT_OFFSET
    return pan, tilt


class ThermalGuidedPTZ(Node):
    def __init__(self):
        super().__init__('thermal_guided_ptz')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/thermal_camera/temp_raw', self.image_callback, 10)
        self.ptz_pub = self.create_publisher(String, '/controlCam', 10)

        self.target_u = 0
        self.target_v = 0
        self.target_locked = False

        self.state = "IDLE"
        self.current_zoom = 1.0
        self.lock = threading.Lock()

        self.running = True
        self.thread = threading.Thread(target=self.control_loop, daemon=True)
        self.thread.start()
        self.get_logger().info(f"节点启动（补偿 pan={PAN_OFFSET}° tilt={TILT_OFFSET}°）")

    def image_callback(self, msg):
        try:
            temp_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono16")
        except Exception:
            return
        if temp_raw is None or temp_raw.size == 0:
            return
        temp_c = (temp_raw.astype(np.float32) / 64.0) - 50.0
        mask = (temp_c > TEMP_THRESHOLD).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            with self.lock: self.target_locked = False
            return
        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < MIN_BLOB_AREA:
            with self.lock: self.target_locked = False
            return
        M = cv2.moments(best)
        if M["m00"] == 0: return
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        with self.lock:
            self.target_u = cx
            self.target_v = cy
            self.target_locked = True

    def control_loop(self):
        while self.running:
            with self.lock:
                locked = self.target_locked
                u, v = self.target_u, self.target_v

            if not locked:
                if self.state != "IDLE":
                    if self.current_zoom > 1.0:
                        self.send_zoom(-(self.current_zoom - 1.0))
                        time.sleep(abs(self.current_zoom - 1.0) / ZOOM_SPEED + ZOOM_DELAY)
                        self.current_zoom = 1.0
                    self.state = "IDLE"
                time.sleep(0.2)
                continue

            pan_off, tilt_off = pixel_to_angle(u, v)
            self.get_logger().info(f"📐 像素({u},{v}) → pan={pan_off:+.2f}° tilt={tilt_off:+.2f}°")

            if self.state == "IDLE":
                # 归零
                self.ptz_pub.publish(String(data="9 1"))
                self.get_logger().info("🔄 归零 (预置位 1)...")
                time.sleep(PRESET_DELAY)

                # 点动（仅当偏差超过阈值时点动一次）
                if abs(pan_off) > MOVE_THRESHOLD:
                    self.jog_once(pan_off, axis='pan')
                if abs(tilt_off) > MOVE_THRESHOLD:
                    self.jog_once(tilt_off, axis='tilt')

                self.state = "ZOOMING"

            elif self.state == "ZOOMING":
                if self.current_zoom < MAX_ZOOM:
                    step = MAX_ZOOM - self.current_zoom
                    self.send_zoom(step)
                    time.sleep(step / ZOOM_SPEED + ZOOM_DELAY)
                    self.current_zoom = MAX_ZOOM
                    self.get_logger().info(f"🔍 已放大至 {MAX_ZOOM:.1f}x")
                time.sleep(1.0)

            else:
                self.state = "IDLE"
                time.sleep(0.2)

    def jog_once(self, degrees, axis):
        """只发送一次点动命令，方向由角度正负决定"""
        if axis == 'pan':
            direction = 0 if degrees > 0 else 1
            self.get_logger().info(f"🕹️ 水平点动一次（{'右' if direction==0 else '左'}）")
        else:
            direction = 2 if degrees > 0 else 3
            self.get_logger().info(f"🕹️ 垂直点动一次（{'上' if direction==2 else '下'}）")
        self.ptz_pub.publish(String(data=f"{direction} 0"))
        time.sleep(JOG_INTERVAL)

    def send_zoom(self, step):
        if step > 0:
            self.ptz_pub.publish(String(data=f"4 {step:.2f}"))
        elif step < 0:
            self.ptz_pub.publish(String(data=f"5 {abs(step):.2f}"))

    def destroy_node(self):
        self.running = False
        self.thread.join()
        super().destroy_node()


def main():
    rclpy.init()
    node = ThermalGuidedPTZ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
