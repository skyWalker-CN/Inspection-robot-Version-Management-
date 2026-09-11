#!/usr/bin/env python3
"""
智能巡检机器人异常检测节点（ROS 图像话题版 + WebRTC + 云台联动）
- 检测火焰、烟雾、未知人员、未知车牌，自动控制云台追踪并放大7倍
- 多同类目标：直接放大7倍，不居中
- 优先级：火焰 > 烟雾 > 未知人员 > 未知车牌
- 火焰：强制停止机器人；非火焰：不停止机器人
- 居中尝试5次未成功 → 强制回到预设位1（9 1）+ 1倍变焦（5 32）
- 目标丢失1.5s直接判为失败，不占用重试次数
- 成功或失败后进入冷却期：火焰60秒，其他类型30秒
- 引入比例控制(P=0.6)防止云台超调震荡
- 追踪锁定防打断：居中途中遇同类或低优先级目标不换目标，遇高优先级目标才打断
- ★ 检测分辨率提升至 960x540（imgsz=960），平衡精度与速度
- ★ 抽帧检测（每5帧检测1次），非检测帧静态复用结果，GPU负载降低80%
- ★ WebRTC 推流固定为 640x360，大幅降低 CPU 编码负载，避免 ping 飙升
- ★ SDP 码率限制（1 Mbps），防止网络波动卡死
- ★ 浏览器断开后安全关闭连接，避免终端报错
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from base_interfaces.msg import AnomalyDetection, AnomalyList
from geometry_msgs.msg import Twist
import cv2
import base64
import time
import argparse
import os
import numpy as np
from datetime import datetime
import json
import threading
import queue
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ----- WebRTC 相关导入 -----
import asyncio
import websockets
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
import logging
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
sys.path.append('/home/jetson/anaconda3/envs/retest/lib/python3.10/site-packages/')

try:
    import torch
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    import dlib
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
except ImportError as e:
    pass

# ========== 相机视场角（1x 光学变倍，写死）==========
CAMERA_HFOV = 55.0
CAMERA_VFOV = 33.0

# ========== PTZ 状态常量 ==========
PTZ_IDLE = "IDLE"
PTZ_CENTERING = "CENTERING"
PTZ_ZOOMING = "ZOOMING"
PTZ_MULTI_ZOOM = "MULTI_ZOOM"
PTZ_SUCCESS = "SUCCESS"
PTZ_FAILED = "FAILED"

# 异常类型优先级（数字越小优先级越高）
PRIORITY_ORDER = ['fire', 'smoke', 'unknown_person', 'unknown_vehicle']

# ========== 异常目标跟踪器 ==========
class AnomalyTargetTracker:
    def __init__(self, disappear_timeout=20.0):
        self.disappear_timeout = disappear_timeout
        self.tracked_targets = {}
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 5.0

    def generate_target_id(self, anomaly):
        target_type = anomaly['type']
        if target_type == 'unknown_vehicle':
            plate_text = anomaly.get('plate_text', '')
            if plate_text:
                return f"vehicle_{plate_text}"
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                return f"vehicle_pos_{(x1+x2)//2//50}_{(y1+y2)//2//50}"
        elif target_type == 'unknown_person':
            person_name = anomaly.get('person_name', '')
            if person_name and person_name not in ("未知", "特征提取失败"):
                return f"person_{person_name}"
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                similar_id = self.find_similar_unknown_person(anomaly)
                if similar_id:
                    return similar_id
                return f"unknown_person_{(x1+x2)//2//30}_{(y1+y2)//2//30}"
        elif target_type in ['fire', 'smoke']:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                return f"{target_type}_pos_{(x1+x2)//2//30}_{(y1+y2)//2//30}"
        bbox = anomaly.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            return f"{target_type}_{(x1+x2)//2//50}_{(y1+y2)//2//50}"
        return f"{target_type}_unknown"

    def should_publish_alert(self, anomaly):
        current_time = time.time()
        target_id = self.generate_target_id(anomaly)
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.cleanup_disappeared_targets()
            self.last_cleanup_time = current_time
        if target_id in self.tracked_targets:
            info = self.tracked_targets[target_id]
            info['last_seen'] = current_time
            if info['alert_published']:
                return False
        self.tracked_targets[target_id] = {
            'type': anomaly['type'],
            'first_seen': current_time,
            'last_seen': current_time,
            'alert_published': True,
            'anomaly_info': anomaly
        }
        return True

    def cleanup_disappeared_targets(self):
        current_time = time.time()
        for tid in list(self.tracked_targets.keys()):
            if current_time - self.tracked_targets[tid]['last_seen'] > self.disappear_timeout:
                del self.tracked_targets[tid]

    def find_similar_unknown_person(self, current_anomaly):
        if current_anomaly['type'] != 'unknown_person':
            return None
        cb = current_anomaly.get('bbox', [])
        if len(cb) != 4:
            return None
        ccx = (cb[0]+cb[2])//2
        ccy = (cb[1]+cb[3])//2
        for tid, info in self.tracked_targets.items():
            if info['type'] == 'unknown_person' and tid.startswith('unknown_person_'):
                parts = tid.split('_')
                if len(parts) >= 4:
                    try:
                        tx = int(parts[2]) * 30
                        ty = int(parts[3]) * 30
                        if ((ccx-tx)**2 + (ccy-ty)**2)**0.5 < 80:
                            return tid
                    except:
                        continue
        return None

# ========== WebRTC 视频轨道（强制缩放至 640x360 推流）==========
class ProcessedFrameTrack(VideoStreamTrack):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.frame_count = 0
        self._last_frame = None
        # ★ 固定推流分辨率为 640x360（宽高比 16:9）
        self.push_width = 640
        self.push_height = 360

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
            frame_bgr = self._last_frame if self._last_frame is not None else \
                np.zeros((480, 640, 3), dtype=np.uint8)

        # ★★★ 强制缩放到 640x360，大幅降低 CPU 编码负载 ★★★
        frame_bgr = cv2.resize(frame_bgr, (self.push_width, self.push_height), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        self.frame_count += 1
        return video_frame

# ========== ROS2 节点 ==========
class AnomalyDetectionPublisher(Node):
    def __init__(self, args):
        super().__init__('anomaly_detection_publisher')
        self.bridge = CvBridge()
        self.anomaly_publisher = self.create_publisher(AnomalyList, '/anomaly_detection/data', 10)
        self.alert_publisher = self.create_publisher(String, '/anomaly_detection/alerts', 10)
        self.ptz_pub = self.create_publisher(String, '/controlCam', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        self.args = args
        self.flame_conf_threshold = args.flame_conf
        self.smoke_conf_threshold = args.smoke_conf
        self.face_conf_threshold = args.face_conf
        self.plate_conf_threshold = args.plate_conf
        self.show_display = args.display

        if self.show_display:
            try:
                import tkinter as tk
                root = tk.Tk()
                self.screen_w = root.winfo_screenwidth()
                self.screen_h = root.winfo_screenheight()
                root.destroy()
            except:
                self.screen_w = 1920
                self.screen_h = 1080

        # ★ 检测分辨率提升至 960（对应 imgsz=960）
        self.detect_width = 960
        self.models = {}
        self.last_screenshot_time = {}
        self.screenshot_interval = 20
        self.target_tracker = AnomalyTargetTracker(disappear_timeout=args.target_timeout)

        self.fire_smoke_cooldown = 10.0
        self.last_fire_alert_time = 0
        self.last_smoke_alert_time = 0
        self.face_cooldown = 5.0
        self.last_face_alert_time = 0
        self.plate_cooldown = 10.0
        self.last_plate_alert_time = 0

        self.plate_cache = {}
        self.plate_cache_timeout = 5.0
        self.grid_size = 10
        self.ocr_queue = queue.Queue(maxsize=4)
        self.ocr_result_queue = queue.Queue()
        self.ocr_threads = []
        self.ocr_running = True
        self.ocr_skip_frames = 1
        self._fast_ocr_frame_tag = -1
        self.ocr_frame_counter = 0

        self.history_folder = "history_anomaly"
        self.known_faces_folder = "/home/jetson/ros2_ws/src/example_python/data/known_faces"
        self.whitelist_file = self.find_whitelist_file()
        os.makedirs(self.history_folder, exist_ok=True)
        os.makedirs(self.known_faces_folder, exist_ok=True)

        self.latest_processed_frame = None
        self.processed_frame_lock = threading.Lock()
        self.webrtc_running = True
        self.webrtc_loop = None
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.webrtc_thread = threading.Thread(target=self.run_webrtc_server, daemon=True)
        self.webrtc_thread.start()

        # ====== PTZ 状态变量 ======
        self.ptz_enabled = args.enable_ptz
        self.ptz_state = PTZ_IDLE
        self.ptz_target_bbox = None
        self.ptz_target_type = None
        self.ptz_multi_target = False
        self.ptz_active_tracking_type = None
        self.ptz_current_zoom = 1.0
        self.ptz_lock = threading.Lock()

        self.ptz_attempts = 0
        self.ptz_max_attempts = 5
        self.ptz_target_lost_time = 0.0
        self.ptz_lost_grace = 1.5
        self.ptz_state_enter_time = time.time()

        # ====== 目标冷却时间配置 ======
        self.ptz_cooldowns = {}
        self.fire_cooldown_time = 60.0
        self.other_cooldown_time = 30.0

        # ====== 机器人运动状态 ======
        self.robot_is_moving = False
        self.last_cmd_vel_time = 0.0
        self.is_stopping_robot = False
        self.last_stop_publish_time = 0.0
        self.stop_publish_rate = 0.05

        # ====== ★ 抽帧相关 ======
        self.detect_interval = 5                         # 每5帧检测一次
        self.frame_counter = 0
        self.last_anomalies = []                         # 上一帧检测结果（非检测帧复用）
        self.last_all_detections = []

        if self.ptz_enabled:
            self.ptz_thread = threading.Thread(target=self.ptz_control_loop, daemon=True)
            self.ptz_thread.start()

        self.init_detection_models()
        self.load_whitelist()
        self.setup_chinese_font()

        self.image_sub = self.create_subscription(
            Image, '/hik_camera/image_raw', self.image_callback, 10
        )
        self.frame_count = 0
        self.frame_shape = (1080, 1920)

    # ---------------- cmd_vel 相关 ----------------
    def cmd_vel_callback(self, msg):
        self.robot_is_moving = (abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01)
        self.last_cmd_vel_time = time.time()

    def publish_stop_high_rate(self, now):
        if now - self.last_stop_publish_time >= self.stop_publish_rate:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            self.last_stop_publish_time = now

    # ---------------- WebRTC ----------------
    def run_webrtc_server(self):
        try:
            self.webrtc_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.webrtc_loop)
            async def start_server():
                self.websocket_server = await websockets.serve(
                    self.browser_handler, "0.0.0.0", 8080
                )
            self.webrtc_loop.run_until_complete(start_server())
            while self.webrtc_running:
                self.webrtc_loop.run_until_complete(asyncio.sleep(0.5))
            for t in asyncio.all_tasks(self.webrtc_loop):
                t.cancel()
            self.webrtc_loop.run_until_complete(asyncio.sleep(0.1))
            self.webrtc_loop.close()
        except Exception:
            pass

    async def browser_handler(self, websocket):
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
                    # ★★★ 添加码率限制：在 video 媒体行后插入 b=AS:1000（1 Mbps）★★★
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
                    cand = self.create_ice_candidate(data["candidate"])
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
            # ★★★ 安全关闭连接，防止 Task exception 报错 ★★★
            if pc is not None:
                await pc.close()

    def create_ice_candidate(self, cand):
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

    def update_processed_frame(self, frame):
        with self.processed_frame_lock:
            self.latest_processed_frame = frame.copy()

    def get_latest_processed_frame(self):
        with self.processed_frame_lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frame_shape = cv_image.shape[:2]
            self.process_frame(cv_image)
        except Exception:
            pass

    # ========== ★ 核心：process_frame 抽帧 + 静态复用 ==========
    def process_frame(self, frame):
        if frame is None:
            return
        self.frame_count += 1
        self.frame_counter += 1

        # 判断是否为检测帧（每 detect_interval 帧检测一次）
        is_detect_frame = (self.frame_counter % self.detect_interval == 0)

        try:
            h, w = frame.shape[:2]
            if w > self.detect_width:
                scale = self.detect_width / w
                new_w = self.detect_width
                new_h = int(h * scale)
                small_frame = cv2.resize(frame, (new_w, new_h))
            else:
                scale = 1.0
                small_frame = frame.copy()

            anomalies = []
            all_detections = []

            if is_detect_frame:
                # ---------- 检测帧：执行完整 YOLO + OCR ----------
                anomalies, all_detections = self.detect_anomalies(small_frame, original_frame=frame)

                # 将检测结果缩放到原图坐标
                if scale != 1.0:
                    for anomaly in anomalies:
                        if 'bbox' in anomaly:
                            anomaly['bbox'] = [int(v / scale) for v in anomaly['bbox']]
                        if 'position' in anomaly:
                            anomaly['position'] = [int(p / scale) for p in anomaly['position']]
                    for det in all_detections:
                        if 'bbox' in det:
                            det['bbox'] = [int(v / scale) for v in det['bbox']]
                        if 'position' in det:
                            det['position'] = [int(p / scale) for p in det['position']]

                # 保存最新检测结果
                self.last_anomalies = anomalies
                self.last_all_detections = all_detections

                # ---- 云台目标更新 ----
                if self.ptz_enabled:
                    self.update_ptz_target(anomalies)

                # ---- 发布报警 ----
                if anomalies:
                    self.publish_anomalies(anomalies)
                    filtered = [a for a in anomalies if self.target_tracker.should_publish_alert(a)]
                    if filtered:
                        self.publish_alerts(filtered)

            else:
                # ---------- 非检测帧：静态复用上一帧结果，零耗时 ----------
                anomalies = self.last_anomalies
                all_detections = self.last_all_detections

                # 非检测帧也更新 PTZ 目标（沿用旧框），保证 PTZ 循环能持续获取 bbox
                if self.ptz_enabled and anomalies:
                    self.update_ptz_target(anomalies)

            # ---------- 保存截图（仅检测帧保存，避免重复） ----------
            if is_detect_frame:
                for anomaly in anomalies:
                    screenshot_path = self.save_anomaly_screenshot(frame, anomaly['type'])
                    anomaly['image_path'] = screenshot_path or ''

            # ---------- 绘制并推送画面（每一帧都执行，保证满帧） ----------
            display_frame = frame.copy()
            display_frame = self.draw_detections(display_frame, anomalies, all_detections)
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(display_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(display_frame, f"Anomalies: {len(anomalies)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            detect_info = f"Detect: {self.frame_counter%self.detect_interval+1}/{self.detect_interval}"
            cv2.putText(display_frame, detect_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
            cv2.putText(display_frame, f"PTZ: {self.ptz_state} attempt:{self.ptz_attempts}/{self.ptz_max_attempts}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.putText(display_frame, f"RobotMoving: {self.robot_is_moving} Stop:{self.is_stopping_robot}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            self.update_processed_frame(display_frame)

            if self.show_display:
                cv2.namedWindow('Anomaly Detection', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Anomaly Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                show_frame = cv2.resize(display_frame, (self.screen_w, self.screen_h))
                cv2.imshow('Anomaly Detection', show_frame)
                cv2.waitKey(1)

        except Exception:
            pass

    # ============== PTZ 目标选择（含追踪锁定与优先级打断）==============
    def update_ptz_target(self, anomalies):
        now = time.time()
        by_type = {}
        for a in anomalies:
            by_type.setdefault(a['type'], []).append(a)

        selected_type = None
        selected_list = None
        for t in PRIORITY_ORDER:
            if t in by_type and by_type[t]:
                if t in self.ptz_cooldowns and now < self.ptz_cooldowns[t]:
                    continue
                selected_type = t
                selected_list = by_type[t]
                break

        with self.ptz_lock:
            # ====== 处于追踪过程中（居中/变焦）======
            if self.ptz_state in (PTZ_CENTERING, PTZ_ZOOMING, PTZ_MULTI_ZOOM) and self.ptz_active_tracking_type is not None:
                
                # 1. 检测是否出现更高优先级目标
                if selected_type is not None and selected_type != self.ptz_active_tracking_type:
                    current_prio = PRIORITY_ORDER.index(self.ptz_active_tracking_type) if self.ptz_active_tracking_type in PRIORITY_ORDER else 99
                    new_prio = PRIORITY_ORDER.index(selected_type) if selected_type in PRIORITY_ORDER else 99
                    if new_prio < current_prio:
                        print(f"[PTZ] 发现更高优先级目标 {selected_type}，中断当前 {self.ptz_active_tracking_type} 追踪！")
                        self.ptz_state = PTZ_IDLE
                        self.ptz_attempts = 0
                        self.ptz_active_tracking_type = None
                        self.is_stopping_robot = False
                    else:
                        return
                
                # 2. 如果是同种目标
                if selected_type == self.ptz_active_tracking_type:
                    if self.ptz_state == PTZ_CENTERING:
                        if selected_list:
                            self.ptz_target_bbox = selected_list[0].get('bbox', None)
                        return
                    if self.ptz_state in (PTZ_ZOOMING, PTZ_MULTI_ZOOM):
                        return

            # ====== 正常 IDLE 状态或被高优先级打断后的逻辑 ======
            if selected_type is None:
                self.ptz_target_type = None
                return

            multi = len(selected_list) > 1
            self.ptz_target_type = selected_type
            self.ptz_multi_target = multi
            if multi:
                self.ptz_target_bbox = None
            else:
                self.ptz_target_bbox = selected_list[0].get('bbox', None)

    # ============== PTZ 控制主循环 ==============
    def ptz_control_loop(self):
        print("[PTZ] 控制循环启动")
        while rclpy.ok():
            try:
                now = time.time()
                with self.ptz_lock:
                    bbox = self.ptz_target_bbox
                    target_type = self.ptz_target_type
                    multi = self.ptz_multi_target

                # ===== 火焰强制停止机器人 =====
                if self.ptz_active_tracking_type == 'fire' and self.ptz_state != PTZ_IDLE:
                    self.is_stopping_robot = True
                    self.publish_stop_high_rate(now)
                else:
                    if self.is_stopping_robot and self.ptz_state == PTZ_IDLE:
                        self.is_stopping_robot = False

                # ===== 处理目标丢失（宽限期）=====
                if bbox is None and not multi:
                    if self.ptz_state in (PTZ_CENTERING, PTZ_ZOOMING):
                        if self.ptz_target_lost_time == 0:
                            self.ptz_target_lost_time = now
                            print(f"[PTZ] 目标短暂丢失，进入 {self.ptz_lost_grace}s 宽限期")
                        elif now - self.ptz_target_lost_time > self.ptz_lost_grace:
                            print("[PTZ] 宽限期超时，目标丢失 → 直接判定失败")
                            self.ptz_target_lost_time = 0
                            self._goto_failed()
                        time.sleep(0.15)
                        continue
                    elif self.ptz_state == PTZ_MULTI_ZOOM:
                        pass
                    elif self.ptz_state == PTZ_IDLE:
                        time.sleep(0.2)
                        continue
                else:
                    self.ptz_target_lost_time = 0

                # ===== 状态机 =====
                if self.ptz_state == PTZ_IDLE:
                    if target_type is None:
                        time.sleep(0.2)
                        continue

                    print(f"[PTZ] 检测到目标: type={target_type}, multi={multi}, "
                          f"robot_moving={self.robot_is_moving}")
                    self.ptz_attempts = 1
                    self.ptz_active_tracking_type = target_type

                    if multi:
                        print("[PTZ] 多同类目标模式：直接放大7倍")
                        self.ptz_state = PTZ_MULTI_ZOOM
                        self.ptz_state_enter_time = now
                    else:
                        print("[PTZ] 单目标模式：复位云台到预设1")
                        self.ptz_pub.publish(String(data="9 1"))
                        time.sleep(2.0)
                        self.ptz_current_zoom = 1.0
                        self.ptz_state = PTZ_CENTERING
                        self.ptz_state_enter_time = time.time()

                elif self.ptz_state == PTZ_CENTERING:
                    self._do_centering(bbox)

                elif self.ptz_state == PTZ_ZOOMING:
                    self._do_zooming(bbox)

                elif self.ptz_state == PTZ_MULTI_ZOOM:
                    self._do_multi_zoom()

                elif self.ptz_state == PTZ_SUCCESS:
                    self._do_reset("SUCCESS")

                elif self.ptz_state == PTZ_FAILED:
                    self._do_reset("FAILED")

                time.sleep(0.1)
            except Exception as e:
                print(f"[PTZ] 控制循环异常: {e}")
                time.sleep(0.3)

    # ---------- 居中 ----------
    def _do_centering(self, bbox):
        if bbox is None or len(bbox) != 4:
            return
        h, w = self.frame_shape
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        img_cx, img_cy = w // 2, h // 2
        dx = cx - img_cx
        dy = cy - img_cy
        current_hfov = CAMERA_HFOV / max(self.ptz_current_zoom, 1.0)
        current_vfov = CAMERA_VFOV / max(self.ptz_current_zoom, 1.0)
        
        pan_angle = dx / w * current_hfov
        tilt_angle = -dy / h * current_vfov

        print(f"[PTZ] CENTERING attempt={self.ptz_attempts}/{self.ptz_max_attempts} "
              f"原始偏差 pan={pan_angle:+.2f} tilt={tilt_angle:+.2f}")

        if abs(pan_angle) < 5.0 and abs(tilt_angle) < 5.0:
            print("[PTZ] 已居中（偏差<5°），进入 ZOOMING")
            self.ptz_state = PTZ_ZOOMING
            self.ptz_state_enter_time = time.time()
            return

        if self.ptz_attempts <= self.ptz_max_attempts:
            P_FACTOR = 0.8
            move_pan = pan_angle * P_FACTOR
            move_tilt = tilt_angle * P_FACTOR
            
            if abs(move_pan) > 0.1 and abs(move_pan) < 1.0:
                move_pan = 1.0 if move_pan > 0 else -1.0
            if abs(move_tilt) > 0.1 and abs(move_tilt) < 1.0:
                move_tilt = 1.0 if move_tilt > 0 else -1.0
                
            print(f"[PTZ] 发送移动指令 实际移动 pan={move_pan:+.2f} tilt={move_tilt:+.2f}，消耗 1 次尝试 ({self.ptz_attempts}/{self.ptz_max_attempts})")
            self.send_move(move_pan, move_tilt)
            self.ptz_attempts += 1
            time.sleep(1.5)
        else:
            print(f"[PTZ] {self.ptz_max_attempts}次尝试均失败 → FAILED")
            self._goto_failed()

    # ---------- 放大（单目标）----------
    def _do_zooming(self, bbox):
        if int(self.ptz_current_zoom) != 7:
            print("[PTZ] 发送变焦指令 4 7（7倍）")
            self.ptz_pub.publish(String(data="4 7"))
            self.ptz_current_zoom = 7.0
            time.sleep(1.2)
        print("[PTZ] 追踪至中心并放大7倍成功！→ SUCCESS")
        self.ptz_state = PTZ_SUCCESS
        self.ptz_state_enter_time = time.time()

    # ---------- 多目标放大 ----------
    def _do_multi_zoom(self):
        if int(self.ptz_current_zoom) != 7:
            print("[PTZ] MULTI_ZOOM: 发送变焦指令 4 7")
            self.ptz_pub.publish(String(data="4 7"))
            self.ptz_current_zoom = 7.0
            time.sleep(1.5)
        print("[PTZ] MULTI_ZOOM 完成 → SUCCESS")
        self.ptz_state = PTZ_SUCCESS
        self.ptz_state_enter_time = time.time()

    # ---------- 复位（成功或失败后）----------
    def _do_reset(self, reason):
        print(f"[PTZ] {reason}: 复位到预设1（9 1）+ 1倍变焦（5 32）")
        if reason == "SUCCESS":
            time.sleep(1.5)
        self.ptz_pub.publish(String(data="9 1"))
        time.sleep(0.6)
        self.ptz_pub.publish(String(data="5 32"))
        self.ptz_current_zoom = 1.0
        time.sleep(1.5)
        
        tracked_type = self.ptz_active_tracking_type
        if tracked_type is not None:
            now = time.time()
            if tracked_type == 'fire':
                self.ptz_cooldowns[tracked_type] = now + self.fire_cooldown_time
                print(f"[PTZ] {tracked_type} 进入冷却期 {self.fire_cooldown_time}s")
            else:
                self.ptz_cooldowns[tracked_type] = now + self.other_cooldown_time
                print(f"[PTZ] {tracked_type} 进入冷却期 {self.other_cooldown_time}s")

        self.ptz_state = PTZ_IDLE
        self.ptz_attempts = 0
        self.ptz_target_lost_time = 0
        self.ptz_active_tracking_type = None
        self.is_stopping_robot = False
        print("[PTZ] 状态切换: IDLE")

    def _goto_failed(self):
        self.ptz_state = PTZ_FAILED
        self.ptz_state_enter_time = time.time()

    # ---------- 发送移动指令 ----------
    def send_move(self, pan, tilt):
        if abs(pan) > 0.1:
            cmd = f"0 {pan:.2f}" if pan > 0 else f"1 {abs(pan):.2f}"
            self.ptz_pub.publish(String(data=cmd))
        if abs(tilt) > 0.1:
            cmd = f"2 {tilt:.2f}" if tilt > 0 else f"3 {abs(tilt):.2f}"
            self.ptz_pub.publish(String(data=cmd))

    # ---------------- 模型与OCR ----------------
    def find_whitelist_file(self):
        possible = ["/home/jetson/ros2_ws/src/example_python/data/whitelist/whitelist.txt"]
        for p in possible:
            if os.path.exists(p):
                return p
        return "whitelist.txt"

    def find_dlib_model_paths(self):
        base_paths = [
            "/home/jetson/ros2_ws/install/example_python/share/example_python/models",
            "/home/jetson/ros2_ws/src/example_python/models",
            "models"
        ]
        for base in base_paths:
            shape = os.path.join(base, "shape_predictor_68_face_landmarks.dat")
            recog = os.path.join(base, "dlib_face_recognition_resnet_model_v1.dat")
            if os.path.exists(shape) and os.path.exists(recog):
                return shape, recog
        return None, None

    def init_detection_models(self):
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.args.fire_model and os.path.exists(self.args.fire_model):
                self.models['fire'] = YOLO(self.args.fire_model)
                self.models['fire'].to(device)
            if self.args.plate_model and os.path.exists(self.args.plate_model):
                self.models['plate'] = YOLO(self.args.plate_model)
                self.models['plate'].to(device)
            if self.args.face_model and os.path.exists(self.args.face_model):
                self.models['face'] = YOLO(self.args.face_model)
                self.models['face'].to(device)
            dlib_shape, dlib_recog = self.find_dlib_model_paths()
            if dlib_shape and dlib_recog:
                self.shape_predictor = dlib.shape_predictor(dlib_shape)
                self.face_recognition_model = dlib.face_recognition_model_v1(dlib_recog)
                self.load_known_face_features()
            else:
                self.shape_predictor = None
                self.face_recognition_model = None
            self.ocr = PaddleOCR(use_angle_cls=False, lang='ch', use_gpu=True, show_log=False)
            self.start_ocr_threads()
        except Exception:
            pass

    def load_whitelist(self):
        self.whitelist = set()
        try:
            if os.path.exists(self.whitelist_file):
                with open(self.whitelist_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        plate = line.strip()
                        if plate:
                            self.whitelist.add(self.clean_plate_text(plate))
        except Exception:
            pass

    def load_known_face_features(self):
        self.known_face_features = []
        self.known_face_names = []
        try:
            for file in os.listdir(self.known_faces_folder):
                if file.endswith(".npy"):
                    feature = np.load(os.path.join(self.known_faces_folder, file))
                    self.known_face_features.append(feature)
                    self.known_face_names.append(os.path.splitext(file)[0])
            if not self.known_face_features:
                for fname in os.listdir(self.known_faces_folder):
                    if fname.lower().endswith(('.jpg','.jpeg','.png')):
                        img = cv2.imread(os.path.join(self.known_faces_folder, fname))
                        if img is not None and 'face' in self.models:
                            results = self.models['face'](img, conf=0.5)
                            for r in results:
                                if r.boxes is not None and len(r.boxes) > 0:
                                    box = r.boxes[0].xyxy[0].cpu().numpy()
                                    x1,y1,x2,y2 = map(int, box)
                                    feat = self.extract_face_features(img, x1, y1, x2, y2)
                                    if feat is not None:
                                        name = os.path.splitext(fname)[0]
                                        self.known_face_features.append(feat)
                                        self.known_face_names.append(name)
                                        np.save(os.path.join(self.known_faces_folder, f"{name}.npy"), feat)
                                    break
        except Exception:
            pass

    def extract_face_features(self, image, x1, y1, x2, y2, min_size=60):
        if self.shape_predictor is None or self.face_recognition_model is None:
            return None
        try:
            face = image[y1:y2, x1:x2]
            if face.shape[0] < min_size or face.shape[1] < min_size:
                return None
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            rect = dlib.rectangle(0, 0, face.shape[1], face.shape[0])
            shape = self.shape_predictor(face_rgb, rect)
            desc = self.face_recognition_model.compute_face_descriptor(face_rgb, shape)
            return np.array(desc)
        except:
            return None

    def recognize_face(self, feature, threshold=0.5):
        if feature is None or not self.known_face_features:
            return "特征提取失败", 1.0
        distances = [np.linalg.norm(feature - known) for known in self.known_face_features]
        min_dist = min(distances)
        idx = distances.index(min_dist)
        name = self.known_face_names[idx] if min_dist < threshold else "未知"
        return name, min_dist

    def setup_chinese_font(self):
        try:
            font_paths = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
            for fp in font_paths:
                if os.path.exists(fp):
                    self.chinese_font = ImageFont.truetype(fp, 20)
                    return
            self.chinese_font = ImageFont.load_default()
        except:
            self.chinese_font = ImageFont.load_default()

    def draw_chinese_text(self, img, text, position, font_size=20, color=(255,255,255), bg_color=None):
        try:
            img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            bbox = draw.textbbox((0,0), text, font=font)
            tw = bbox[2]-bbox[0]
            th = bbox[3]-bbox[1]
            if bg_color:
                draw.rectangle((position[0]-2, position[1]-th-2, position[0]+tw+2, position[1]+2), fill=bg_color)
            else:
                for adj in range(-1,2):
                    for adj2 in range(-1,2):
                        if adj != 0 or adj2 != 0:
                            draw.text((position[0]+adj, position[1]+adj2), text, font=font, fill=(0,0,0))
            draw.text(position, text, font=font, fill=(color[2],color[1],color[0]))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return img

    def detect_anomalies(self, frame, original_frame=None):
        if original_frame is None:
            original_frame = frame
        anomalies = []
        all_detections = []
        oh, ow = original_frame.shape[:2]
        sh, sw = frame.shape[:2]
        scale_x = ow / sw
        scale_y = oh / sh

        if 'fire' in self.models:
            min_conf = min(self.flame_conf_threshold, self.smoke_conf_threshold)
            # ★ imgsz=960
            results = self.models['fire'](frame, conf=min_conf, imgsz=960, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_names = result.names
                        class_name = class_names[cls] if cls in class_names else 'unknown'
                        if 'fire' in class_name.lower() or '火' in class_name:
                            anomaly_type = 'fire'
                            if conf < self.flame_conf_threshold:
                                continue
                        elif 'smoke' in class_name.lower() or '烟' in class_name:
                            anomaly_type = 'smoke'
                            if conf < self.smoke_conf_threshold:
                                continue
                        else:
                            continue
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        anomalies.append({
                            'type': anomaly_type,
                            'confidence': conf,
                            'position': [(x1+x2)//2, (y1+y2)//2],
                            'bbox': [x1, y1, x2, y2],
                            'description': f'{anomaly_type} detected with confidence {conf:.2f}'
                        })

        if 'face' in self.models:
            # ★ imgsz=960
            results = self.models['face'](frame, conf=self.face_conf_threshold, imgsz=960, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        orig_x1 = int(x1 * scale_x)
                        orig_y1 = int(y1 * scale_y)
                        orig_x2 = int(x2 * scale_x)
                        orig_y2 = int(y2 * scale_y)
                        face_feature = self.extract_face_features(original_frame, orig_x1, orig_y1, orig_x2, orig_y2)
                        name, distance = self.recognize_face(face_feature)
                        if name == "未知" or name == "特征提取失败":
                            anomalies.append({
                                'type': 'unknown_person',
                                'confidence': conf,
                                'position': [(x1+x2)//2, (y1+y2)//2],
                                'bbox': [x1, y1, x2, y2],
                                'description': f'Unknown person detected with confidence {conf:.2f}, distance: {distance:.2f}',
                                'person_name': name,
                                'recognition_distance': distance
                            })
                        else:
                            all_detections.append({
                                'type': 'known_person',
                                'confidence': conf,
                                'position': [(x1+x2)//2, (y1+y2)//2],
                                'bbox': [x1, y1, x2, y2],
                                'person_name': name,
                                'recognition_distance': distance
                            })

        plate_detections = self.detect_plate_optimized(frame, original_frame)
        for plate_info in plate_detections:
            bbox = plate_info['bbox']
            x1, y1, x2, y2 = bbox
            detection_info = {
                'type': 'vehicle_plate',
                'confidence': plate_info['confidence'],
                'position': [(x1+x2)//2, (y1+y2)//2],
                'bbox': bbox,
                'plate_text': plate_info['plate_text'],
                'is_registered': plate_info['is_registered'],
                'ocr_processed': plate_info['ocr_processed']
            }
            all_detections.append(detection_info)
            if (plate_info['ocr_processed'] and plate_info['plate_text'] and 
                plate_info['plate_text'] not in ["size_error", "no_result", "low_confidence", "ocr_error"] 
                and not plate_info['is_registered']):
                clean_plate_text = self.normalize_plate(plate_info['plate_text'])
                if self.is_valid_plate(clean_plate_text):
                    anomalies.append({
                        'type': 'unknown_vehicle',
                        'confidence': plate_info['confidence'],
                        'position': [(x1+x2)//2, (y1+y2)//2],
                        'bbox': bbox,
                        'description': f'Unknown vehicle: {clean_plate_text} (confidence: {plate_info["confidence"]:.2f})',
                        'plate_text': clean_plate_text
                    })
        return anomalies, all_detections

    def detect_plate_optimized(self, frame, original_frame=None):
        if original_frame is None:
            original_frame = frame
        plate_detections = []
        self.ocr_frame_counter += 1
        should_process_ocr = (self.ocr_frame_counter % self.ocr_skip_frames) == 0
        processed_results = {}
        while not self.ocr_result_queue.empty():
            try:
                bbox_id, (plate_number, ocr_confidence), result_frame_idx = self.ocr_result_queue.get_nowait()
                if result_frame_idx >= self.ocr_frame_counter - 8:
                    processed_results[bbox_id] = (plate_number, ocr_confidence)
            except:
                break

        if 'plate' in self.models:
            # ★ imgsz=960
            results = self.models['plate'](frame, conf=self.plate_conf_threshold, imgsz=960, verbose=False)
            oh, ow = original_frame.shape[:2]
            sh, sw = frame.shape[:2]
            scale_x = ow / sw
            scale_y = oh / sh

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1_grid = (x1 // self.grid_size) * self.grid_size
                        y1_grid = (y1 // self.grid_size) * self.grid_size
                        x2_grid = (x2 // self.grid_size) * self.grid_size
                        y2_grid = (y2 // self.grid_size) * self.grid_size
                        bbox_key = f"{x1_grid}_{y1_grid}_{x2_grid}_{y2_grid}"
                        plate_info = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'plate_text': '',
                            'is_registered': False,
                            'ocr_processed': False
                        }
                        cached_result = self.get_cached_plate_result(bbox_key)
                        if cached_result:
                            plate_number, ocr_confidence = cached_result
                            clean_plate_number = self.normalize_plate(plate_number)
                            if self.is_valid_plate(clean_plate_number):
                                plate_info['plate_text'] = clean_plate_number
                                plate_info['is_registered'] = clean_plate_number in self.whitelist
                            plate_info['ocr_processed'] = True
                        elif bbox_key in processed_results:
                            plate_number, ocr_confidence = processed_results[bbox_key]
                            self.cache_plate_result(bbox_key, (plate_number, ocr_confidence))
                            clean_plate_number = self.normalize_plate(plate_number)
                            if self.is_valid_plate(clean_plate_number):
                                plate_info['plate_text'] = clean_plate_number
                                plate_info['is_registered'] = clean_plate_number in self.whitelist
                            plate_info['ocr_processed'] = True
                        else:
                            orig_x1 = int(x1 * scale_x)
                            orig_y1 = int(y1 * scale_y)
                            orig_x2 = int(x2 * scale_x)
                            orig_y2 = int(y2 * scale_y)
                            pad_w = int((orig_x2 - orig_x1) * 0.1)
                            pad_h = int((orig_y2 - orig_y1) * 0.1)
                            orig_x1 = max(0, orig_x1 - pad_w)
                            orig_y1 = max(0, orig_y1 - pad_h)
                            orig_x2 = min(ow, orig_x2 + pad_w)
                            orig_y2 = min(oh, orig_y2 + pad_h)
                            plate_crop = original_frame[orig_y1:orig_y2, orig_x1:orig_x2]
                            if plate_crop.size > 0:
                                if self._fast_ocr_frame_tag != self.ocr_frame_counter:
                                    fast_text, fast_conf = self.process_plate_ocr_async(plate_crop)
                                    if isinstance(fast_text, str) and fast_text not in ["size_error", "no_result", "low_confidence", "ocr_error"]:
                                        clean_plate_number = self.normalize_plate(fast_text)
                                        if self.is_valid_plate(clean_plate_number):
                                            plate_info['plate_text'] = clean_plate_number
                                            plate_info['is_registered'] = clean_plate_number in self.whitelist
                                        plate_info['ocr_processed'] = True
                                    self.cache_plate_result(bbox_key, (fast_text, fast_conf))
                                    self._fast_ocr_frame_tag = self.ocr_frame_counter
                                if not self.ocr_queue.full() and should_process_ocr:
                                    try:
                                        self.ocr_queue.put_nowait((plate_crop.copy(), bbox_key, self.ocr_frame_counter))
                                    except:
                                        pass
                        plate_detections.append(plate_info)
        return plate_detections

    def get_cached_plate_result(self, bbox_key):
        current_time = time.time()
        if bbox_key in self.plate_cache:
            ent = self.plate_cache[bbox_key]
            if current_time - ent['timestamp'] < self.plate_cache_timeout:
                return ent['result']
            else:
                del self.plate_cache[bbox_key]
        try:
            x1, y1, x2, y2 = map(int, bbox_key.split('_'))
        except:
            return None
        tolerance = 15
        for cached_key, cache_entry in list(self.plate_cache.items()):
            if current_time - cache_entry['timestamp'] >= self.plate_cache_timeout:
                del self.plate_cache[cached_key]
                continue
            try:
                cx1, cy1, cx2, cy2 = map(int, cached_key.split('_'))
                if (abs(x1-cx1) <= tolerance and abs(y1-cy1) <= tolerance and
                    abs(x2-cx2) <= tolerance and abs(y2-cy2) <= tolerance):
                    return cache_entry['result']
            except:
                continue
        return None

    def cache_plate_result(self, bbox_key, result):
        self.plate_cache[bbox_key] = {'result': result, 'timestamp': time.time()}

    def start_ocr_threads(self):
        for _ in range(2):
            t = threading.Thread(target=self.ocr_worker, daemon=True)
            t.start()
            self.ocr_threads.append(t)

    def ocr_worker(self):
        while self.ocr_running:
            try:
                task = self.ocr_queue.get(timeout=1)
                if task is None:
                    break
                crop_img, bbox_id, frame_idx = task
                result = self.process_plate_ocr_async(crop_img)
                self.ocr_result_queue.put((bbox_id, result, frame_idx))
                self.ocr_queue.task_done()
            except:
                pass

    def process_plate_ocr_async(self, crop_img):
        try:
            if crop_img.shape[0] < 20 or crop_img.shape[1] < 60:
                return "size_error", 0.0
            height, width = crop_img.shape[:2]
            if height > 64:
                scale = 64 / height
                new_width = int(width * scale)
                crop_img = cv2.resize(crop_img, (new_width, 64))
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
            enhanced = clahe.apply(gray)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            result = self.ocr.ocr(rgb, cls=True)
            if not result or not result[0]:
                return "no_result", 0.0
            best_text = ""
            best_conf = 0.0
            for line in result:
                if not line:
                    continue
                for item in line:
                    if len(item) >= 2 and item[1]:
                        txt, conf = item[1]
                        if len(txt) >= 4 and conf > best_conf:
                            best_text = txt
                            best_conf = conf
            if best_conf < 0.3:
                return "low_confidence", best_conf
            return best_text, best_conf
        except:
            return "ocr_error", 0.0

    def clean_plate_text(self, text):
        import re
        return re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text.upper())

    def normalize_plate(self, text):
        t = self.clean_plate_text(text)
        trans = str.maketrans({'O':'0','I':'1','Z':'2','S':'5','B':'8'})
        return t.translate(trans)

    def is_valid_plate(self, text):
        t = self.normalize_plate(text)
        if len(t) < 6 or len(t) > 8:
            return False
        prov_set = set("京津沪渝辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙晋宁新港澳")
        if t[0] not in prov_set and t[0] not in "使领学警":
            return False
        import re
        if not re.match(r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{4,6}$', t):
            return False
        return True

    def save_anomaly_screenshot(self, frame, anomaly_type):
        now = time.time()
        if anomaly_type in self.last_screenshot_time and now - self.last_screenshot_time[anomaly_type] < self.screenshot_interval:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.history_folder, f"{anomaly_type}_{ts}.jpg")
        cv2.imwrite(path, frame)
        self.last_screenshot_time[anomaly_type] = now
        return path

    def draw_detections(self, frame, anomalies, all_detections=None):
        drawn_boxes = set()
        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                box_key = f"{x1}_{y1}_{x2}_{y2}"
                if box_key in drawn_boxes:
                    continue
                drawn_boxes.add(box_key)
                colors = {
                    'fire': (0, 0, 255),
                    'smoke': (0, 0, 255),
                    'unknown_person': (0, 0, 255),
                    'unknown_vehicle': (0, 0, 255)
                }
                color = colors.get(anomaly['type'], (255,255,255))
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                if anomaly['type'] == 'unknown_person':
                    label = f"未知人员: {anomaly.get('person_name','未知')} ({anomaly.get('recognition_distance',0.0):.2f})"
                elif anomaly['type'] == 'unknown_vehicle':
                    label = f"未知车辆: {anomaly.get('plate_text','未知')}"
                elif anomaly['type'] == 'fire':
                    label = f"火焰: {anomaly['confidence']:.2f}"
                elif anomaly['type'] == 'smoke':
                    label = f"烟雾: {anomaly['confidence']:.2f}"
                else:
                    label = f"{anomaly['type']}: {anomaly['confidence']:.2f}"
                frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=color)
        if all_detections:
            for detection in all_detections:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    box_key = f"{x1}_{y1}_{x2}_{y2}"
                    if box_key in drawn_boxes:
                        continue
                    if detection['type'] == 'known_person':
                        drawn_boxes.add(box_key)
                        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                        label = f"已知人员: {detection.get('person_name','已知')} ({detection.get('recognition_distance',0.0):.2f})"
                        frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=(0,255,0))
                    elif detection['type'] == 'vehicle_plate':
                        plate_text = detection.get('plate_text', '')
                        is_registered = detection.get('is_registered', False)
                        ocr_processed = detection.get('ocr_processed', False)
                        if ocr_processed and is_registered and plate_text not in ["size_error","no_result","low_confidence","ocr_error"]:
                            drawn_boxes.add(box_key)
                            clean_text = self.normalize_plate(plate_text)
                            label = f"{clean_text} (已登记)"
                            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=(0,255,0))
                        elif not ocr_processed:
                            drawn_boxes.add(box_key)
                            cv2.rectangle(frame, (x1,y1),(x2,y2), (128,128,128), 2)
                            frame = self.draw_chinese_text(frame, "检测中...", (x1, y1-30), font_size=24, color=(128,128,128))
                        elif ocr_processed and plate_text in ["size_error","no_result","low_confidence","ocr_error"]:
                            drawn_boxes.add(box_key)
                            error_messages = {
                                "size_error": "图像太小",
                                "no_result": "无识别结果",
                                "low_confidence": "置信度低",
                                "ocr_error": "识别失败"
                            }
                            label = error_messages.get(plate_text, "识别失败")
                            cv2.rectangle(frame, (x1,y1),(x2,y2), (100,100,100), 2)
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=(100,100,100))
        return frame

    def publish_anomalies(self, anomalies):
        anomaly_list = AnomalyList()
        anomaly_list.header.stamp = self.get_clock().now().to_msg()
        anomaly_list.header.frame_id = "camera_frame"
        anomaly_list.total_count = len(anomalies)
        anomaly_list.frame_timestamp = time.time()
        for a in anomalies:
            msg = AnomalyDetection()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            msg.anomaly_type = a['type']
            msg.timestamp = time.time()
            msg.confidence = a['confidence']
            if 'position' in a:
                msg.position.x = float(a['position'][0])
                msg.position.y = float(a['position'][1])
            msg.image_path = a.get('image_path', '')
            msg.description = a.get('description', '')
            anomaly_list.anomalies.append(msg)
        self.anomaly_publisher.publish(anomaly_list)

    def publish_alerts(self, anomalies):
        import random
        current_time = time.time()
        custom_descriptions = {
            'fire': '已造成火情，建议立即前往处理',
            'smoke': '烟雾可能造成火灾隐患，建议尽快前往查看',
            'unknown_person': '非公司员工或已登记来访人员，建议尽快前往询问情况',
            'unknown_vehicle': '非公司已登记车辆，建议尽快前往查看情况'
        }
        for a in anomalies:
            atype = a['type']
            if atype == 'fire' and current_time - self.last_fire_alert_time < self.fire_smoke_cooldown:
                continue
            if atype == 'smoke' and current_time - self.last_smoke_alert_time < self.fire_smoke_cooldown:
                continue
            if atype == 'unknown_person' and current_time - self.last_face_alert_time < self.face_cooldown:
                continue
            if atype == 'unknown_vehicle' and current_time - self.last_plate_alert_time < self.plate_cooldown:
                continue
            if atype == 'fire':
                self.last_fire_alert_time = current_time
            elif atype == 'smoke':
                self.last_smoke_alert_time = current_time
            elif atype == 'unknown_person':
                self.last_face_alert_time = current_time
            elif atype == 'unknown_vehicle':
                self.last_plate_alert_time = current_time
            lat = round(random.uniform(39.9000,39.9100),6)
            lon = round(random.uniform(116.3000,116.4000),6)
            img_path = a.get('image_path','')
            b64 = ""
            if img_path and os.path.exists(img_path):
                with open(img_path,'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
            alert_data = {
                "type": atype,
                "confidence": a['confidence'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": custom_descriptions.get(atype, a.get('description','')),
                "position": {"latitude": lat, "longitude": lon},
                "screenshot": b64
            }
            self.alert_publisher.publish(String(data=json.dumps(alert_data, ensure_ascii=False)))

    def cleanup(self):
        self.webrtc_running = False
        if self.webrtc_loop is not None and self.webrtc_loop.is_running():
            self.webrtc_loop.call_soon_threadsafe(self.webrtc_loop.stop)
        if hasattr(self, 'webrtc_thread') and self.webrtc_thread.is_alive():
            self.webrtc_thread.join(timeout=2.0)
        self.ocr_running = False
        for _ in range(len(self.ocr_threads)):
            try:
                self.ocr_queue.put(None, timeout=0.5)
            except:
                pass
        self.is_stopping_robot = False
        if self.show_display:
            cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fire-model', default='/home/jetson/ros2_ws/src/example_python/models/best.pt')
    parser.add_argument('--plate-model', default='/home/jetson/ros2_ws/src/example_python/models/plate_best.pt')
    parser.add_argument('--face-model', default='/home/jetson/ros2_ws/src/example_python/models/yolov8n-face.pt')
    parser.add_argument('--flame-conf', type=float, default=0.7)
    parser.add_argument('--smoke-conf', type=float, default=0.7)
    parser.add_argument('--face-conf', type=float, default=0.7)
    parser.add_argument('--plate-conf', type=float, default=0.7)
    parser.add_argument('--target-timeout', type=float, default=20.0)
    parser.add_argument('--display', action='store_true', help='启用本地全屏显示窗口')
    parser.add_argument('--enable-ptz', action='store_true', help='启用云台自动跟踪与变焦')
    return parser.parse_args()

def main():
    rclpy.init(args=sys.argv)
    args = parse_args()
    node = None
    try:
        node = AnomalyDetectionPublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n🛑 停止")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.cleanup()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
