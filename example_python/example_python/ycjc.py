#!/usr/bin/env python3
"""
智能巡检机器人异常检测节点（订阅 ROS 图像话题版 + WebRTC 发布处理结果）
优化：缩小图像进行检测，WebRTC 推送原始分辨率（带检测框）。
本地显示自动充满屏幕（全屏显示，图像拉伸适配）。
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from base_interfaces.msg import AnomalyDetection, AnomalyList
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
    print("✅ 检测模型库加载成功")
except ImportError as e:
    print(f"⚠️ 检测模型库导入失败: {e}")

# ========== 异常目标跟踪器（不变） ==========
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
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return f"vehicle_pos_{center_x//50}_{center_y//50}"
        elif target_type == 'unknown_person':
            person_name = anomaly.get('person_name', '')
            if person_name and person_name != "未知" and person_name != "特征提取失败":
                return f"person_{person_name}"
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                similar_id = self.find_similar_unknown_person(anomaly)
                if similar_id:
                    return similar_id
                return f"unknown_person_{center_x//30}_{center_y//30}"
        elif target_type in ['fire', 'smoke']:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return f"{target_type}_pos_{center_x//30}_{center_y//30}"
        bbox = anomaly.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            return f"{target_type}_{center_x//50}_{center_y//50}"
        return f"{target_type}_unknown"

    def should_publish_alert(self, anomaly):
        current_time = time.time()
        target_id = self.generate_target_id(anomaly)
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.cleanup_disappeared_targets()
            self.last_cleanup_time = current_time
        if target_id in self.tracked_targets:
            target_info = self.tracked_targets[target_id]
            target_info['last_seen'] = current_time
            if target_info['alert_published']:
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
        for target_id in list(self.tracked_targets.keys()):
            if current_time - self.tracked_targets[target_id]['last_seen'] > self.disappear_timeout:
                print(f"🗑️ 清理消失目标: {target_id}")
                del self.tracked_targets[target_id]

    def find_similar_unknown_person(self, current_anomaly):
        if current_anomaly['type'] != 'unknown_person':
            return None
        current_bbox = current_anomaly.get('bbox', [])
        if len(current_bbox) != 4:
            return None
        current_center_x = (current_bbox[0] + current_bbox[2]) // 2
        current_center_y = (current_bbox[1] + current_bbox[3]) // 2
        position_threshold = 80
        for target_id, target_info in self.tracked_targets.items():
            if (target_info['type'] == 'unknown_person' and
                target_id.startswith('unknown_person_')):
                parts = target_id.split('_')
                if len(parts) >= 4:
                    try:
                        tracked_x = int(parts[2]) * 30
                        tracked_y = int(parts[3]) * 30
                        distance = ((current_center_x - tracked_x) ** 2 +
                                   (current_center_y - tracked_y) ** 2) ** 0.5
                        if distance < position_threshold:
                            return target_id
                    except:
                        continue
        return None

    def get_tracking_info(self):
        return {
            'total_targets': len(self.tracked_targets),
            'targets': {tid: {'type': info['type']} for tid, info in self.tracked_targets.items()}
        }


# ========== WebRTC 视频轨道（发送处理后的原始分辨率帧） ==========
class ProcessedFrameTrack(VideoStreamTrack):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.frame_count = 0
        self.start_time = time.time()
        self._last_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        try:
            frame_bgr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.node.thread_pool,
                    self.node.get_latest_processed_frame
                ),
                timeout=0.1
            )
            self._last_frame = frame_bgr
        except (asyncio.TimeoutError, Exception):
            if self._last_frame is not None:
                frame_bgr = self._last_frame
            else:
                frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"WebRTC 发送帧率: {fps:.2f} fps")
        return video_frame


# ========== ROS2 节点（检测缩小，显示充满屏幕） ==========
class AnomalyDetectionPublisher(Node):
    def __init__(self, args):
        super().__init__('anomaly_detection_publisher')

        self.bridge = CvBridge()
        self.anomaly_publisher = self.create_publisher(AnomalyList, '/anomaly_detection/data', 10)
        self.alert_publisher = self.create_publisher(String, '/anomaly_detection/alerts', 10)

        self.args = args
        self.flame_conf_threshold = args.flame_conf
        self.smoke_conf_threshold = args.smoke_conf
        self.face_conf_threshold = args.face_conf
        self.plate_conf_threshold = args.plate_conf
        self.show_display = args.display                     # 本地显示开关

        # 检测用缩小尺寸（保持比例，宽度为640）
        self.detect_width = 640

        # ---------- 获取屏幕尺寸（用于全屏显示）----------
        try:
            import tkinter as tk
            root = tk.Tk()
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
            print(f"🖥️ 屏幕分辨率: {self.screen_width}x{self.screen_height}")
        except:
            self.screen_width = 1920
            self.screen_height = 1080
            print(f"⚠️ 无法获取屏幕分辨率，默认使用 1920x1080")

        self.models = {}
        self.last_screenshot_time = {}
        self.screenshot_interval = 20
        self.target_tracker = AnomalyTargetTracker(disappear_timeout=args.target_timeout)

        # 冷却时间
        self.fire_smoke_cooldown = 10.0
        self.last_fire_alert_time = 0
        self.last_smoke_alert_time = 0
        self.face_cooldown = 5.0
        self.last_face_alert_time = 0
        self.plate_cooldown = 10.0
        self.last_plate_alert_time = 0

        # 车牌缓存与 OCR
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

        # ---------- WebRTC 相关 ----------
        self.latest_processed_frame = None
        self.processed_frame_lock = threading.Lock()
        self.webrtc_running = True
        self.webrtc_loop = None
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        self.webrtc_thread = threading.Thread(target=self.run_webrtc_server, daemon=True)
        self.webrtc_thread.start()
        self.get_logger().info("WebRTC 信令服务器已启动 (ws://0.0.0.0:8080)")

        self.init_detection_models()
        self.load_whitelist()
        self.setup_chinese_font()

        print(f"🎯 置信度阈值: 火焰={self.flame_conf_threshold}, 烟雾={self.smoke_conf_threshold}, "
              f"人脸={self.face_conf_threshold}, 车牌={self.plate_conf_threshold}")

        self.image_sub = self.create_subscription(
            Image,
            '/hik_camera/image_raw',
            self.image_callback,
            10
        )
        self.get_logger().info("已订阅 /hik_camera/image_raw")
        self.frame_count = 0

    # ---------- WebRTC 服务器（不变）----------
    def run_webrtc_server(self):
        try:
            self.webrtc_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.webrtc_loop)

            async def start_server():
                self.websocket_server = await websockets.serve(
                    self.browser_handler, "0.0.0.0", 8080
                )
                print("✅ WebRTC 信令服务器运行在 ws://0.0.0.0:8080")

            self.webrtc_loop.run_until_complete(start_server())
            while self.webrtc_running:
                self.webrtc_loop.run_until_complete(asyncio.sleep(0.5))
            tasks = asyncio.all_tasks(self.webrtc_loop)
            for t in tasks:
                t.cancel()
            self.webrtc_loop.run_until_complete(asyncio.sleep(0.1))
            self.webrtc_loop.close()
        except Exception as e:
            print(f"❌ WebRTC 服务器启动失败: {e}")

    async def browser_handler(self, websocket):
        client_addr = websocket.remote_address
        print(f"新浏览器连接: {client_addr}")
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
            print(f"ICE state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ["failed", "closed"]:
                await pc.close()

        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "offer":
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await websocket.send(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    }))
                elif msg_type == "candidate":
                    candidate_obj = self.create_ice_candidate(data["candidate"])
                    if candidate_obj:
                        try:
                            await pc.addIceCandidate(candidate_obj)
                        except Exception as e:
                            print(f"添加候选失败: {e}")
                elif msg_type == "bye":
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"浏览器 {client_addr} 断开")
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
            component=component,
            foundation=foundation,
            protocol=protocol,
            priority=priority,
            ip=ip,
            port=port,
            type=typ,
            tcpType=tcp_type,
            relatedAddress=None,
            relatedPort=None,
            candidate=candidate_str,
            sdpMid=sdpMid,
            sdpMLineIndex=sdpMLineIndex
        )

    # ---------- 处理后帧存取 ----------
    def update_processed_frame(self, frame):
        with self.processed_frame_lock:
            self.latest_processed_frame = frame.copy()

    def get_latest_processed_frame(self):
        with self.processed_frame_lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # ---------- 图像回调（开始处理）----------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_frame(cv_image)
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {e}")

    def process_frame(self, frame):
        if frame is None:
            return
        self.frame_count += 1

        try:
            # ---------- 缩小图像用于检测 ----------
            h, w = frame.shape[:2]
            if w > self.detect_width:
                scale = self.detect_width / w
                new_w = self.detect_width
                new_h = int(h * scale)
                small_frame = cv2.resize(frame, (new_w, new_h))
            else:
                scale = 1.0
                small_frame = frame.copy()

            anomalies, all_detections = self.detect_anomalies(small_frame)

            # ---------- 坐标映射回原始分辨率 ----------
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

            # 截图保存（使用原始帧）
            for anomaly in anomalies:
                screenshot_path = self.save_anomaly_screenshot(frame, anomaly['type'])
                anomaly['image_path'] = screenshot_path or ''

            # 在原始帧上绘制检测结果
            display_frame = frame.copy()
            display_frame = self.draw_detections(display_frame, anomalies, all_detections)

            # 叠加系统信息
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(display_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(display_frame, f"Anomalies: {len(anomalies)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.putText(display_frame, f"OCR Q:{self.ocr_queue.qsize()} R:{self.ocr_result_queue.qsize()} C:{len(self.plate_cache)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
            tracking_info = self.target_tracker.get_tracking_info()
            cv2.putText(display_frame, f"Tracked: {tracking_info['total_targets']}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

            # 更新 WebRTC 推送帧（原始分辨率带检测框）
            self.update_processed_frame(display_frame)

            # ---------- 本地显示：自动充满屏幕 ----------
            if self.show_display:
                cv2.namedWindow('Anomaly Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Anomaly Detection', self.screen_width, self.screen_height)
                # 拉伸图像以填满整个窗口
                display_resized = cv2.resize(display_frame, (self.screen_width, self.screen_height))
                cv2.imshow('Anomaly Detection', display_resized)
                cv2.waitKey(1)

            # 发布异常数据
            if anomalies:
                self.publish_anomalies(anomalies)
                filtered = [a for a in anomalies if self.target_tracker.should_publish_alert(a)]
                if filtered:
                    self.publish_alerts(filtered)
                    print(f"🎯 过滤后发布 {len(filtered)}/{len(anomalies)} 个异常警报")
                else:
                    print(f"🔄 跳过 {len(anomalies)} 个重复异常警报")
        except Exception as e:
            self.get_logger().error(f"处理帧失败: {e}")

    # ========== 以下为检测模型初始化、OCR、绘制等辅助方法（完整保留） ==========
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
            print(f"使用设备: {device}")
            if self.args.fire_model and os.path.exists(self.args.fire_model):
                self.models['fire'] = YOLO(self.args.fire_model)
                self.models['fire'].to(device)
                print("✅ 火焰烟雾模型加载成功")
            if self.args.plate_model and os.path.exists(self.args.plate_model):
                self.models['plate'] = YOLO(self.args.plate_model)
                self.models['plate'].to(device)
                print("✅ 车牌模型加载成功")
            if self.args.face_model and os.path.exists(self.args.face_model):
                self.models['face'] = YOLO(self.args.face_model)
                self.models['face'].to(device)
                print("✅ 人脸模型加载成功")
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
        except Exception as e:
            print(f"模型加载失败: {e}")

    def load_whitelist(self):
        self.whitelist = set()
        try:
            if os.path.exists(self.whitelist_file):
                with open(self.whitelist_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        plate = line.strip()
                        if plate:
                            clean = self.clean_plate_text(plate)
                            self.whitelist.add(clean)
                print(f"✅ 车牌白名单加载成功，共 {len(self.whitelist)} 个")
        except Exception as e:
            print(f"❌ 白名单加载失败: {e}")

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
                        if img is not None:
                            results = self.models['face'](img, conf=0.5)
                            for r in results:
                                if r.boxes is not None and len(r.boxes)>0:
                                    box = r.boxes[0].xyxy[0].cpu().numpy()
                                    x1,y1,x2,y2 = map(int, box)
                                    feat = self.extract_face_features(img, x1, y1, x2, y2)
                                    if feat is not None:
                                        name = os.path.splitext(fname)[0]
                                        self.known_face_features.append(feat)
                                        self.known_face_names.append(name)
                                        np.save(os.path.join(self.known_faces_folder, f"{name}.npy"), feat)
                                    break
            print(f"✅ 已知人脸特征数: {len(self.known_face_features)}")
        except Exception as e:
            print(f"加载人脸特征失败: {e}")

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
                    print(f"✅ 中文字体加载成功: {fp}")
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
                stroke_color = (0,0,0)
                for adj in range(-1,2):
                    for adj2 in range(-1,2):
                        if adj != 0 or adj2 != 0:
                            draw.text((position[0]+adj, position[1]+adj2), text, font=font, fill=stroke_color)
            draw.text(position, text, font=font, fill=(color[2],color[1],color[0]))
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except:
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return img

    def detect_anomalies(self, frame):
        anomalies = []
        all_detections = []

        if 'fire' in self.models:
            min_conf = min(self.flame_conf_threshold, self.smoke_conf_threshold)
            results = self.models['fire'](frame, conf=min_conf)
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
                        anomaly = {
                            'type': anomaly_type,
                            'confidence': conf,
                            'position': [(x1+x2)//2, (y1+y2)//2],
                            'bbox': [x1, y1, x2, y2],
                            'description': f'{anomaly_type} detected with confidence {conf:.2f}'
                        }
                        anomalies.append(anomaly)

        if 'face' in self.models:
            results = self.models['face'](frame, conf=self.face_conf_threshold)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        face_feature = self.extract_face_features(frame, x1, y1, x2, y2)
                        name, distance = self.recognize_face(face_feature)
                        if name == "未知" or name == "特征提取失败":
                            anomaly = {
                                'type': 'unknown_person',
                                'confidence': conf,
                                'position': [(x1+x2)//2, (y1+y2)//2],
                                'bbox': [x1, y1, x2, y2],
                                'description': f'Unknown person detected with confidence {conf:.2f}, distance: {distance:.2f}',
                                'person_name': name,
                                'recognition_distance': distance
                            }
                            anomalies.append(anomaly)
                        else:
                            known_person = {
                                'type': 'known_person',
                                'confidence': conf,
                                'position': [(x1+x2)//2, (y1+y2)//2],
                                'bbox': [x1, y1, x2, y2],
                                'person_name': name,
                                'recognition_distance': distance
                            }
                            all_detections.append(known_person)

        plate_detections = self.detect_plate_optimized(frame)
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
            if (plate_info['ocr_processed'] and
                plate_info['plate_text'] and
                plate_info['plate_text'] not in ["size_error", "no_result", "low_confidence", "ocr_error"] and
                not plate_info['is_registered']):
                clean_plate_text = self.normalize_plate(plate_info['plate_text'])
                if self.is_valid_plate(clean_plate_text):
                    anomaly = {
                        'type': 'unknown_vehicle',
                        'confidence': plate_info['confidence'],
                        'position': [(x1+x2)//2, (y1+y2)//2],
                        'bbox': bbox,
                        'description': f'Unknown vehicle: {clean_plate_text} (confidence: {plate_info["confidence"]:.2f})',
                        'plate_text': clean_plate_text
                    }
                    anomalies.append(anomaly)

        return anomalies, all_detections

    def detect_plate_optimized(self, frame):
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
            results = self.models['plate'](frame, conf=self.plate_conf_threshold)
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
                            else:
                                plate_info['plate_text'] = ''
                                plate_info['is_registered'] = False
                            plate_info['ocr_processed'] = True
                        elif bbox_key in processed_results:
                            plate_number, ocr_confidence = processed_results[bbox_key]
                            self.cache_plate_result(bbox_key, (plate_number, ocr_confidence))
                            clean_plate_number = self.normalize_plate(plate_number)
                            if self.is_valid_plate(clean_plate_number):
                                plate_info['plate_text'] = clean_plate_number
                                plate_info['is_registered'] = clean_plate_number in self.whitelist
                            else:
                                plate_info['plate_text'] = ''
                                plate_info['is_registered'] = False
                            plate_info['ocr_processed'] = True
                        else:
                            plate_crop = frame[y1:y2, x1:x2]
                            if plate_crop.size > 0:
                                if self._fast_ocr_frame_tag != self.ocr_frame_counter:
                                    fast_text, fast_conf = self.process_plate_ocr_async(plate_crop)
                                    if isinstance(fast_text, str) and fast_text not in ["size_error", "no_result", "low_confidence", "ocr_error"]:
                                        clean_plate_number = self.normalize_plate(fast_text)
                                        if self.is_valid_plate(clean_plate_number):
                                            plate_info['plate_text'] = clean_plate_number
                                            plate_info['is_registered'] = clean_plate_number in self.whitelist
                                        else:
                                            plate_info['plate_text'] = ''
                                            plate_info['is_registered'] = False
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
                if (abs(x1 - cx1) <= tolerance and abs(y1 - cy1) <= tolerance and
                    abs(x2 - cx2) <= tolerance and abs(y2 - cy2) <= tolerance):
                    return cache_entry['result']
            except:
                continue
        return None

    def cache_plate_result(self, bbox_key, result):
        self.plate_cache[bbox_key] = {'result': result, 'timestamp': time.time()}

    def start_ocr_threads(self):
        num_threads = 2
        for _ in range(num_threads):
            t = threading.Thread(target=self.ocr_worker, daemon=True)
            t.start()
            self.ocr_threads.append(t)
        print(f"✅ 启动了 {num_threads} 个OCR异步处理线程")

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
                if not line: continue
                for item in line:
                    if len(item)>=2 and item[1]:
                        txt, conf = item[1]
                        if len(txt)>=4 and conf>best_conf:
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
                    person_name = anomaly.get('person_name', '未知')
                    distance = anomaly.get('recognition_distance', 0.0)
                    label = f"未知人员: {person_name} ({distance:.2f})"
                elif anomaly['type'] == 'unknown_vehicle':
                    plate_text = anomaly.get('plate_text', '未知')
                    label = f"未知车辆: {plate_text}"
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
                        name = detection.get('person_name', '已知')
                        distance = detection.get('recognition_distance', 0.0)
                        label = f"已知人员: {name} ({distance:.2f})"
                        frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=(0,255,0))
                    elif detection['type'] == 'vehicle_plate':
                        plate_text = detection.get('plate_text', '')
                        is_registered = detection.get('is_registered', False)
                        ocr_processed = detection.get('ocr_processed', False)
                        if ocr_processed and is_registered and plate_text not in ["size_error", "no_result", "low_confidence", "ocr_error"]:
                            drawn_boxes.add(box_key)
                            color = (0,255,0)
                            clean_text = self.normalize_plate(plate_text)
                            label = f"{clean_text} (已登记)"
                            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=color)
                        elif not ocr_processed:
                            drawn_boxes.add(box_key)
                            color = (128,128,128)
                            label = "检测中..."
                            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=color)
                        elif ocr_processed and plate_text in ["size_error", "no_result", "low_confidence", "ocr_error"]:
                            drawn_boxes.add(box_key)
                            color = (100,100,100)
                            error_messages = {
                                "size_error": "图像太小",
                                "no_result": "无识别结果",
                                "low_confidence": "置信度低",
                                "ocr_error": "识别失败"
                            }
                            label = error_messages.get(plate_text, "识别失败")
                            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=24, color=color)
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
        print(f"📡 发布异常数据: {len(anomalies)} 个异常")

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
                "description": custom_descriptions.get(atype, a['description']),
                "position": {"latitude": lat, "longitude": lon},
                "screenshot": b64
            }
            self.alert_publisher.publish(String(data=json.dumps(alert_data, ensure_ascii=False)))
            screenshot_info = f" (含截图: {len(b64) > 0})" if b64 else " (无截图)"
            print(f"🚨 发布警报: {atype} (置信度: {a['confidence']:.3f}){screenshot_info}")

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
        if self.show_display:
            cv2.destroyAllWindows()
        self.get_logger().info("资源已清理")


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
    parser.add_argument('--display', action='store_true', help='启用本地显示窗口（自动充满屏幕）')
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
