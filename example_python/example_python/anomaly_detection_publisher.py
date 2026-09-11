#!/usr/bin/env python3
"""
智能巡检机器人异常检测发布节点
集成火焰、烟雾、人脸、车牌检测功能
支持实时画面显示、异常数据发布和截图保存
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
from pathlib import Path
from queue import Queue
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 添加anaconda环境路径以正确导入torch
sys.path.append('/home/jetson/anaconda3/envs/retest/lib/python3.10/site-packages/')

# 异常检测相关库
try:
    import torch
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    import dlib
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
    print("✅ 检测模型库加载成功")
except ImportError as e:
    print(f"⚠️  检测模型库导入失败: {e}")
    print("请安装必要的依赖包")
    


class AnomalyTargetTracker:
    """异常目标跟踪器，用于避免重复发布同一目标的警报"""
    
    def __init__(self, disappear_timeout=20.0):
        self.disappear_timeout = disappear_timeout  # 目标消失超时时间（秒）
        self.tracked_targets = {}  # 跟踪的目标状态 {target_id: target_info}
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 5.0  # 清理间隔（秒）
        
    def generate_target_id(self, anomaly):
        """生成目标唯一标识符"""
        target_type = anomaly['type']
        
        if target_type == 'unknown_vehicle':
            # 车牌目标：使用车牌号作为唯一标识
            plate_text = anomaly.get('plate_text', '')
            if plate_text:
                return f"vehicle_{plate_text}"
            else:
                # 如果没有车牌号，使用位置区域
                bbox = anomaly.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    return f"vehicle_pos_{center_x//50}_{center_y//50}"  # 50像素网格
                
        elif target_type == 'unknown_person':
            # 人脸目标：尝试使用人脸特征匹配已追踪的人脸
            person_name = anomaly.get('person_name', '')
            if person_name and person_name != "未知" and person_name != "特征提取失败":
                # 如果有识别结果，使用人名作为标识
                return f"person_{person_name}"
            else:
                # 对于未知人脸，尝试与已追踪的人脸进行特征匹配
                bbox = anomaly.get('bbox', [])
                if len(bbox) == 4:
                    # 首先检查是否有与现有目标相似的人脸
                    similar_target_id = self.find_similar_unknown_person(anomaly)
                    if similar_target_id:
                        return similar_target_id
                    
                    # 如果没有相似的，生成新的ID
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    return f"unknown_person_{center_x//30}_{center_y//30}"  # 30像素网格，更精细
                
        elif target_type in ['fire', 'smoke']:
            # 火焰/烟雾目标：使用位置区域
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return f"{target_type}_pos_{center_x//30}_{center_y//30}"  # 30像素网格
        
        # 默认使用类型+位置
        bbox = anomaly.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            return f"{target_type}_{center_x//50}_{center_y//50}"
        
        return f"{target_type}_unknown"
    
    def should_publish_alert(self, anomaly):
        """判断是否应该发布警报"""
        current_time = time.time()
        target_id = self.generate_target_id(anomaly)
        
        # 定期清理过期目标
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.cleanup_disappeared_targets()
            self.last_cleanup_time = current_time
        
        # 检查目标是否已经发布过警报
        if target_id in self.tracked_targets:
            target_info = self.tracked_targets[target_id]
            # 更新最后发现时间
            target_info['last_seen'] = current_time
            # 如果已经发布过警报，不再发布
            if target_info['alert_published']:
                return False
        
        # 新目标或重新出现的目标，可以发布警报
        self.tracked_targets[target_id] = {
            'type': anomaly['type'],
            'first_seen': current_time,
            'last_seen': current_time,
            'alert_published': True,  # 标记为已发布
            'anomaly_info': anomaly
        }
        
        return True
    
    def cleanup_disappeared_targets(self):
        """清理消失的目标"""
        current_time = time.time()
        targets_to_remove = []
        
        for target_id, target_info in self.tracked_targets.items():
            # 如果目标消失超过设定时间，则清理
            if current_time - target_info['last_seen'] > self.disappear_timeout:
                targets_to_remove.append(target_id)
                print(f"🗑️  清理消失目标: {target_id} (类型: {target_info['type']})")
        
        for target_id in targets_to_remove:
            del self.tracked_targets[target_id]
    
    def find_similar_unknown_person(self, current_anomaly):
        """查找相似的未知人脸目标"""
        if current_anomaly['type'] != 'unknown_person':
            return None
        
        current_bbox = current_anomaly.get('bbox', [])
        if len(current_bbox) != 4:
            return None
        
        current_center_x = (current_bbox[0] + current_bbox[2]) // 2
        current_center_y = (current_bbox[1] + current_bbox[3]) // 2
        
        # 查找相似位置的未知人脸目标
        position_threshold = 80  # 80像素的位置容差
        
        for target_id, target_info in self.tracked_targets.items():
            if (target_info['type'] == 'unknown_person' and 
                target_id.startswith('unknown_person_')):
                
                # 解析已追踪目标的位置
                parts = target_id.split('_')
                if len(parts) >= 4:
                    try:
                        tracked_x = int(parts[2]) * 30  # 恢复实际坐标
                        tracked_y = int(parts[3]) * 30
                        
                        # 计算位置距离
                        distance = ((current_center_x - tracked_x) ** 2 + 
                                   (current_center_y - tracked_y) ** 2) ** 0.5
                        
                        if distance < position_threshold:
                            return target_id
                    except (ValueError, IndexError):
                        continue
        
        return None

    def get_tracking_info(self):
        """获取跟踪信息用于调试"""
        return {
            'total_targets': len(self.tracked_targets),
            'targets': {tid: {
                'type': info['type'],
                'duration': time.time() - info['first_seen'],
                'last_seen_ago': time.time() - info['last_seen']
            } for tid, info in self.tracked_targets.items()}
        }


class AnomalyDetectionPublisher(Node):
    def __init__(self, args):
        super().__init__('anomaly_detection_publisher')
        
        # 添加图像转换桥接器
        self.bridge = CvBridge()
        
        # 创建订阅者替代原有的相机初始化
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10  # 队列大小
        )
        
        self.processed_image_publisher = self.create_publisher(Image, '/web_image/processed', 10)
        
        
        # 添加新变量
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 创建发布者
        self.image_publisher = self.create_publisher(String, '/web_image/compressed', 10)
        self.anomaly_publisher = self.create_publisher(AnomalyList, '/anomaly_detection/data', 10)
        self.alert_publisher = self.create_publisher(String, '/anomaly_detection/alerts', 10)
        
        # 参数配置
        self.args = args
        #self.camera_source = args.source
        
        # 分别设置不同目标类型的置信度阈值
        self.flame_conf_threshold = args.flame_conf  # 火焰置信度阈值
        self.smoke_conf_threshold = args.smoke_conf  # 烟雾置信度阈值
        self.face_conf_threshold = args.face_conf   # 人脸置信度阈值
        self.plate_conf_threshold = args.plate_conf  # 车牌置信度阈值
        
        self.show_display = True  # 默认显示实时画面
        
        # 相机相关
        self.cap = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # 异常检测相关
        self.models = {}
        self.last_screenshot_time = {}  # 记录上次截图时间，防止重复截图发送
        self.screenshot_interval = 20  # 截图间隔（秒）
        
        # 异常目标跟踪器
        self.target_tracker = AnomalyTargetTracker(disappear_timeout=args.target_timeout)
        
        # 火焰和烟雾检测冷却时间控制
        self.fire_smoke_cooldown = 10.0  # 20秒冷却时间
        self.last_fire_alert_time = 0  # 上次火焰警报时间
        self.last_smoke_alert_time = 0  # 上次烟雾警报时间
        
        # 人脸检测冷却时间控制
        self.face_cooldown = 5.0  # 5秒冷却时间
        self.last_face_alert_time = 0  # 上次人脸警报时间
        
        # 车牌检测冷却时间控制
        self.plate_cooldown = 10.0  # 10秒冷却时间
        self.last_plate_alert_time = 0  # 上次车牌警报时间
        
        # 车牌检测优化相关 - 参考my_predict.py的实现
        self.plate_cache = {}  # 车牌识别缓存
        self.plate_cache_timeout = 5.0  # 缓存超时时间（秒）
        self.last_plate_ocr_time = 0
        self.plate_ocr_interval = 0.5  # OCR间隔（秒）
        self.plate_detection_results = []  # 存储车牌检测结果
        
        # 异步OCR处理（优化延迟）
        self.ocr_queue = Queue(maxsize=4)
        self.ocr_result_queue = Queue()
        self.ocr_threads = []
        self.ocr_running = True
        self.ocr_frame_counter = 0
        # 提高识别频率：每帧尝试一次OCR
        self.ocr_skip_frames = 1
        # 低开销日志与快速同步识别帧标记
        self.debug_ocr = False
        self._fast_ocr_frame_tag = -1
        
        # 网格对齐参数（提高缓存命中率）
        self.grid_size = 10
        
        # 创建必要的文件夹
        self.history_folder = "history_anomaly"
        self.known_faces_folder = "/home/jetson/ros2_ws/src/example_python/data/known_faces"
        # 使用绝对路径或在ROS2包中查找whitelist文件
        self.whitelist_file = self.find_whitelist_file()
        os.makedirs(self.history_folder, exist_ok=True)
        os.makedirs(self.known_faces_folder, exist_ok=True)
        
        # 初始化检测组件
        self.init_detection_models()
        self.load_whitelist()
        
        # 设置中文字体
        self.setup_chinese_font()
        
        self.ptz_pub = self.create_publisher(String, '/ptz/move', 10)
        
        # 输出置信度阈值设置信息
        print(f"🎯 置信度阈值设置:")
        print(f"   - 火焰检测: {self.flame_conf_threshold}")
        print(f"   - 烟雾检测: {self.smoke_conf_threshold}")
        print(f"   - 人脸检测: {self.face_conf_threshold}")
        print(f"   - 车牌检测: {self.plate_conf_threshold}")
        
        
    def image_callback(self, msg):
        """处理接收到的图像消息"""
        print(123)
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # 更新最新帧
            with self.frame_lock:
                self.latest_frame = cv_image
                
            # 立即处理帧
            self.process_frame(cv_image)
            
        except Exception as e:
            self.get_logger().error(f"图像处理失败: {str(e)}")
            
    def process_frame(self, frame):
        """处理图像帧（替代原来的timer_callback逻辑）"""
        if frame is None:
            return
            
        self.frame_count += 1
        current_time = time.time()
        
        try:
            # 以下是原timer_callback中的处理逻辑，保持不变
            anomalies, all_detections = self.detect_anomalies(frame)
            
            # 保存异常截图
            for anomaly in anomalies:
                screenshot_path = self.save_anomaly_screenshot(frame, anomaly['type'])
                if screenshot_path:
                    anomaly['image_path'] = screenshot_path
                else:
                    anomaly['image_path'] = ''
            
            # 在画面上绘制检测结果
            display_frame = frame.copy()
            display_frame = self.draw_detections(display_frame, anomalies, all_detections)
            
                        # 添加系统信息
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Anomalies: {len(anomalies)}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 添加OCR性能信息
            ocr_queue_size = self.ocr_queue.qsize()
            ocr_result_size = self.ocr_result_queue.qsize()
            cache_size = len(self.plate_cache)
            cv2.putText(display_frame, f"OCR Q:{ocr_queue_size} R:{ocr_result_size} C:{cache_size}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 添加目标跟踪器信息
            tracking_info = self.target_tracker.get_tracking_info()
            tracked_count = tracking_info['total_targets']
            cv2.putText(display_frame, f"Tracked: {tracked_count}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 显示实时画面
            #if self.show_display:
                #cv2.imshow('Anomaly Detection - Live Feed', display_frame)
                #cv2.waitKey(1)
            
            # 发布图像数据
            self.publish_image(display_frame)
            
            # 发布异常数据（所有检测到的异常都发布，用于实时显示）
            if anomalies:
                self.publish_anomalies(anomalies)
                
                # 过滤需要发布警报的异常（避免重复警报）
                filtered_anomalies = []
                for anomaly in anomalies:
                    if self.target_tracker.should_publish_alert(anomaly):
                        filtered_anomalies.append(anomaly)
                
                # 只发布新的或重新出现的异常警报
                if filtered_anomalies:
                    self.publish_alerts(filtered_anomalies)
                    print(f"🎯 过滤后发布 {len(filtered_anomalies)}/{len(anomalies)} 个异常警报")
                else:
                    print(f"🔄 跳过 {len(anomalies)} 个重复异常警报")
            
        except Exception as e:
            self.get_logger().error(f"处理失败: {e}")

    def find_whitelist_file(self):
        """查找白名单文件，优先在ROS2包目录中查找"""
        possible_paths = [
            "/home/jetson/ros2_ws/src/example_python/data/whitelist/whitelist.txt",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 找到白名单文件: {path}")
                return path
        
        # 如果都找不到，返回默认路径
        print("⚠️  未找到白名单文件，将使用默认路径")
        return "whitelist.txt"

    def find_dlib_model_paths(self):
        """查找dlib模型文件，优先在ROS2包目录中查找"""
        # 可能的dlib模型路径
        possible_base_paths = [
            "/home/jetson/ros2_ws/install/example_python/share/example_python/models",
            "/home/jetson/ros2_ws/src/example_python/models",
            "models"
        ]
        
        shape_predictor_filename = "shape_predictor_68_face_landmarks.dat"
        face_recognition_filename = "dlib_face_recognition_resnet_model_v1.dat"
        
        for base_path in possible_base_paths:
            shape_predictor_path = os.path.join(base_path, shape_predictor_filename)
            face_recognition_path = os.path.join(base_path, face_recognition_filename)
            
            if os.path.exists(shape_predictor_path) and os.path.exists(face_recognition_path):
                print(f"✅ 找到dlib模型文件: {base_path}")
                return shape_predictor_path, face_recognition_path
        
        print("⚠️  未找到dlib模型文件")
        return None, None

    def init_detection_models(self):
        """初始化检测模型"""
        try:
            print("🔄 正在加载检测模型...")
            
            # 检查CUDA可用性
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("⚠️  使用CPU，检测速度可能较慢")
            
            # 加载火焰烟雾检测模型
            if self.args.fire_model and os.path.exists(self.args.fire_model):
                self.models['fire'] = YOLO(self.args.fire_model)
                self.models['fire'].to(device)
                print(f"✅ 火焰烟雾模型加载成功: {self.args.fire_model}")
            else:
                print("❌ 火焰烟雾模型文件未找到")
            
            # 加载车牌检测模型
            if self.args.plate_model and os.path.exists(self.args.plate_model):
                self.models['plate'] = YOLO(self.args.plate_model)
                self.models['plate'].to(device)
                print(f"✅ 车牌检测模型加载成功: {self.args.plate_model}")
            else:
                print("❌ 车牌检测模型文件未找到")
            
            # 加载人脸检测模型
            if self.args.face_model and os.path.exists(self.args.face_model):
                self.models['face'] = YOLO(self.args.face_model)
                self.models['face'].to(device)
                print(f"✅ 人脸检测模型加载成功: {self.args.face_model}")
                
                # 加载dlib人脸识别模型
                try:
                    # 使用更智能的路径查找
                    dlib_shape_predictor_path, dlib_face_recognition_model_path = self.find_dlib_model_paths()
                    
                    if dlib_shape_predictor_path and dlib_face_recognition_model_path:
                        self.shape_predictor = dlib.shape_predictor(dlib_shape_predictor_path)
                        self.face_recognition_model = dlib.face_recognition_model_v1(dlib_face_recognition_model_path)
                        print("✅ dlib人脸识别模型加载成功")
                        
                        # 加载已知人脸特征
                        self.load_known_face_features()
                    else:
                        print("❌ dlib模型文件未找到，将使用简化人脸识别")
                        self.shape_predictor = None
                        self.face_recognition_model = None
                except Exception as e:
                    print(f"❌ dlib模型加载失败: {e}")
                    self.shape_predictor = None
                    self.face_recognition_model = None
            else:
                print("❌ 人脸检测模型文件未找到")
            
            # 初始化OCR（用于车牌识别）- 优化配置
            self.ocr = PaddleOCR(
                use_angle_cls=False,  # 禁用角度分类器以提高速度
                lang='ch',
                use_gpu=True,  # 使用GPU加速
                show_log=False  # 关闭日志输出
                #device = 'cpu',
            )
            print("✅ OCR初始化成功")
            
            # 启动异步OCR处理线程
            self.start_ocr_threads()
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def load_whitelist(self):
        """加载车牌白名单"""
        self.whitelist = set()
        self.raw_whitelist = set()  # 保存原始白名单用于调试
        try:
            if os.path.exists(self.whitelist_file):
                with open(self.whitelist_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        plate = line.strip()
                        if plate:
                            self.raw_whitelist.add(plate)
                            # 对白名单中的车牌也进行清理，确保匹配一致性
                            clean_plate = self.clean_plate_text(plate)
                            self.whitelist.add(clean_plate)
                            print(f"✅ 白名单车牌: 原始='{plate}' 清理后='{clean_plate}'")
                print(f"✅ 车牌白名单加载成功，共 {len(self.whitelist)} 个")
            else:
                print("⚠️  车牌白名单文件不存在，所有车牌都将标记为可疑")
        except Exception as e:
            print(f"❌ 车牌白名单加载失败: {e}")

    def load_known_face_features(self):
        """加载已知人脸特征数据"""
        self.known_face_features = []
        self.known_face_names = []
        
        try:
            # 首先尝试加载已保存的特征文件
            for file in os.listdir(self.known_faces_folder):
                if file.endswith(".npy"):
                    feature = np.load(os.path.join(self.known_faces_folder, file))
                    self.known_face_features.append(feature)
                    self.known_face_names.append(os.path.splitext(file)[0])
            
            if self.known_face_features:
                print(f"✅ 从特征文件加载了 {len(self.known_face_features)} 个已知人脸")
                return
            
            # 如果没有特征文件，从图片提取特征
            if not os.path.exists(self.known_faces_folder):
                print("⚠️  已知人脸文件夹不存在")
                return
                
            face_files = [f for f in os.listdir(self.known_faces_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not face_files:
                print("⚠️  未找到已知人脸图片文件")
                return
                
            print(f"🔄 正在提取 {len(face_files)} 个人脸特征...")
            
            for face_file in face_files:
                try:
                    # 加载图片
                    img_path = os.path.join(self.known_faces_folder, face_file)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        print(f"❌ 无法加载图片: {face_file}")
                        continue
                    
                    # 使用YOLO检测人脸
                    results = self.models['face'](img, conf=0.5)
                    
                    face_found = False
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0:
                            # 使用第一个检测到的人脸
                            box = boxes[0]
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # 提取人脸特征
                            feature = self.extract_face_features(img, x1, y1, x2, y2)
                            
                            if feature is not None:
                                self.known_face_features.append(feature)
                                name = os.path.splitext(face_file)[0]
                                self.known_face_names.append(name)
                                
                                # 保存特征文件
                                feature_path = os.path.join(self.known_faces_folder, f"{name}.npy")
                                np.save(feature_path, feature)
                                
                                print(f"✅ 提取人脸特征成功: {name}")
                                face_found = True
                                break
                    
                    if not face_found:
                        print(f"❌ 未在图片中检测到人脸: {face_file}")
                        
                except Exception as e:
                    print(f"❌ 处理人脸图片失败 {face_file}: {e}")
                    
            print(f"✅ 成功加载 {len(self.known_face_features)} 个已知人脸特征")
            
        except Exception as e:
            print(f"❌ 加载已知人脸特征失败: {e}")
            self.known_face_features = []
            self.known_face_names = []

    def extract_face_features(self, image, x1, y1, x2, y2, min_size=60):
        """提取人脸特征"""
        if self.shape_predictor is None or self.face_recognition_model is None:
            return None
            
        try:
            # 裁剪人脸区域
            face_img = image[y1:y2, x1:x2]
            
            if face_img.shape[0] < min_size or face_img.shape[1] < min_size:
                return None
            
            # 转换为RGB格式
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 使用dlib进行特征提取
            rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
            shape = self.shape_predictor(face_img_rgb, rect)
            descriptor = self.face_recognition_model.compute_face_descriptor(face_img_rgb, shape)
            
            return np.array(descriptor)
        except Exception as e:
            print(f"提取人脸特征失败: {e}")
            return None

    def recognize_face(self, feature, threshold=0.5):
        """识别人脸，返回人名和距离"""
        if feature is None:
            return "特征提取失败", 1.0
        
        if not self.known_face_features:
            return "未知", 1.0
        
        distances = [np.linalg.norm(feature - known) for known in self.known_face_features]
        min_dist = min(distances)
        name = self.known_face_names[distances.index(min_dist)] if min_dist < threshold else "未知"
        
        return name, min_dist

    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            # 尝试不同的字体路径
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "/usr/share/fonts/truetype/arphic/ukai.ttc",  # 中文字体
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.chinese_font = ImageFont.truetype(font_path, 20)
                    print(f"✅ 中文字体加载成功: {font_path}")
                    return True
            
            # 如果找不到字体文件，使用默认字体
            self.chinese_font = ImageFont.load_default()
            print("⚠️  使用默认字体，可能不支持中文显示")
            return False
            
        except Exception as e:
            print(f"❌ 字体加载失败: {e}")
            self.chinese_font = ImageFont.load_default()
            return False

    def draw_chinese_text(self, img, text, position, font_size=20, color=(255, 255, 255), bg_color=None):
        """
        在OpenCV图像上绘制中文文本，支持描边效果
        Args:
            img: OpenCV图像 (BGR格式)
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font_size: 字体大小
            color: 文本颜色 (B, G, R)
            bg_color: 背景颜色，None表示透明背景
        Returns:
            修改后的图像
        """
        try:
            # 将OpenCV图像转换为PIL图像
            img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 设置字体
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # 获取文本尺寸
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制背景（如果指定）
            if bg_color is not None:
                bg_position = [
                    position[0] - 2,
                    position[1] - text_height - 2,
                    position[0] + text_width + 2,
                    position[1] + 2
                ]
                draw.rectangle(bg_position, fill=bg_color)
            
            # 转换颜色格式 (PIL使用RGB格式)
            text_color = (color[2], color[1], color[0])  # BGR转RGB
            
            # 如果没有背景色，添加描边效果增强可读性
            if bg_color is None:
                # 绘制黑色描边
                stroke_color = (0, 0, 0)  # 黑色描边
                for adj in range(-1, 2):
                    for adj2 in range(-1, 2):
                        if adj != 0 or adj2 != 0:
                            draw.text((position[0] + adj, position[1] + adj2), text, 
                                    font=font, fill=stroke_color)
            
            # 绘制主要文本
            draw.text(position, text, font=font, fill=text_color)
            
            # 转换回OpenCV格式
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
            
        except Exception as e:
            print(f"中文文本绘制失败: {e}")
            # 如果失败，使用OpenCV的putText（可能显示乱码，但不会崩溃）
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return img

    def detect_plate_optimized(self, frame):
        """优化的车牌检测函数 - 使用异步OCR处理"""
        plate_detections = []
        current_time = time.time()
        
        # 增加帧计数器
        self.ocr_frame_counter += 1
        should_process_ocr = (self.ocr_frame_counter % self.ocr_skip_frames) == 0
        
        # 处理异步OCR结果
        processed_results = {}
        while not self.ocr_result_queue.empty():
            try:
                bbox_id, (plate_number, ocr_confidence), result_frame_idx = self.ocr_result_queue.get_nowait()
                # 缩小时间窗口，减少跨帧等待，降低可见延迟
                if result_frame_idx >= self.ocr_frame_counter - 8:
                    processed_results[bbox_id] = (plate_number, ocr_confidence)
            except:
                break
        
        # 车牌检测
        if 'plate' in self.models:
            results = self.models['plate'](frame, conf=self.plate_conf_threshold)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # 使用网格对齐的边界框标识（提高缓存命中率）
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
                        
                        # 首先检查缓存
                        cached_result = self.get_cached_plate_result(bbox_key)
                        if cached_result:
                            plate_number, ocr_confidence = cached_result
                            # 归一化并校验
                            clean_plate_number = self.normalize_plate(plate_number)
                            if self.is_valid_plate(clean_plate_number):
                                plate_info['plate_text'] = clean_plate_number
                                plate_info['is_registered'] = clean_plate_number in self.whitelist
                            else:
                                plate_info['plate_text'] = ''
                                plate_info['is_registered'] = False
                            plate_info['ocr_processed'] = True
                            if self.debug_ocr and clean_plate_number != plate_number:
                                print(f"🔍 车牌文本清理: 原始='{plate_number}' 清理后='{clean_plate_number}' 是否在白名单={'✅' if plate_info['is_registered'] else '❌'}")
                        elif bbox_key in processed_results:
                            # 使用异步处理的结果
                            plate_number, ocr_confidence = processed_results[bbox_key]
                            self.cache_plate_result(bbox_key, (plate_number, ocr_confidence))
                            # 归一化并校验，尽量避免错误文本显示导致等待
                            clean_plate_number = self.normalize_plate(plate_number)
                            if self.is_valid_plate(clean_plate_number):
                                plate_info['plate_text'] = clean_plate_number
                                plate_info['is_registered'] = clean_plate_number in self.whitelist
                            else:
                                plate_info['plate_text'] = ''
                                plate_info['is_registered'] = False
                            plate_info['ocr_processed'] = True
                            if self.debug_ocr and clean_plate_number != plate_number:
                                print(f"🔍 车牌文本清理: 原始='{plate_number}' 清理后='{clean_plate_number}' 是否在白名单={'✅' if plate_info['is_registered'] else '❌'}")
                        else:
                            # 没有缓存/异步结果：先做一次快速同步识别用于当帧显示，然后再入队异步提升稳健性
                            plate_crop = frame[y1:y2, x1:x2]
                            if plate_crop.size > 0:
                                # 每帧仅对首个目标做一次同步识别，立即得到文本，减少可见延迟
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
                                        # 快速结果也写入缓存，便于同帧其它逻辑复用
                                        self.cache_plate_result(bbox_key, (fast_text, fast_conf))
                                        self._fast_ocr_frame_tag = self.ocr_frame_counter
                                # 若队列未满且当前策略允许，继续提交异步任务做更稳的识别
                                if not self.ocr_queue.full() and should_process_ocr:
                                    try:
                                        self.ocr_queue.put_nowait((plate_crop.copy(), bbox_key, self.ocr_frame_counter))
                                    except:
                                        pass  # 队列满，跳过
                        
                        plate_detections.append(plate_info)
        
        return plate_detections

    def preprocess_plate_image(self, plate_img):
        """预处理车牌图像以提高OCR精度"""
        # 调整大小
        height, width = plate_img.shape[:2]
        if height < 32:
            scale = 32 / height
            new_width = int(width * scale)
            plate_img = cv2.resize(plate_img, (new_width, 32), interpolation=cv2.INTER_CUBIC)
        
        # 转换为灰度图
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def cleanup_plate_cache(self, current_time, max_age=30):
        """清理过期的车牌缓存"""
        keys_to_remove = []
        for key, value in self.plate_cache.items():
            if current_time - value['timestamp'] > max_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.plate_cache[key]

    def detect_anomalies(self, frame):
        """检测异常目标"""
        anomalies = []
        all_detections = []  # 存储所有检测结果，包括已知人脸
        current_time = time.time()
        
        try:
            # 火焰烟雾检测 - 使用最低的阈值进行初步检测，后续根据类别过滤
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
                            
                            # 转换为异常类型并检查对应的置信度阈值
                            if 'fire' in class_name.lower() or '火' in class_name:
                                anomaly_type = 'fire'
                                if conf < self.flame_conf_threshold:
                                    continue  # 置信度不够，跳过
                            elif 'smoke' in class_name.lower() or '烟' in class_name:
                                anomaly_type = 'smoke'
                                if conf < self.smoke_conf_threshold:
                                    continue  # 置信度不够，跳过
                            else:
                                continue
                            
                            # 获取边界框
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            anomaly = {
                                'type': anomaly_type,
                                'confidence': conf,
                                'position': [(x1 + x2) // 2, (y1 + y2) // 2],
                                'bbox': [x1, y1, x2, y2],
                                'description': f'{anomaly_type} detected with confidence {conf:.2f}'
                            }
                            anomalies.append(anomaly)
            
            # 人脸检测与识别
            if 'face' in self.models:
                results = self.models['face'](frame, conf=self.face_conf_threshold)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            # 提取人脸特征并进行识别
                            face_feature = self.extract_face_features(frame, x1, y1, x2, y2)
                            name, distance = self.recognize_face(face_feature)
                            
                            # 只有未知人脸才记录为异常
                            if name == "未知" or name == "特征提取失败":
                                anomaly = {
                                    'type': 'unknown_person',
                                    'confidence': conf,
                                    'position': [(x1 + x2) // 2, (y1 + y2) // 2],
                                    'bbox': [x1, y1, x2, y2],
                                    'description': f'Unknown person detected with confidence {conf:.2f}, distance: {distance:.2f}',
                                    'person_name': name,
                                    'recognition_distance': distance
                                }
                                anomalies.append(anomaly)
                            else:
                                # 已知人脸，记录但不作为异常
                                print(f"✅ 识别到已知人员: {name} (距离: {distance:.2f}, 置信度: {conf:.2f})")
                                known_person = {
                                    'type': 'known_person',
                                    'confidence': conf,
                                    'position': [(x1 + x2) // 2, (y1 + y2) // 2],
                                    'bbox': [x1, y1, x2, y2],
                                    'person_name': name,
                                    'recognition_distance': distance
                                }
                                all_detections.append(known_person)
            
            # 优化的车牌检测
            plate_detections = self.detect_plate_optimized(frame)
            for plate_info in plate_detections:
                bbox = plate_info['bbox']
                x1, y1, x2, y2 = bbox
                
                # 将所有车牌检测结果添加到all_detections中，用于画面显示
                detection_info = {
                    'type': 'vehicle_plate',
                    'confidence': plate_info['confidence'],
                    'position': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'bbox': bbox,
                    'plate_text': plate_info['plate_text'],
                    'is_registered': plate_info['is_registered'],
                    'ocr_processed': plate_info['ocr_processed']
                }
                all_detections.append(detection_info)
                
                # 只有有效OCR结果且未登记的车牌才作为异常
                if (plate_info['ocr_processed'] and 
                    plate_info['plate_text'] and 
                    plate_info['plate_text'] not in ["size_error", "no_result", "low_confidence", "ocr_error"] and
                    not plate_info['is_registered']):
                    
                    # 归一化并合法性校验，仅对合法车牌发布异常
                    clean_plate_text = self.normalize_plate(plate_info['plate_text'])
                    if not self.is_valid_plate(clean_plate_text):
                        continue
                    
                    anomaly = {
                        'type': 'unknown_vehicle',
                        'confidence': plate_info['confidence'],
                        'position': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'bbox': bbox,
                        'description': f'Unknown vehicle: {clean_plate_text} (confidence: {plate_info["confidence"]:.2f})',
                        'plate_text': clean_plate_text
                    }
                    anomalies.append(anomaly)
                    print(f"🚨 检测到未登记车牌: {clean_plate_text}")
            
        except Exception as e:
            print(f"异常检测处理失败: {e}")
        
        return anomalies, all_detections

    def clean_plate_text(self, text):
        """清理车牌文本"""
        if not text:
            return ""
        # 简化的车牌文本清理
        import re
        # 移除所有非中文、大写字母和数字的字符
        cleaned = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text.upper())
        return cleaned

    def normalize_plate(self, text):
        """统一车牌文本：清理并归一常见OCR混淆字符，不影响速度"""
        t = self.clean_plate_text(text)
        # 常见混淆：O->0、I->1、Z->2、S->5、B->8
        trans = str.maketrans({'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'})
        t = t.translate(trans)
        return t

    def is_valid_plate(self, text):
        """严格但高效的大陆车牌合法性校验，避免显示乱码/非车牌文本"""
        t = self.normalize_plate(text)
        if not t:
            return False
        # 长度限制：常见长度6-8（含新能源等）
        if len(t) < 6 or len(t) > 8:
            return False
        # 首位必须是省份汉字或特殊牌照前缀（使/领/学/警）
        prov_set = set("京津沪渝辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙晋宁新港澳")
        if t[0] not in prov_set and t[0] not in "使领学警":
            return False
        # 第二位必须是大写英文字母；后续4-6位为大写字母或数字
        import re
        if not re.match(r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{4,6}$', t):
            return False
        # 中文字符数量不应超过2（考虑部分特殊牌照）
        chinese_count = sum(1 for c in t if '\u4e00' <= c <= '\u9fa5')
        if chinese_count > 2:
            return False
        return True

    def save_anomaly_screenshot(self, frame, anomaly_type):
        """保存异常截图"""
        current_time = time.time()
        
        # 检查是否在截图间隔内
        if anomaly_type in self.last_screenshot_time:
            if current_time - self.last_screenshot_time[anomaly_type] < self.screenshot_interval:
                return None
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{anomaly_type}_{timestamp}.jpg"
        filepath = os.path.join(self.history_folder, filename)
        
        try:
            # 保存截图
            cv2.imwrite(filepath, frame)
            self.last_screenshot_time[anomaly_type] = current_time
            print(f"📸 异常截图已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ 截图保存失败: {e}")
            return None

    def draw_detections(self, frame, anomalies, all_detections=None):
        """在画面上绘制检测结果"""
        
        # 创建已绘制的边界框集合，避免重复绘制
        drawn_boxes = set()
        
        # 先绘制异常检测结果
        for anomaly in anomalies:
            bbox = anomaly.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                box_key = f"{x1}_{y1}_{x2}_{y2}"
                
                # �避重复绘制相同位置的框
                if box_key in drawn_boxes:
                    continue
                drawn_boxes.add(box_key)
                
                # 根据异常类型选择颜色
                colors = {
                    'fire': (0, 0, 255),      # 红色
                    'smoke': (0, 0, 255),   # 橙色
                    'unknown_person': (0, 0, 255),    # 蓝色
                    'unknown_vehicle': (0, 0, 255)  # 黄色
                }
                color = colors.get(anomaly['type'], (255, 255, 255))
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                if anomaly['type'] == 'unknown_person':
                    person_name = anomaly.get('person_name', 'Unknown')
                    distance = anomaly.get('recognition_distance', 0.0)
                    label = f"未知人员: {person_name} ({distance:.2f})"
                elif anomaly['type'] == 'unknown_vehicle':
                    plate_text = anomaly.get('plate_text', 'Unknown')
                    label = f"未知车辆: {plate_text}"
                elif anomaly['type'] == 'fire':
                    label = f"火焰: {anomaly['confidence']:.2f}"
                elif anomaly['type'] == 'smoke':
                    label = f"烟雾: {anomaly['confidence']:.2f}"
                else:
                    label = f"{anomaly['type']}: {anomaly['confidence']:.2f}"
                
                # 使用中文文本绘制函数（不使用背景色）
                frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=16, color=color)
        
        # 再绘制非异常的检测结果（已知人脸和已登记车牌）
        if all_detections:
            for detection in all_detections:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    box_key = f"{x1}_{y1}_{x2}_{y2}"
                    
                    # 避免重复绘制相同位置的框
                    if box_key in drawn_boxes:
                        continue
                    
                    if detection['type'] == 'known_person':
                        drawn_boxes.add(box_key)
                        
                        # 绿色边界框表示已知人员
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 显示人员姓名
                        name = detection.get('person_name', 'Known')
                        distance = detection.get('recognition_distance', 0.0)
                        label = f"已知人员: {name} ({distance:.2f})"
                        frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=16, color=(0, 255, 0))
                                   
                    elif detection['type'] == 'vehicle_plate':
                        plate_text = detection.get('plate_text', '')
                        is_registered = detection.get('is_registered', False)
                        ocr_processed = detection.get('ocr_processed', False)
                        
                        # 已登记的车牌 - 绿色显示
                        if ocr_processed and is_registered and plate_text not in ["size_error", "no_result", "low_confidence", "ocr_error"]:
                            drawn_boxes.add(box_key)
                            color = (0, 255, 0)  # 绿色 - 已登记
                            clean_text = self.normalize_plate(plate_text)
                            label = f"{clean_text} (已登记)"
                            
                            # 绘制车牌边界框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # 使用中文文本绘制函数显示车牌号和状态
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=16, color=color)
                        # 处理中的车牌 - 灰色显示
                        elif not ocr_processed:
                            drawn_boxes.add(box_key)
                            color = (128, 128, 128)  # 灰色 - 处理中
                            label = "检测中..."
                            
                            # 绘制车牌边界框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # 使用中文文本绘制函数显示车牌号和状态
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=16, color=color)
                        # OCR失败的情况 - 暗灰色显示
                        elif ocr_processed and plate_text in ["size_error", "no_result", "low_confidence", "ocr_error"]:
                            drawn_boxes.add(box_key)
                            color = (100, 100, 100)  # 暗灰色 - OCR失败
                            error_messages = {
                                "size_error": "图像太小",
                                "no_result": "无识别结果", 
                                "low_confidence": "置信度低",
                                "ocr_error": "识别失败"
                            }
                            label = error_messages.get(plate_text, "识别失败")
                            
                            # 绘制车牌边界框
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # 使用中文文本绘制函数显示状态
                            frame = self.draw_chinese_text(frame, label, (x1, y1-30), font_size=16, color=color)
        
        # 云台控制逻辑 - 优先处理异常
        directions = []
    
    # 决定使用哪个检测列表
        if anomalies:  # 优先处理异常
            detection_list = anomalies
            print("检测到异常，优先跟踪异常目标")
        elif all_detections:  # 没有异常时处理正常检测
            detection_list = all_detections
            print("无异常，跟踪正常检测目标")
        else:  # 都没有检测到
            detection_list = []
            print("无任何检测目标")
    
    # 收集检测框的方向
        for item in detection_list:
            bbox = item.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                direction = self.analyze_position(frame.shape, bbox_center_x, bbox_center_y)
                directions.append(direction)
    
    # 云台控制决策逻辑

        self.control_pan_tilt(directions)
        
        return frame

    
    def control_pan_tilt(self, directions):
        """统一的云台控制决策逻辑"""
        if not directions:
            print("无有效方向，云台不移动")
            return
    
    # 将方向列表转换为集合，便于检查
        directions_set = set(directions)

    # 检查基本方向
        hasTop = "上" in directions_set
        hasBottom = "下" in directions_set
        hasLeft = "左" in directions_set
        hasRight = "右" in directions_set

    # 检查复合方向
        hasLT = "左上" in directions_set
        hasLB = "左下" in directions_set
        hasRT = "右上" in directions_set
        hasRB = "右下" in directions_set
        hasMiddle = "中" in directions_set

    # 扩展检查：考虑复合方向中包含的相反方向
        hasLeftExtended = hasLeft or hasLT or hasLB
        hasRightExtended = hasRight or hasRT or hasRB
        hasTopExtended = hasTop or hasLT or hasRT
        hasBottomExtended = hasBottom or hasLB or hasRB

    # 如果有"中"，云台无需运动
        if hasMiddle:
            print("云台无需运动")
        else:
            # 检查是否存在相反方向（包括复合方向中的分量）
            if (hasLeftExtended and hasRightExtended) or (hasTopExtended and hasBottomExtended):
                print("存在相反方向，云台不移动")
            else:
                # 确定主要移动方向
                if hasTopExtended:
                    if hasLT and not hasRT:
                        print("控制云台往左上运动")
                        ptzMsg = String()
                        ptzMsg.data = "-0.5;0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    elif hasRT and not hasLT:
                        print("控制云台往右上运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.5;0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    else:
                        print("控制云台往上运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.0;0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                elif hasBottomExtended:
                    if hasLB and not hasRB:
                        print("控制云台往左下运动")
                        ptzMsg = String()
                        ptzMsg.data = "-0.5;-0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    elif hasRB and not hasLB:
                        print("控制云台往右下运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.5;-0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    else:
                        print("控制云台往下运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.0;-0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                elif hasLeftExtended:
                    if hasLT and not hasLB:
                        print("控制云台往左上运动")
                        ptzMsg = String()
                        ptzMsg.data = "-0.5;0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    elif hasLB and not hasLT:
                        print("控制云台往左下运动")
                        ptzMsg = String()
                        ptzMsg.data = "-0.5;-0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    else:
                        print("控制云台往左运动")
                        ptzMsg = String()
                        ptzMsg.data = "-0.5;0.0;0.5"
                        self.ptz_pub.publish(ptzMsg)
                elif hasRightExtended:
                    if hasRT and not hasRB:
                        print("控制云台往右上运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.5;0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    elif hasRB and not hasRT:
                        print("控制云台往右下运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.5;-0.5;0.5"
                        self.ptz_pub.publish(ptzMsg)
                    else:
                        print("控制云台往右运动")
                        ptzMsg = String()
                        ptzMsg.data = "0.5;0.0;0.5"
                        self.ptz_pub.publish(ptzMsg)
                elif hasLT:
                    print("控制云台往左上运动")
                    ptzMsg = String()
                    ptzMsg.data = "-0.5;0.5;0.5"
                    self.ptz_pub.publish(ptzMsg)
                elif hasLB:
                    print("控制云台往左下运动")
                    ptzMsg = String()
                    ptzMsg.data = "-0.5;-0.5;0.5"
                    self.ptz_pub.publish(ptzMsg)
                elif hasRT:
                    print("控制云台往右上运动")
                    ptzMsg = String()
                    ptzMsg.data = "0.5;0.5;0.5"
                    self.ptz_pub.publish(ptzMsg)
                elif hasRB:
                    print("控制云台往右下运动")
                    ptzMsg = String()
                    ptzMsg.data = "0.5;-0.5;0.5"
                    self.ptz_pub.publish(ptzMsg)
                else:
                    print("无有效方向，云台不移动")

    def analyze_position(self, frame_shape, bbox_center_x, bbox_center_y):
        # 获取画面中心
        if len(frame_shape) == 3:
            frame_height, frame_width, _ = frame_shape
        else:
            frame_height, frame_width = frame_shape
            
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
    # 计算中心区域阈值（画面的20%）
        threshold_x = frame_width * 0.2
        threshold_y = frame_height * 0.2
    
        direction = ""
    
    # 判断水平
        if bbox_center_x < frame_center_x - threshold_x:
            direction += "左"
        elif bbox_center_x > frame_center_x + threshold_x:
            direction += "右"
        
    # 判断垂直
        if bbox_center_y < frame_center_y - threshold_y:
            direction += "上"
        elif bbox_center_y > frame_center_y + threshold_y:
            direction += "下"
           
        if not direction:
            direction = "中"
        
        return direction

    def publish_image(self, frame):
        """发布图像数据"""
        try:
            resized_img = cv2.resize(frame,(1280,640))
            processed_img = self.bridge.cv2_to_imgmsg(resized_img,encoding='bgr8')
            self.processed_image_publisher.publish(processed_img)
            success, jpeg_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                image_as_str = base64.b64encode(jpeg_frame).decode('utf-8')
                img_msg = String()
                img_msg.data = image_as_str
                self.image_publisher.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"图像发布失败: {e}")

    def publish_anomalies(self, anomalies):
        """发布异常数据"""
        try:
            anomaly_list = AnomalyList()
            anomaly_list.header.stamp = self.get_clock().now().to_msg()
            anomaly_list.header.frame_id = "camera_frame"
            anomaly_list.total_count = len(anomalies)
            anomaly_list.frame_timestamp = time.time()
            
            for anomaly in anomalies:
                anomaly_msg = AnomalyDetection()
                anomaly_msg.header.stamp = self.get_clock().now().to_msg()
                anomaly_msg.header.frame_id = "camera_frame"
                anomaly_msg.anomaly_type = anomaly['type']
                anomaly_msg.timestamp = time.time()
                anomaly_msg.confidence = anomaly['confidence']
                
                # 设置位置信息
                if 'position' in anomaly:
                    anomaly_msg.position.x = float(anomaly['position'][0])
                    anomaly_msg.position.y = float(anomaly['position'][1])
                    anomaly_msg.position.z = 0.0
                
                anomaly_msg.image_path = anomaly.get('image_path', '')
                anomaly_msg.description = anomaly.get('description', '')
                
                anomaly_list.anomalies.append(anomaly_msg)
            
            self.anomaly_publisher.publish(anomaly_list)
            print(f"📡 发布异常数据: {len(anomalies)} 个异常")
            
        except Exception as e:
            self.get_logger().error(f"异常数据发布失败: {e}")

    def publish_alerts(self, anomalies):
        """发布警报信息"""
        try:
            import random
            current_time = time.time()
            
            for anomaly in anomalies:
                anomaly_type = anomaly['type']
                
                # 检查冷却时间控制
                should_alert = False
                
                if anomaly_type == 'fire':
                    if current_time - self.last_fire_alert_time >= self.fire_smoke_cooldown:
                        should_alert = True
                        self.last_fire_alert_time = current_time
                elif anomaly_type == 'smoke':
                    if current_time - self.last_smoke_alert_time >= self.fire_smoke_cooldown:
                        should_alert = True
                        self.last_smoke_alert_time = current_time
                elif anomaly_type == 'unknown_person':
                    if current_time - self.last_face_alert_time >= self.face_cooldown:
                        should_alert = True
                        self.last_face_alert_time = current_time
                elif anomaly_type == 'unknown_vehicle':
                    if current_time - self.last_plate_alert_time >= self.plate_cooldown:
                        should_alert = True
                        self.last_plate_alert_time = current_time
                else:
                    # 对于其他类型的异常，默认发布警报
                    should_alert = True
                
                # 如果在冷却时间内，跳过此次警报
                if not should_alert:
                    print(f"⏰ 冷却时间内，跳过 {anomaly_type} 警报")
                    continue
                
                # 生成自定义描述信息
                custom_descriptions = {
                    'fire': '已造成火情，建议立即前往处理',
                    'smoke': '烟雾可能造成火灾隐患，建议尽快前往查看',
                    'unknown_person': '非公司员工或已登记来访人员，建议尽快前往询问情况',
                    'unknown_vehicle': '非公司已登记车辆，建议尽快前往查看情况'
                }
                
                # 生成随机经纬度坐标（示例坐标，实际应用中应使用真实位置）
                latitude = round(random.uniform(39.9000, 39.9100), 6)  # 北京地区示例
                longitude = round(random.uniform(116.3000, 116.4000), 6)
                
                # 获取异常截图的base64编码
                screenshot_base64 = ""
                image_path = anomaly.get('image_path', '')
                if image_path and os.path.exists(image_path):
                    try:
                        with open(image_path, 'rb') as img_file:
                            screenshot_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                        print(f"📸 成功编码异常截图: {image_path}")
                    except Exception as e:
                        print(f"❌ 截图编码失败: {e}")
                        screenshot_base64 = ""
                else:
                    print(f"⚠️  未找到异常截图文件: {image_path}")
                
                alert_data = {
                    "type": anomaly_type,
                    "confidence": anomaly['confidence'],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 精确到秒
                    "description": custom_descriptions.get(anomaly_type, anomaly['description']),
                    "position": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "screenshot": screenshot_base64  # 添加异常截图的base64编码
                }
                
                # 发布警报
                alert_msg = String()
                alert_msg.data = json.dumps(alert_data, ensure_ascii=False)
                self.alert_publisher.publish(alert_msg)
                
                # 输出调试信息
                screenshot_info = f" (含截图: {len(screenshot_base64) > 0})" if screenshot_base64 else " (无截图)"
                print(f"🚨 发布警报: {anomaly_type} (置信度: {anomaly['confidence']:.3f}){screenshot_info}")
                
        except Exception as e:
            self.get_logger().error(f"警报发布失败: {e}")

    def cleanup(self):
        """清理资源"""
        # 停止OCR线程
        self.ocr_running = False
        for _ in range(len(self.ocr_threads)):
            try:
                self.ocr_queue.put(None, timeout=1)
            except:
                pass
        
        # 清理目标跟踪器
        if hasattr(self, 'target_tracker'):
            self.target_tracker.tracked_targets.clear()
            print("✅ 目标跟踪器已清理")
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 资源已清理，OCR线程已停止")

    def start_ocr_threads(self):
        """启动异步OCR处理线程"""
        num_threads = 2  # 使用2个线程处理OCR
        for i in range(num_threads):
            thread = threading.Thread(target=self.ocr_worker, daemon=True)
            thread.start()
            self.ocr_threads.append(thread)
        print(f"✅ 启动了 {num_threads} 个OCR异步处理线程")
    
    def ocr_worker(self):
        """OCR异步处理工作线程"""
        while self.ocr_running:
            try:
                # 从队列获取OCR任务
                task = self.ocr_queue.get(timeout=1)
                if task is None:
                    break
                
                crop_img, bbox_id, frame_idx = task
                
                # 执行OCR识别
                result = self.process_plate_ocr_async(crop_img)
                
                # 将结果放入结果队列
                self.ocr_result_queue.put((bbox_id, result, frame_idx))
                
                self.ocr_queue.task_done()
            except Exception as e:
                if "Empty" not in str(e):  # 忽略队列为空的异常
                    print(f"OCR线程处理异常: {e}")
                continue

    def process_plate_ocr_async(self, crop_img):
        """优化后的异步车牌OCR处理"""
        try:
            # 简化的预处理流程
            if crop_img.shape[0] < 20 or crop_img.shape[1] < 60:
                return "size_error", 0.0
            
            # 调整图像大小以提高OCR速度
            height, width = crop_img.shape[:2]
            if height > 64:  # 如果车牌太大，缩小以提高速度
                scale = 64 / height
                new_width = int(width * scale)
                crop_img = cv2.resize(crop_img, (new_width, 64))
            
            # 简化预处理：只进行对比度增强
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            enhanced_img = clahe.apply(gray_img)
            processed_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
            
            # OCR识别
            ocr_result = self.ocr.ocr(processed_img, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                return "no_result", 0.0
            
            # 选择最佳识别结果
            plate_number = ""
            best_confidence = 0.0
            
            for line in ocr_result:
                if not line:
                    continue
                for item in line:
                    if len(item) < 2 or not item[1]:
                        continue
                    text, conf = item[1]
                    if len(text) >= 4 and conf > best_confidence:
                        plate_number = text
                        best_confidence = conf
            
            if not plate_number or best_confidence < 0.3:
                return "low_confidence", best_confidence
                
            return plate_number, best_confidence
            
        except Exception as e:
            print(f"异步OCR处理失败: {e}")
            return "ocr_error", 0.0

    def get_cached_plate_result(self, bbox_key):
        """获取缓存的车牌识别结果（优化版）"""
        current_time = time.time()
        
        # 检查精确匹配的缓存
        if bbox_key in self.plate_cache:
            cache_entry = self.plate_cache[bbox_key]
            if current_time - cache_entry['timestamp'] < self.plate_cache_timeout:
                return cache_entry['result']
            else:
                # 清除过期缓存
                del self.plate_cache[bbox_key]
        
        # 模糊匹配附近区域的缓存
        x1, y1, x2, y2 = map(int, bbox_key.split('_'))
        tolerance = 15  # 像素容差
        
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
        """缓存车牌识别结果"""
        self.plate_cache[bbox_key] = {
            'result': result,
            'timestamp': time.time()
        }


def parse_args():
    parser = argparse.ArgumentParser(description="智能巡检机器人异常检测发布节点")
    #parser.add_argument('--source', type=str, default='6', help='相机源')
    parser.add_argument('--fire-model', type=str, default='/home/jetson/ros2_ws/src/example_python/models/fire_smoke_best.pt', help='火焰烟雾模型路径')
    parser.add_argument('--plate-model', type=str, default='/home/jetson/ros2_ws/src/example_python/models/plate_best.pt', help='车牌模型路径')
    parser.add_argument('--face-model', type=str, default='/home/jetson/ros2_ws/src/example_python/models/yolov8n-face.pt', help='人脸模型路径')
    parser.add_argument('--conf', type=float, default=0.7, help='通用置信度阈值（向后兼容）')
    parser.add_argument('--flame-conf', type=float, default=0.7, help='火焰检测置信度阈值')
    parser.add_argument('--smoke-conf', type=float, default=0.7, help='烟雾检测置信度阈值')
    parser.add_argument('--face-conf', type=float, default=0.7, help='人脸检测置信度阈值')
    parser.add_argument('--plate-conf', type=float, default=0.7, help='车牌检测置信度阈值')
    parser.add_argument('--target-timeout', type=float, default=20.0, help='目标消失超时时间（秒）')
    parser.add_argument('--no-display', action='store_true', help='禁用实时画面显示')
    # 忽略未知参数（ROS参数）
    args, unknown = parser.parse_known_args()
    return args


def main():
    rclpy.init(args=sys.argv)
    args = parse_args()
    
    publisher = None

    try:
        publisher = AnomalyDetectionPublisher(args)
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        print("\n🛑 接收到停止信号")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        if publisher is not None:
            publisher.cleanup()
            publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main() 
