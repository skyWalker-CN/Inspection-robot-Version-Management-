#!/usr/bin/env python3

import cv2
import numpy as np
import math
import argparse
import os
import time
from threading import Lock, Thread, Event

# ROS2 相关导入
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class IntegratedLaneNavNode(Node):
    def __init__(self, waypoints_list, roi_ratio=0.1, input_source="ros2", video_source=None,
                 center_tolerance=0.08, max_center_shift=50, front_angle=math.pi):
        super().__init__('integrated_lane_nav_node')

        # -------------------- 状态管理 --------------------
        self.state = "LANE_DETECTION"          # 当前状态: LANE_DETECTION / NAVIGATION
        self.current_waypoint_index = 0         # 当前路径点索引
        self.waypoints_list = waypoints_list    # 路径点列表: [[A1, B1], [A2, B2], ...]
        self.state_lock = Lock()
        self.navigation_in_progress = False      # 导航是否正在进行
        self.navigation_thread = None            # 导航线程
        self.navigation_complete_event = Event() # 导航完成事件
        self.navigation_success = False          # 导航是否成功

        # 车道模式冷却期相关
        self.lane_mode_block_until = None        # 车道模式禁止切换至导航的截止时间
        self.block_duration = 300.0              # 冷却时长 5分钟（单位：秒）

        # -------------------- 车道检测相关 --------------------
        self.bridge = CvBridge()
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.input_source = input_source
        self.video_source = video_source
        self.cap = None

        # 车道检测控制参数
        self.roi_ratio = roi_ratio
        self.center_tolerance = center_tolerance
        self.max_center_shift = max_center_shift
        self.linear_speed = 0.1
        self.angular_speed = 0.1
        self.min_distance_to_center = 100        # 车道线距离中心最小像素阈值
        self.min_slope = 0.3                      # 最小斜率
        self.max_slope = 1.0                      # 最大斜率
        self.mid_roi_start_ratio = 0.5
        self.mid_roi_end_ratio = 0.6

        # 车道检测状态变量
        self.bottom_lane_width = None
        self.mid_lane_width = None
        self.last_known_center_bottom = None
        self.last_known_center_mid = None
        self.frames_without_both_lanes_bottom = 0
        self.frames_without_both_lanes_mid = 0
        self.max_frames_without_lanes = 200
        self.center_queue_bottom = []
        self.center_queue_mid = []
        self.max_queue_len = 25
        self.fused_center_queue = []
        self.max_fused_queue_len = 50
        self.current_center_point_bottom = None
        self.current_center_point_mid = None
        self.frame_width = 640
        self.frame_center_x = 0
        self.current_frame = None
        self.debug_info_bottom = {}
        self.debug_info_mid = {}
        self.colors = {
            'bottom_left': (0, 0, 255),
            'bottom_right': (255, 0, 0),
            'mid_left': (0, 165, 255),
            'mid_right': (255, 0, 255),
            'bottom_center': (0, 255, 0),
            'mid_center': (255, 255, 0),
            'center_line': (255, 0, 255),
            'tolerance_line': (200, 200, 200),
            'bottom_roi': (255, 255, 0),
            'mid_roi': (255, 100, 100),
            'fused_center': (255, 255, 255),
        }

        # 融合中心点（供控制使用），使用锁保护
        self.fused_center_lock = Lock()
        self.fused_center_point = None
        self.fused_source = "none"

        # ---------- 新增：图像帧缓冲与处理定时器 ----------
        self.latest_frame = None
        self.frame_lock = Lock()
        # 图像处理定时器，频率10Hz（与控制定时器相同）
        self.process_timer = self.create_timer(0.1, self.process_latest_frame)

        # -------------------- 激光雷达相关 --------------------
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.obstacle_detected = False           # 障碍物检测标志
        self.obstacle_distance = float('inf')    # 障碍物距离
        self.scan_ranges = None
        self.obstacle_threshold = 4.0             # 障碍物检测阈值（米）
        self.obstacle_count = 0                    # 连续检测到障碍物的次数
        self.obstacle_threshold_count = 10         # 连续次数阈值
        self.front_angle = front_angle             # 正前方角度（弧度）

        # -------------------- 外部控制（controlMove） --------------------
        self.control_sub = self.create_subscription(
            String, 'controlMove', self.control_callback, 10)
        self.control_command = "move"              # 默认允许运动

        # -------------------- TF2 相关 --------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------------------- Nav2 导航相关 --------------------
        self.navigator = BasicNavigator()
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # -------------------- 设置输入源 --------------------
        self.setup_input_source()

        # -------------------- 创建定时器 --------------------
        self.control_timer = self.create_timer(0.1, self.control_timer_callback)      # 10Hz 控制
        self.state_machine_timer = self.create_timer(1.0, self.state_machine_callback) # 1Hz 状态机

        # -------------------- 创建窗口 --------------------
        cv2.namedWindow('Integrated Lane Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Mid ROI Edges', cv2.WINDOW_NORMAL)

        self.get_logger().info('整合节点已启动，初始状态: 车道检测')
        self.get_logger().info(f'正前方角度设为: {front_angle:.2f} rad')
        self.get_logger().info('已订阅 controlMove 话题，指令 "stop" 将强制停止机器人（仅在车道检测模式下）')
        self.get_logger().info('当前逻辑：检测到障碍物 → 切换到导航模式；导航完成后 → 切换回车到检测模式，并启动5分钟冷却期（期间忽略障碍物）')
        self.get_logger().info('优化：图像处理已移至独立定时器，控制命令将以10Hz稳定发布')
        self.get_logger().info('修复：在detect_lanes中添加近距离过滤，中间ROI检测已恢复正常')

    # -------------------- 激光雷达回调 --------------------
    def scan_callback(self, msg):
        """激光雷达回调函数，检测正前方障碍物"""
        if self.state != "LANE_DETECTION" and self.state != "NAVIGATION":
            # 在两个模式下都检测，以便及时切换
            pass

        self.scan_ranges = msg.ranges

        if self.scan_ranges:
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            angle_increment = msg.angle_increment

            # 使用指定的正前方角度
            front_index = int((self.front_angle - angle_min) / angle_increment)
            front_index = max(0, min(front_index, len(self.scan_ranges)-1))

            # 检测范围：正前方 ±15 度
            angle_range = 15 * np.pi / 180
            indices_range = int(angle_range / angle_increment)
            start_idx = max(0, front_index - indices_range)
            end_idx = min(len(self.scan_ranges)-1, front_index + indices_range)

            min_distance = float('inf')
            for i in range(start_idx, end_idx + 1):
                dist = self.scan_ranges[i]
                if not (math.isinf(dist) or math.isnan(dist) or dist <= 0.0):
                    if dist < min_distance:
                        min_distance = dist

            self.obstacle_distance = min_distance

            # 判断是否有障碍物（距离小于阈值且有效）
            obstacle_in_range = (min_distance < self.obstacle_threshold) and (min_distance != float('inf'))

            if obstacle_in_range:
                self.obstacle_count += 1
                if self.obstacle_count >= self.obstacle_threshold_count and not self.obstacle_detected:
                    self.obstacle_detected = True
                    self.get_logger().info(f'连续{self.obstacle_threshold_count}次检测到前方障碍物，距离: {min_distance:.2f}米')
            else:
                self.obstacle_count = 0
                if self.obstacle_detected:
                    self.obstacle_detected = False
                    self.get_logger().info('前方障碍物消失')

    # -------------------- controlMove 回调 --------------------
    def control_callback(self, msg):
        """接收外部控制指令"""
        self.control_command = msg.data
        self.get_logger().info(f'收到 controlMove 指令: {self.control_command}')

    # -------------------- 输入源设置 --------------------
    def setup_input_source(self):
        if self.input_source == "ros2":
            self.image_subscriber = self.create_subscription(
                Image, '/camera/color/image_raw', self.ros2_image_callback, 10)
        elif self.input_source == "camera":
            try:
                camera_id = int(self.video_source) if self.video_source else 0
                self.cap = cv2.VideoCapture(camera_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if self.cap.isOpened():
                    self.camera_timer = self.create_timer(0.033, self.camera_capture_callback)
            except Exception as e:
                self.get_logger().error(f'摄像头初始化错误: {str(e)}')
        elif self.input_source == "video":
            if self.video_source and os.path.exists(self.video_source):
                self.cap = cv2.VideoCapture(self.video_source)
                if self.cap.isOpened():
                    self.video_timer = self.create_timer(0.033, self.video_capture_callback)

    # -------------------- 图像回调函数（仅保存最新帧）--------------------
    def ros2_image_callback(self, msg):
        """ROS2图像回调：仅保存最新帧，不进行耗时处理"""
        if self.state != "LANE_DETECTION":  # 非车道模式不处理，节省资源
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'图像转换错误: {str(e)}')

    def camera_capture_callback(self):
        """摄像头采集回调：仅保存最新帧"""
        if self.state != "LANE_DETECTION":
            return
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.latest_frame = frame

    def video_capture_callback(self):
        """视频文件采集回调：仅保存最新帧"""
        if self.state != "LANE_DETECTION":
            return
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.latest_frame = frame
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放

    # -------------------- 定时处理最新帧 --------------------
    def process_latest_frame(self):
        """由定时器调用，处理最新保存的图像帧（10Hz）"""
        if self.state != "LANE_DETECTION":
            return
        with self.frame_lock:
            if self.latest_frame is None:
                return
            # 复制一份，避免处理过程中被覆盖
            frame = self.latest_frame.copy()
        # 调用原有的处理函数（包含检测和显示）
        self.process_and_display_frame(frame)

    # -------------------- 车道检测核心方法 --------------------
    def set_roi_regions(self, frame):
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_center_x = width // 2

        bottom_start_row = int(height * (1 - self.roi_ratio))
        bottom_end_row = height
        bottom_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        bottom_mask[bottom_start_row:bottom_end_row, :] = 255

        mid_start_row = int(height * self.mid_roi_start_ratio)
        mid_end_row = int(height * self.mid_roi_end_ratio)
        mid_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mid_mask[mid_start_row:mid_end_row, :] = 255

        return bottom_mask, bottom_start_row, bottom_end_row, mid_mask, mid_start_row, mid_end_row

    def preprocess_frame(self, frame, bottom_mask, mid_mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_all = cv2.Canny(blurred, 50, 150)
        bottom_edges = cv2.bitwise_and(edges_all, edges_all, mask=bottom_mask)
        mid_edges = cv2.bitwise_and(edges_all, edges_all, mask=mid_mask)
        return bottom_edges, mid_edges, gray

    def detect_lanes(self, edges, original_frame, start_row, end_row):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                                 minLineLength=25, maxLineGap=12)
        left_lanes = []
        right_lanes = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (y1 < start_row and y2 < start_row) or (y1 > end_row and y2 > end_row):
                    continue
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                slope_abs = abs(slope)
                if slope_abs < self.min_slope or slope_abs > self.max_slope:
                    continue
                if slope < 0 and x1 < original_frame.shape[1] * 0.7:
                    left_lanes.append(line[0])
                elif slope > 0 and x1 > original_frame.shape[1] * 0.3:
                    right_lanes.append(line[0])

        # ========== 关键修复：先过滤掉距离中心过近的车道线 ==========
        left_lanes = self.filter_lanes_by_distance(left_lanes, start_row, end_row, is_left=True)
        right_lanes = self.filter_lanes_by_distance(right_lanes, start_row, end_row, is_left=False)
        return left_lanes, right_lanes

    def filter_lanes_by_distance(self, lanes, roi_start_row, roi_end_row, is_left=True):
        filtered = []
        for lane in lanes:
            bottom_x, _ = self.get_lane_bottom_point(lane, roi_start_row, roi_end_row)
            if is_left and bottom_x > self.frame_center_x:
                continue
            if not is_left and bottom_x < self.frame_center_x:
                continue
            if abs(self.frame_center_x - bottom_x) < self.min_distance_to_center:
                continue
            filtered.append(lane)
        return filtered

    def select_closest_lane(self, lanes, is_left=True):
        if not lanes:
            return None
        closest = None
        min_dist = float('inf')
        for lane in lanes:
            x1, y1, x2, y2 = lane
            mid_x = (x1 + x2) / 2
            dist = abs(mid_x - self.frame_center_x)
            if dist < min_dist:
                min_dist = dist
                closest = lane
        return closest

    def get_lane_bottom_point(self, lane, roi_start_row, roi_end_row):
        if lane is None:
            return None, None
        x1, y1, x2, y2 = lane
        points = []
        if roi_start_row <= y1 <= roi_end_row:
            points.append((x1, y1))
        if roi_start_row <= y2 <= roi_end_row:
            points.append((x2, y2))
        if not points:
            if y1 > y2:
                return x1, y1
            else:
                return x2, y2
        else:
            return max(points, key=lambda p: p[1])

    def calculate_center_for_roi(self, left_lanes, right_lanes, roi_start_row, roi_end_row,
                                 frame_shape, debug_info_key="bottom"):
        """为指定ROI区域计算中心点 - 每个ROI单边只选择一条最靠近画面中心的线"""
        height, width = frame_shape[:2]

        # 关键修改：先筛选出最靠近中心的左右车道线
        left_lane = self.select_closest_lane(left_lanes, is_left=True)
        right_lane = self.select_closest_lane(right_lanes, is_left=False)

        # 更新调试信息，显示选择的车道线信息
        if debug_info_key == "bottom":
            debug_info = self.debug_info_bottom
            lane_width = self.bottom_lane_width
        else:
            debug_info = self.debug_info_mid
            lane_width = self.mid_lane_width

        # 重置调试信息
        debug_info = {
            'left_bottom_x': None,
            'right_bottom_x': None,
            'left_distance': None,
            'right_distance': None,
            'decision': 'unknown',
            'left_too_close': False,
            'right_too_close': False,
            'any_too_close': False,
            'left_lane_selected': left_lane is not None,
            'right_lane_selected': right_lane is not None,
            'left_lanes_count': len(left_lanes),
            'right_lanes_count': len(right_lanes),
        }

        if debug_info_key == "bottom":
            self.debug_info_bottom = debug_info
        else:
            self.debug_info_mid = debug_info

        # 检查车道线是否距离中心点过近
        left_too_close = False
        right_too_close = False

        if left_lane is not None:
            left_bottom_x, _ = self.get_lane_bottom_point(left_lane, roi_start_row, roi_end_row)
            left_distance_to_center = abs(self.frame_center_x - left_bottom_x)
            debug_info['left_bottom_x'] = left_bottom_x
            debug_info['left_distance'] = left_distance_to_center

            # 检查左车道线是否在中心线左侧
            if left_bottom_x > self.frame_center_x:
                # 左车道线在中心线右侧，位置错误
                debug_info['decision'] = 'left_on_wrong_side'
                left_too_close = True
            elif left_distance_to_center < self.min_distance_to_center:
                # 左车道线距离中心过近
                debug_info['left_too_close'] = True
                left_too_close = True

        if right_lane is not None:
            right_bottom_x, _ = self.get_lane_bottom_point(right_lane, roi_start_row, roi_end_row)
            right_distance_to_center = abs(right_bottom_x - self.frame_center_x)
            debug_info['right_bottom_x'] = right_bottom_x
            debug_info['right_distance'] = right_distance_to_center

            # 检查右车道线是否在中心线右侧
            if right_bottom_x < self.frame_center_x:
                # 右车道线在中心线左侧，位置错误
                debug_info['decision'] = 'right_on_wrong_side'
                right_too_close = True
            elif right_distance_to_center < self.min_distance_to_center:
                # 右车道线距离中心过近
                debug_info['right_too_close'] = True
                right_too_close = True

        # 检查是否有车道线距离中心点过近
        any_too_close = left_too_close or right_too_close
        debug_info['any_too_close'] = any_too_close

        # 核心规则：只要有任何车道线距离中心点过近，就按照没有检测到处理
        if any_too_close:
            if debug_info_key == "bottom":
                self.frames_without_both_lanes_bottom += 1
            else:
                self.frames_without_both_lanes_mid += 1

            # 确定具体的决策原因
            if left_too_close and right_too_close:
                debug_info['decision'] = 'both_too_close'
            elif left_too_close:
                debug_info['decision'] = 'left_too_close'
            elif right_too_close:
                debug_info['decision'] = 'right_too_close'

            # 不返回中心点
            return None, None, None

        # 检查左右车道线是否都存在
        if left_lane is not None and right_lane is not None:
            # 获取车道线底部端点
            left_bottom_x, left_bottom_y = self.get_lane_bottom_point(left_lane, roi_start_row, roi_end_row)
            right_bottom_x, right_bottom_y = self.get_lane_bottom_point(right_lane, roi_start_row, roi_end_row)

            # 检查两条车道线是否都在画面中心的同一侧
            if (left_bottom_x < self.frame_center_x and right_bottom_x < self.frame_center_x) or \
                    (left_bottom_x > self.frame_center_x and right_bottom_x > self.frame_center_x):
                # 两条车道线都在同一侧，视为无效检测
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'both_on_same_side'
                return None, left_lane, right_lane

            # 计算车道宽度
            if debug_info_key == "bottom":
                self.bottom_lane_width = abs(right_bottom_x - left_bottom_x)
                lane_width = self.bottom_lane_width
            else:
                self.mid_lane_width = abs(right_bottom_x - left_bottom_x)
                lane_width = self.mid_lane_width

            # 计算中心点
            center_x = (left_bottom_x + right_bottom_x) / 2
            center_y = (left_bottom_y + right_bottom_y) / 2

            if debug_info_key == "bottom":
                self.frames_without_both_lanes_bottom = 0
            else:
                self.frames_without_both_lanes_mid = 0
            debug_info['decision'] = 'valid_center'
            return (int(center_x), int(center_y)), left_lane, right_lane

        # 只有左车道线检测到的情况
        elif left_lane is not None:
            # 检查左车道线是否在中心线左侧（合理位置）
            left_bottom_x, _ = self.get_lane_bottom_point(left_lane, roi_start_row, roi_end_row)
            if left_bottom_x > self.frame_center_x:
                # 左车道线在中心线右侧，视为无效
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'left_on_wrong_side'
                return None, left_lane, None

            # 如果有已知的车道宽度，可以估计中心点
            if lane_width is not None:
                # 估计右侧车道线位置
                estimated_right_x = left_bottom_x + lane_width
                center_x = (left_bottom_x + estimated_right_x) / 2
                center_y = (roi_start_row + roi_end_row) / 2  # 使用ROI区域中间作为y坐标

                debug_info['decision'] = 'estimated_from_left'
                return (int(center_x), int(center_y)), left_lane, None
            else:
                # 没有已知的车道宽度，无法估计中心点
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'no_width_for_left'
                return None, left_lane, None

        # 只有右车道线检测到的情况
        elif right_lane is not None:
            # 检查右车道线是否在中心线右侧（合理位置）
            right_bottom_x, _ = self.get_lane_bottom_point(right_lane, roi_start_row, roi_end_row)
            if right_bottom_x < self.frame_center_x:
                # 右车道线在中心线左侧，视为无效
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'right_on_wrong_side'
                return None, None, right_lane

            # 如果有已知的车道宽度，可以估计中心点
            if lane_width is not None:
                # 估计左侧车道线位置
                estimated_left_x = right_bottom_x - lane_width
                center_x = (estimated_left_x + right_bottom_x) / 2
                center_y = (roi_start_row + roi_end_row) / 2  # 使用ROI区域中间作为y坐标

                debug_info['decision'] = 'estimated_from_right'
                return (int(center_x), int(center_y)), None, right_lane
            else:
                # 没有已知的车道宽度，无法估计中心点
                if debug_info_key == "bottom":
                    self.frames_without_both_lanes_bottom += 1
                else:
                    self.frames_without_both_lanes_mid += 1
                debug_info['decision'] = 'no_width_for_right'
                return None, None, right_lane

        else:
            # 两条车道线都未检测到
            if debug_info_key == "bottom":
                self.frames_without_both_lanes_bottom += 1
            else:
                self.frames_without_both_lanes_mid += 1
            debug_info['decision'] = 'no_lanes'
            return None, None, None

    def process_frame(self, frame):
        """核心处理函数：两个ROI区域独立处理，并加入中心点平滑（基于历史平均值）"""
        # 获取两个ROI区域
        bottom_mask, bottom_start_row, bottom_end_row, mid_mask, mid_start_row, mid_end_row = self.set_roi_regions(
            frame)

        # 预处理帧，获取边缘图像
        bottom_edges, mid_edges, gray = self.preprocess_frame(frame, bottom_mask, mid_mask)

        # 独立处理底部ROI区域（90%-100%）
        bottom_left_lanes, bottom_right_lanes = self.detect_lanes(bottom_edges, frame, bottom_start_row, bottom_end_row)
        center_point_bottom, bottom_left_lane, bottom_right_lane = self.calculate_center_for_roi(
            bottom_left_lanes, bottom_right_lanes, bottom_start_row, bottom_end_row,
            frame.shape, debug_info_key="bottom"
        )

        # 独立处理中间ROI区域（40%-60%）
        mid_left_lanes, mid_right_lanes = self.detect_lanes(mid_edges, frame, mid_start_row, mid_end_row)
        center_point_mid, mid_left_lane, mid_right_lane = self.calculate_center_for_roi(
            mid_left_lanes, mid_right_lanes, mid_start_row, mid_end_row,
            frame.shape, debug_info_key="mid"
        )

        # ----- 中心点平滑处理：基于历史平均值 -----
        # 底部 ROI 平滑
        bottom_smoothed = False
        if center_point_bottom is not None:
            # 计算当前队列的平均值（如果有）
            if len(self.center_queue_bottom) > 0:
                avg_x = sum(p[0] for p in self.center_queue_bottom) / len(self.center_queue_bottom)
                avg_y = sum(p[1] for p in self.center_queue_bottom) / len(self.center_queue_bottom)
                avg_point = (int(avg_x), int(avg_y))
            else:
                avg_point = None

            if avg_point is not None:
                diff_x = abs(center_point_bottom[0] - avg_point[0])
                if diff_x > self.max_center_shift:
                    # 变化过大，采用平均值
                    center_point_bottom = avg_point
                    bottom_smoothed = True
                    # 不将当前异常点加入队列，但更新 last_known_center 为平均值（可选）
                else:
                    # 变化正常，将当前点加入队列
                    self.center_queue_bottom.append(center_point_bottom)
                    if len(self.center_queue_bottom) > self.max_queue_len:
                        self.center_queue_bottom.pop(0)
            else:
                # 队列为空，直接使用当前点并加入队列
                self.center_queue_bottom.append(center_point_bottom)
            # 更新 last_known_center_bottom 为最终使用的中心点
            self.last_known_center_bottom = center_point_bottom
        else:
            # 没有检测到中心点，队列保持不变
            pass

        # 中间 ROI 平滑
        mid_smoothed = False
        if center_point_mid is not None:
            if len(self.center_queue_mid) > 0:
                avg_x = sum(p[0] for p in self.center_queue_mid) / len(self.center_queue_mid)
                avg_y = sum(p[1] for p in self.center_queue_mid) / len(self.center_queue_mid)
                avg_point = (int(avg_x), int(avg_y))
            else:
                avg_point = None

            if avg_point is not None:
                diff_x = abs(center_point_mid[0] - avg_point[0])
                if diff_x > self.max_center_shift:
                    center_point_mid = avg_point
                    mid_smoothed = True
                else:
                    self.center_queue_mid.append(center_point_mid)
                    if len(self.center_queue_mid) > self.max_queue_len:
                        self.center_queue_mid.pop(0)
            else:
                self.center_queue_mid.append(center_point_mid)
            self.last_known_center_mid = center_point_mid
        else:
            pass

        # ----- 计算融合中心点（两个ROI中心点的平均值或历史平均值）-----
        fused_center_point = None
        fused_source = "none"  # 用于显示来源：actual / historical

        if center_point_bottom is not None and center_point_mid is not None:
            # 两个都存在，计算融合中心点
            fused_x = (center_point_bottom[0] + center_point_mid[0]) // 2
            fused_y = (center_point_bottom[1] + center_point_mid[1]) // 2
            fused_center_point = (fused_x, fused_y)
            fused_source = "actual"
            # 加入队列
            self.fused_center_queue.append(fused_center_point)
            if len(self.fused_center_queue) > self.max_fused_queue_len:
                self.fused_center_queue.pop(0)

        elif center_point_bottom is not None or center_point_mid is not None:
            # 只有一个中心点存在，且队列不为空，则使用历史平均值
            if len(self.fused_center_queue) > 0:
                avg_x = sum(p[0] for p in self.fused_center_queue) / len(self.fused_center_queue)
                avg_y = sum(p[1] for p in self.fused_center_queue) / len(self.fused_center_queue)
                fused_center_point = (int(avg_x), int(avg_y))
                fused_source = "historical"
                # 不将推测点加入队列，保持队列为实际有效点
            # 队列为空，则保持 None

        else:
            # 两个中心点都消失，清空历史队列（因为后续可能重新出现有效点）
            #self.fused_center_queue = []
            fused_center_point = None
            fused_source = "none"

        # 保存融合中心点（线程安全）
        with self.fused_center_lock:
            self.fused_center_point = fused_center_point
            self.fused_source = fused_source

        # 绘制结果图像
        result = frame.copy()

        # 绘制两条容错参考线（左右各一条）
        self.draw_reference_lines(result)

        # 绘制两个ROI区域的边界
        # 底部ROI区域边界
        cv2.line(result, (0, bottom_start_row), (result.shape[1], bottom_start_row), self.colors['bottom_roi'], 2)
        cv2.line(result, (0, bottom_end_row), (result.shape[1], bottom_end_row), self.colors['bottom_roi'], 2)

        # 中间ROI区域边界
        cv2.line(result, (0, mid_start_row), (result.shape[1], mid_start_row), self.colors['mid_roi'], 2)
        cv2.line(result, (0, mid_end_row), (result.shape[1], mid_end_row), self.colors['mid_roi'], 2)

        # 只绘制选定的有效车道线（如果不过近且位置正确）
        # 底部ROI车道线
        if bottom_left_lane is not None and not self.debug_info_bottom.get('left_too_close', False):
            x1, y1, x2, y2 = bottom_left_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['bottom_left'], 3)  # 红色 - 底部ROI左车道线

        if bottom_right_lane is not None and not self.debug_info_bottom.get('right_too_close', False):
            x1, y1, x2, y2 = bottom_right_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['bottom_right'], 3)  # 蓝色 - 底部ROI右车道线

        # 中间ROI车道线 - 使用新的颜色
        if mid_left_lane is not None and not self.debug_info_mid.get('left_too_close', False):
            x1, y1, x2, y2 = mid_left_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['mid_left'], 3)  # 橙色 - 中间ROI左车道线

        if mid_right_lane is not None and not self.debug_info_mid.get('right_too_close', False):
            x1, y1, x2, y2 = mid_right_lane
            cv2.line(result, (x1, y1), (x2, y2), self.colors['mid_right'], 3)  # 紫色 - 中间ROI右车道线

        # 绘制底部ROI中心点
        if center_point_bottom is not None:
            cv2.circle(result, center_point_bottom, 15, self.colors['bottom_center'], -1)  # 绿色 - 底部ROI中心点

        # 绘制中间ROI中心点
        if center_point_mid is not None:
            cv2.circle(result, center_point_mid, 12, self.colors['mid_center'], -1)  # 青色 - 中间ROI中心点

        # 绘制融合中心点（白色）
        if fused_center_point is not None:
            cv2.circle(result, fused_center_point, 10, self.colors['fused_center'], -1)  # 白色 - 融合中心点

        # 显示两个ROI区域的车道线状态
        bottom_left_detected = bottom_left_lane is not None
        bottom_right_detected = bottom_right_lane is not None
        bottom_left_valid = bottom_left_detected and not self.debug_info_bottom.get('left_too_close', False)
        bottom_right_valid = bottom_right_detected and not self.debug_info_bottom.get('right_too_close', False)

        mid_left_detected = mid_left_lane is not None
        mid_right_detected = mid_right_lane is not None
        mid_left_valid = mid_left_detected and not self.debug_info_mid.get('left_too_close', False)
        mid_right_valid = mid_right_detected and not self.debug_info_mid.get('right_too_close', False)

        bottom_lanes_status = f'Bottom ROI: L={bottom_left_valid}, R={bottom_right_valid}'
        mid_lanes_status = f'Mid ROI: L={mid_left_valid}, R={mid_right_valid}'

        cv2.putText(result, bottom_lanes_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, mid_lanes_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 如果两个ROI都检测到了中心点，计算并显示差异
        if center_point_bottom is not None and center_point_mid is not None:
            diff_x = center_point_mid[0] - center_point_bottom[0]
            diff_text = f'Center Diff: {diff_x}px'
            color = (0, 0, 255) if abs(diff_x) > 20 else (0, 255, 0)  # 差异超过20像素显示红色
            # cv2.putText(result, diff_text, (result.shape[1] - 200, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result, bottom_edges, mid_edges, center_point_bottom, center_point_mid

    def draw_reference_lines(self, result):
        """绘制参考线"""
        cv2.line(result, (self.frame_center_x, 0), (self.frame_center_x, result.shape[0]),
                 self.colors['center_line'], 2)

        tolerance_pixels = int(self.frame_width * self.center_tolerance / 2)
        left_tolerance_x = self.frame_center_x - tolerance_pixels
        right_tolerance_x = self.frame_center_x + tolerance_pixels

        cv2.line(result, (left_tolerance_x, 0), (left_tolerance_x, result.shape[0]),
                 self.colors['tolerance_line'], 1, cv2.LINE_AA)
        cv2.line(result, (right_tolerance_x, 0), (right_tolerance_x, result.shape[0]),
                 self.colors['tolerance_line'], 1, cv2.LINE_AA)

        overlay = result.copy()
        cv2.rectangle(overlay, (left_tolerance_x, 0), (right_tolerance_x, result.shape[0]), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, result, 0.9, 0, result)

    def process_and_display_frame(self, frame):
        """处理并显示帧，同时保存融合中心点"""
        result, edges, mid_edges, center_bottom, center_mid = self.process_frame(frame)
        # fused_center_point 已经通过 self.fused_center_point 保存，无需额外操作
        cv2.imshow('Integrated Lane Detection', result)
        cv2.imshow('Edges', edges)
        cv2.imshow('Mid ROI Edges', mid_edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.cleanup_and_shutdown()
        elif key == ord('s'):
            if frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"lane_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.get_logger().info(f'已保存帧到: {filename}')
        elif key == ord('+'):
            self.center_tolerance = min(self.center_tolerance + 0.02, 0.5)
            self.get_logger().info(f'增加容错距离: {self.center_tolerance:.2f}')
        elif key == ord('-'):
            self.center_tolerance = max(self.center_tolerance - 0.02, 0.01)
            self.get_logger().info(f'减少容错距离: {self.center_tolerance:.2f}')
        elif key == ord('d'):
            self.min_distance_to_center = min(self.min_distance_to_center + 5, 200)
            self.get_logger().info(f'增加距离阈值: {self.min_distance_to_center}像素')
        elif key == ord('a'):
            self.min_distance_to_center = max(self.min_distance_to_center - 5, 10)
            self.get_logger().info(f'减少距离阈值: {self.min_distance_to_center}像素')

    # -------------------- 控制发布 --------------------
    def control_timer_callback(self):
        """控制定时器：根据状态发布速度命令"""
        # 获取当前状态（使用锁保证线程安全）
        with self.state_lock:
            current_state = self.state

        # 导航模式下，不发布任何速度命令，完全由 Nav2 控制
        if current_state == "NAVIGATION":
            return

        # 以下为车道检测模式的处理
        if self.control_command == "stop":
            # 强制停车
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist)
            return

        if self.control_command == "move":
            # 正常车道控制
            self.publish_lane_control()
        else:
            # 未知指令，安全起见停车
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_publisher.publish(twist)
            self.get_logger().warn(f'未知 control command: {self.control_command}, 停止')

    def publish_lane_control(self):
        """根据融合中心点发布车道保持控制命令"""
        with self.fused_center_lock:
            center = self.fused_center_point
            width = self.frame_width
            center_x = self.frame_center_x

        twist = Twist()
        if center is not None:
            offset = (center[0] - center_x) / (width / 2)
            tolerance = self.center_tolerance
            if abs(offset) < tolerance:
                twist.linear.x = self.linear_speed
                twist.angular.z = 0.0
            elif offset < -tolerance:
                twist.linear.x = self.linear_speed
                twist.angular.z = self.angular_speed
            else:
                twist.linear.x = self.linear_speed
                twist.angular.z = -self.angular_speed
        else:
            # 无融合中心点，减速或停止
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    # -------------------- 状态机 --------------------
    def state_machine_callback(self):
        with self.state_lock:
            now = self.get_clock().now()

            if self.state == "LANE_DETECTION":
                can_switch = False
                if self.obstacle_detected:
                    # 检查冷却期
                    if self.lane_mode_block_until is not None and now < self.lane_mode_block_until:
                        remaining = (self.lane_mode_block_until - now).nanoseconds / 1e9
                        self.get_logger().info(f'车道模式冷却期内，剩余 {remaining:.1f} 秒，忽略障碍物检测')
                    else:
                        can_switch = True

                if can_switch and self.current_waypoint_index < len(self.waypoints_list) and not self.navigation_in_progress:
                    self.get_logger().info('检测到前方障碍物，切换到导航模式')
                    # 先停止车道控制
                    stop = Twist()
                    stop.linear.x = 0.0
                    stop.angular.z = 0.0
                    self.cmd_vel_publisher.publish(stop)
                    time.sleep(0.1)
                    self.state = "NAVIGATION"
                    self.start_navigation()

            elif self.state == "NAVIGATION":
                if self.navigation_complete_event.is_set():
                    if self.navigation_success:
                        self.get_logger().info('导航成功完成，切换回车到检测模式，并启动冷却期')
                        # 设置冷却期，之后忽略障碍物检测
                        self.lane_mode_block_until = now + Duration(seconds=self.block_duration)
                        # 递增路径点索引
                        self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints_list)
                    else:
                        self.get_logger().warn('导航失败，切换回车到检测模式，保留当前路径点')
                    self.navigation_complete_event.clear()
                    self.state = "LANE_DETECTION"

    # -------------------- 导航相关 --------------------
    def start_navigation(self):
        """启动导航到当前路径点对的目标点"""
        if self.current_waypoint_index >= len(self.waypoints_list):
            self.get_logger().warn('所有路径点已完成，无法启动导航')
            return
        if self.navigation_in_progress:
            self.get_logger().warn('导航正在进行中，忽略新请求')
            return

        start_pose, goal_pose = self.waypoints_list[self.current_waypoint_index]
        self.get_logger().info(f'开始导航: 路径点 {self.current_waypoint_index+1}/{len(self.waypoints_list)}')

        self.navigation_in_progress = True
        self.navigation_thread = Thread(target=self._navigation_thread, args=(start_pose, goal_pose), daemon=True)
        self.navigation_thread.start()

    def _navigation_thread(self, start_pose, goal_pose):
        """导航线程"""
        try:
            self.get_logger().info('等待 Nav2 激活...')
            self.navigator.waitUntilNav2Active()
            self.get_logger().info('Nav2 已激活')

            # 设置初始位置
            self._publish_initial_pose(start_pose)
            time.sleep(2)

            self.get_logger().info(f'开始导航到目标...')
            self.navigator.goToPose(goal_pose)

            # 等待导航完成
            while not self.navigator.isTaskComplete():
                time.sleep(0.5)

            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info('导航成功完成!')
                self.navigation_success = True
            else:
                self.get_logger().error(f'导航失败，结果: {result}')
                self.navigation_success = False

        except Exception as e:
            self.get_logger().error(f'导航线程异常: {e}')
            self.navigation_success = False
        finally:
            self.navigation_in_progress = False
            self.navigation_complete_event.set()

    def _publish_initial_pose(self, pose):
        """发布初始位置到 /initialpose"""
        try:
            initial_pose_msg = PoseWithCovarianceStamped()
            initial_pose_msg.header = pose.header
            initial_pose_msg.header.frame_id = 'map'
            initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
            initial_pose_msg.pose.pose = pose.pose
            initial_pose_msg.pose.covariance = [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942
            ]
            for i in range(3):
                self.initial_pose_pub.publish(initial_pose_msg)
                time.sleep(0.5)
        except Exception as e:
            self.get_logger().error(f'发布初始位置失败: {e}')

    # -------------------- 清理退出 --------------------
    def cleanup_and_shutdown(self):
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop)

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        exit(0)


# -------------------- 辅助函数 --------------------
def normalize_quaternion(qx, qy, qz, qw):
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return 0.0, 0.0, 0.0, 1.0
    return qx/norm, qy/norm, qz/norm, qw/norm

def create_pose_stamped(node, x, y, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0, frame_id='map'):
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def main():
    parser = argparse.ArgumentParser(description='整合车道检测和导航系统（激光雷达触发切换：检测到障碍物时切换至导航，导航成功后冷却5分钟）')
    parser.add_argument('--input', type=str, default='video',
                        choices=['ros2', 'camera', 'video'],
                        help='输入源类型')
    parser.add_argument('--source', type=str, default='/home/jetson/ros2_ws/src/example_python/example_python/6.mp4',
                        help='源路径: 摄像头ID或视频文件路径')
    parser.add_argument('--roi', type=float, default=0.1,
                        help='底部ROI比例')
    parser.add_argument('--tolerance', type=float, default=0.08,
                        help='单边容错比例')
    parser.add_argument('--max-shift', type=int, default=25,
                        help='中心点平滑最大位移')
    parser.add_argument('--obstacle-threshold', type=float, default=4.0,
                        help='障碍物检测阈值（米）')
    parser.add_argument('--obstacle-count', type=int, default=10,
                        help='连续检测次数阈值')
    parser.add_argument('--front-angle', type=float, default=math.pi,
                        help='正前方角度（弧度），默认为π（雷达0度是后方）')
    parser.add_argument('--block-duration', type=float, default=300.0,
                        help='导航成功后车道模式禁止切换的冷却时长（秒），默认300秒（5分钟）')

    args = parser.parse_args()

    rclpy.init()

    # 创建临时节点生成路径点（示例，请根据实际地图修改）
    temp_node = rclpy.create_node('temp_pose_creator')
    waypoints_list = [
        [
            create_pose_stamped(temp_node, 2.3, -5.1, 0.0, 0.0, 0.0, 0.69, 0.71),  # A1
            create_pose_stamped(temp_node, -0.48, -4.49, 0.0, 0.0, 0.0, 0.0, 1.0)   # B1
        ],
        [
            create_pose_stamped(temp_node, 5.9, -0.6, 0.0, 0.0, 0.0, -0.03, 0.99),  # A2
            create_pose_stamped(temp_node, 4.18, 0.98, 0.0, 0.0, 0.0, 0.99, 0.01)   # B2
        ]
    ]
    temp_node.destroy_node()

    node = IntegratedLaneNavNode(
        waypoints_list=waypoints_list,
        roi_ratio=args.roi,
        input_source=args.input,
        video_source=args.source,
        center_tolerance=args.tolerance,
        max_center_shift=args.max_shift,
        front_angle=args.front_angle
    )

    # 设置障碍物参数
    node.obstacle_threshold = args.obstacle_threshold
    node.obstacle_threshold_count = args.obstacle_count
    node.block_duration = args.block_duration

    print("整合车道检测和导航系统（激光雷达触发切换：检测到障碍物时切换至导航，导航成功后冷却5分钟）")
    print("=" * 50)
    print(f"初始状态: 车道检测模式")
    print(f"正前方角度: {args.front_angle:.2f} rad")
    print(f"障碍物阈值: {args.obstacle_threshold} 米, 连续次数: {args.obstacle_count}")
    print(f"导航成功后冷却时长: {args.block_duration} 秒")
    print("按 'q' 退出程序")
    print("注意: 检测到障碍物 -> 导航模式；导航完成后 -> 车道检测模式；导航成功后5分钟内忽略障碍物检测")
    print("优化：图像处理已移至独立定时器，控制命令将以10Hz稳定发布")
    print("修复：在detect_lanes中添加近距离过滤，中间ROI检测已恢复正常")

    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        node.cleanup_and_shutdown()


if __name__ == "__main__":
    main()
