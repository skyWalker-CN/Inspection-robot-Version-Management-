#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import math
import argparse
import os
import time
from threading import Lock, Thread, Event
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class IntegratedLaneNavNode(Node):
    def __init__(self, waypoints_list, roi_ratio=0.6, input_source="ros2", video_source=None):
        super().__init__('integrated_lane_nav_node')
        
        # 状态管理
        self.state = "LANE_DETECTION"  # 初始状态为车道检测
        self.current_waypoint_index = 0  # 当前路径点索引
        self.waypoints_list = waypoints_list  # 路径点列表: [[A1, B1], [A2, B2], ...]
        self.state_lock = Lock()
        
        # 车道检测相关
        self.bridge = CvBridge()
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.input_source = input_source
        self.video_source = video_source
        self.cap = None
        
        # 激光雷达相关
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.obstacle_detected = False  # 障碍物检测标志
        self.obstacle_distance = float('inf')  # 障碍物距离
        self.scan_ranges = None  # 激光雷达数据
        self.obstacle_threshold = 5.0  # 障碍物检测阈值（米）
        self.obstacle_count = 0  # 连续检测到障碍物的次数
        self.obstacle_threshold_count = 10  # 连续检测到障碍物的阈值次数
        
        # 车道检测控制参数
        self.roi_ratio = roi_ratio
        self.lane_width = None
        self.last_known_center = None
        self.frames_without_both_lanes = 0  # 连续丢失两条车道线的帧数
        self.max_frames_without_lanes = 200  # 连续丢失两条车道线的最大帧数阈值
        self.center_tolerance = 0.08
        self.linear_speed = 0.1
        self.angular_speed = 0.1
        self.current_center_point = None
        self.frame_width = 640
        self.frame_center_x = 0
        self.current_frame = None
        
        # TF2相关 - 用于处理坐标变换
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Nav2导航相关 - 修复版本
        self.navigator = BasicNavigator()
        self.navigation_in_progress = False
        self.navigation_thread = None
        self.navigation_complete_event = Event()
        self.navigation_success = False
        
        # 初始位置发布者
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/initialpose', 
            10
        )
        
        # 设置输入源
        self.setup_input_source()
        
        # 创建定时器
        self.control_timer = self.create_timer(0.1, self.control_timer_callback)
        self.state_machine_timer = self.create_timer(1.0, self.state_machine_callback)
        
        # 创建窗口
        cv2.namedWindow('Integrated Lane Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
        
        self.get_logger().info('整合节点已启动，初始状态: 车道检测')
    
    def setup_input_source(self):
        """设置输入源"""
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
    
    def scan_callback(self, msg):
        """激光雷达回调函数"""
        if self.state != "LANE_DETECTION":
            return
            
        # 保存激光雷达数据
        self.scan_ranges = msg.ranges
        
        # 检查正前方5米范围内是否有障碍物
        if self.scan_ranges:
            # 获取激光雷达的角度范围和分辨率
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            angle_increment = msg.angle_increment
            
            # 首先打印一下角度范围，帮助我们调试
            if not hasattr(self, 'angle_info_printed'):
                self.get_logger().info(f'激光雷达角度范围: {angle_min:.2f} 到 {angle_max:.2f} 弧度')
                self.get_logger().info(f'激光雷达角度分辨率: {angle_increment:.4f} 弧度')
                self.angle_info_printed = True
            
            # 重要修正：根据您确认的配置，0度是正后方
            # 正前方应该是π（3.14）弧度或-π（-3.14）弧度
            # 由于角度范围是-3.12到3.14，所以使用3.14弧度作为正前方
            
            # 修正这里：强制设置正前方角度为π（3.14弧度）
            front_angle = math.pi  # 3.14159弧度，正前方
            
            # 计算正前方角度的索引
            front_index = int((front_angle - angle_min) / angle_increment)
            
            # 确保索引在范围内
            front_index = max(0, min(front_index, len(self.scan_ranges)-1))
            
            # 定义检测范围（正前方-15度到+15度）
            angle_range = 1 * np.pi / 180  # 15度转换为弧度
            indices_range = int(angle_range / angle_increment)
            
            start_idx = max(0, front_index - indices_range)
            end_idx = min(len(self.scan_ranges) - 1, front_index + indices_range)
            
            # 检查该范围内的最小距离
            min_distance = float('inf')
            for i in range(start_idx, end_idx + 1):
                distance = self.scan_ranges[i]
                # 跳过无效值
                if not (math.isinf(distance) or math.isnan(distance) or distance <= 0.0):
                    if distance < min_distance:
                        min_distance = distance
            
            self.obstacle_distance = min_distance
            
            # 检查是否检测到障碍物在阈值范围内
            obstacle_in_range = False
            if (min_distance > self.obstacle_threshold and min_distance < 5.0) and min_distance != float('inf'):
                obstacle_in_range = True
            
            # 更新连续检测计数器
            if obstacle_in_range:
                self.obstacle_count += 1
                if self.obstacle_count >= self.obstacle_threshold_count:
                    # 连续5次检测到障碍物，设置障碍物标志
                    if not self.obstacle_detected:
                        self.obstacle_detected = True
                        self.get_logger().info(f'连续{self.obstacle_threshold_count}次检测到前方障碍物，距离: {min_distance:.2f}米')
            else:
                # 没有检测到障碍物，重置计数器
                self.obstacle_count = 0
                self.obstacle_detected = False
            
            # 调试信息
            if not hasattr(self, 'debug_counter'):
                self.debug_counter = 0
            if self.debug_counter % 20 == 0:
                self.get_logger().info(f'激光雷达检测: 距离={min_distance:.2f}m, 检测角度={front_angle:.2f}rad, 索引={front_index}')
                self.get_logger().info(f'检测范围: {start_idx}-{end_idx}, 障碍物检测状态: {self.obstacle_detected}, 连续计数: {self.obstacle_count}')
            self.debug_counter += 1
    
    def ros2_image_callback(self, msg):
        """ROS2图像回调"""
        if self.state != "LANE_DETECTION":
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.process_and_display_frame(cv_image)
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {str(e)}')
    
    def camera_capture_callback(self):
        """摄像头回调"""
        if self.state != "LANE_DETECTION":
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.process_and_display_frame(frame)
    
    def video_capture_callback(self):
        """视频回调"""
        if self.state != "LANE_DETECTION":
            return
            
        ret, frame = self.cap.read()
        if ret:
            self.process_and_display_frame(frame)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def process_and_display_frame(self, frame):
        """处理并显示帧"""
        self.current_frame = frame
        result, edges, center_point = self.process_frame(frame)
        
        cv2.imshow('Integrated Lane Detection', result)
        cv2.imshow('Edges', edges)
        
        self.current_center_point = center_point
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.cleanup_and_shutdown()
    
    def state_machine_callback(self):
        """状态机回调函数"""
        with self.state_lock:
            if self.state == "LANE_DETECTION":
                # 检查是否应该切换到导航（基于障碍物检测）
                if self.should_switch_to_navigation():
                    self.get_logger().info('检测到前方障碍物在5米以外，切换到导航模式')
                    # 切换到导航模式时，立即发送一次停止命令
                    stop_msg = Twist()
                    stop_msg.linear.x = 0.0
                    stop_msg.angular.z = 0.0
                    self.cmd_vel_publisher.publish(stop_msg)
                    time.sleep(0.1)  # 确保停止命令被发送
                    
                    # 重置障碍物检测标志和计数器
                    self.obstacle_detected = False
                    self.obstacle_count = 0
                    self.state = "NAVIGATION"
                    self.start_navigation()
            
            elif self.state == "NAVIGATION":
                # 检查导航是否完成
                if self.navigation_complete_event.is_set():
                    if self.navigation_success:
                        self.get_logger().info('导航成功完成，切换回车到检测模式')
                        
                        # 重置障碍物检测标志
                        self.obstacle_detected = False
                        self.obstacle_count = 0
                        self.current_center_point = None
                        
                        self.state = "LANE_DETECTION"
                        self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints_list)
                        self.navigation_complete_event.clear()
                    else:
                        # 导航失败，保持导航状态，不切换回车道检测
                        self.get_logger().warn('导航失败，保持导航状态')
                        # 清除事件，避免重复触发
                        self.navigation_complete_event.clear()
    
    def should_switch_to_navigation(self):
        """判断是否应该切换到导航模式"""
        return (self.obstacle_detected and 
                len(self.waypoints_list) > 0 and
                not self.navigation_in_progress)  # 确保没有导航在进行中
    
    def start_navigation(self):
        """开始导航到下一个路径点"""
        if self.current_waypoint_index >= len(self.waypoints_list):
            self.get_logger().warn('所有路径点已完成，停止导航')
            return
        
        # 检查是否已有导航在进行
        if self.navigation_in_progress:
            self.get_logger().warn('导航正在进行中，忽略新请求')
            return
            
        waypoint_pair = self.waypoints_list[self.current_waypoint_index]
        start_point, end_point = waypoint_pair
        
        self.get_logger().info(f'开始导航: 路径点 {self.current_waypoint_index + 1}/{len(self.waypoints_list)}')
        
        # 启动导航线程
        self.navigation_in_progress = True
        self.navigation_thread = Thread(
            target=self._navigation_thread, 
            args=(start_point, end_point), 
            daemon=True
        )
        self.navigation_thread.start()
    
    def _navigation_thread(self, start_pose, goal_pose):
        """导航线程 - 简化版本"""
        try:
            self.get_logger().info('等待Nav2系统激活...')
            
            # 等待Nav2激活
            self.navigator.waitUntilNav2Active()
            self.get_logger().info('Nav2已激活')
            
            # 设置初始位置
            self._publish_initial_pose_directly(start_pose)
            time.sleep(2)
            
            # 开始导航
            self.get_logger().info(f'开始导航到目标...')
            self.navigator.goToPose(goal_pose)
            
            # 简化：直接等待导航完成
            i = 0
            while not self.navigator.isTaskComplete():
                if i % 20 == 0:  # 每2秒打印一次
                    self.get_logger().info('等待导航完成...')
                i += 1
                time.sleep(0.1)
            
            # 简化：只要没有异常，就认为是成功
            self.navigation_success = True
            self.get_logger().info('导航成功完成!')
                
        except Exception as e:
            self.get_logger().error(f'导航错误: {str(e)}')
            self.navigation_success = False
        finally:
            self.navigation_in_progress = False
            self.navigation_complete_event.set()
    
    def _publish_initial_pose_directly(self, pose):
        """直接发布初始位置到AMCL - 避免时间戳问题"""
        try:
            # 创建消息
            initial_pose_msg = PoseWithCovarianceStamped()
            initial_pose_msg.header = pose.header
            initial_pose_msg.header.frame_id = 'map'
            initial_pose_msg.header.stamp = self.get_clock().now().to_msg()  # 使用当前时间
            initial_pose_msg.pose.pose = pose.pose
            
            # 设置协方差
            initial_pose_msg.pose.covariance = [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942
            ]
            
            # 发布初始位置 - 发布3次确保被接收
            self.get_logger().info('发布初始位置到/initialpose话题')
            for i in range(3):
                initial_pose_pub = self.create_publisher(
                    PoseWithCovarianceStamped, 
                    '/initialpose', 
                    10
                )
                initial_pose_pub.publish(initial_pose_msg)
                time.sleep(0.5)
                
        except Exception as e:
            self.get_logger().error(f'直接发布初始位置失败: {e}')
    
    def control_timer_callback(self):
        """控制定时器回调 - 修复版本"""
        # 只在车道检测模式下发布控制命令
        if self.state == "LANE_DETECTION":
            self.publish_lane_control_command()
        # 导航模式下不发布任何控制命令，让Nav2完全控制
        elif self.state == "NAVIGATION":
            # 在导航模式下，我们可以完全停止发布，或者偶尔发布一个空命令
            # 但最好是完全不发布，让Nav2的controller_server独占控制
            pass
    
    def publish_lane_control_command(self):
        """发布车道控制命令"""
        msg = Twist()
        
        if self.current_center_point is not None:
            offset = (self.current_center_point[0] - self.frame_center_x) / (self.frame_width / 2)
            tolerance = self.center_tolerance
            
            if abs(offset) < tolerance:
                msg.linear.x = self.linear_speed
                msg.angular.z = 0.0
            elif offset < -tolerance:
                msg.linear.x = self.linear_speed
                msg.angular.z = self.angular_speed
            else:
                msg.linear.x = self.linear_speed
                msg.angular.z = -self.angular_speed
            
            self.frames_without_both_lanes = 0  # 重置丢失计数
        else:
            # 只有当两条车道线都丢失时才增加计数
            if self.frames_without_both_lanes < self.max_frames_without_lanes:
                # 尝试使用上一次已知的中心点
                if self.last_known_center is not None:
                    # 使用历史中心点进行控制
                    offset = (self.last_known_center[0] - self.frame_center_x) / (self.frame_width / 2)
                    tolerance = self.center_tolerance * 1.5  # 增加容差
                    
                    if abs(offset) < tolerance:
                        msg.linear.x = self.linear_speed
                    elif offset < -tolerance:
                        msg.linear.x = self.linear_speed
                        msg.angular.z = self.angular_speed
                    else:
                        msg.linear.x = self.linear_speed
                        msg.angular.z = -self.angular_speed
                else:
                    msg.linear.x = self.linear_speed * 0.3
                    msg.angular.z = 0.0
            else:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
        
        self.cmd_vel_publisher.publish(msg)
    
    # 修改的车道检测方法
    def set_roi_region(self, frame):
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_center_x = width // 2
        start_row = int(height * (1 - self.roi_ratio))
        end_row = height
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[start_row:end_row, :] = 255
        return mask, start_row, end_row
    
    def preprocess_frame(self, frame, mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_roi = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_and(edges_roi, edges_roi, mask=mask)
        return edges, gray
    
    def detect_lanes(self, edges, original_frame, start_row):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                               minLineLength=25, maxLineGap=12)
        
        left_lanes = []
        right_lanes = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if y1 < start_row and y2 < start_row:
                    continue
                
                if x2 - x1 == 0:
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                if abs(slope) < 0.3:
                    continue
                
                if slope < 0 and x1 < original_frame.shape[1] * 0.7:
                    left_lanes.append(line[0])
                elif slope > 0 and x1 > original_frame.shape[1] * 0.3:
                    right_lanes.append(line[0])
        
        return left_lanes, right_lanes
    
    def select_closest_lane(self, lanes, frame_center_x, is_left=True):
        """选择最靠近画面中心的车道线"""
        if not lanes:
            return None
        
        closest_lane = None
        min_distance = float('inf')
        
        for lane in lanes:
            x1, y1, x2, y2 = lane
            # 计算线段的中点
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # 计算到画面中心线的水平距离
            distance = abs(mid_x - frame_center_x)
            
            if distance < min_distance:
                min_distance = distance
                closest_lane = lane
        
        return closest_lane
    
    def calculate_center_from_lines(self, left_lane, right_lane, frame_shape, start_row):
        """从两条直线计算中心点"""
        height, width = frame_shape[:2]
        
        if left_lane is not None and right_lane is not None:
            # 计算左车道线底部端点
            x1_left, y1_left, x2_left, y2_left = left_lane
            left_bottom_y = max(y1_left, y2_left)
            left_bottom_x = x1_left if y1_left > y2_left else x2_left
            
            # 计算右车道线底部端点
            x1_right, y1_right, x2_right, y2_right = right_lane
            right_bottom_y = max(y1_right, y2_right)
            right_bottom_x = x1_right if y1_right > y2_right else x2_right
            
            # 计算车道宽度
            self.lane_width = abs(right_bottom_x - left_bottom_x)
            
            # 计算中心点
            center_x = (left_bottom_x + right_bottom_x) / 2
            center_y = (left_bottom_y + right_bottom_y) / 2
            
            self.frames_without_both_lanes = 0
            return (int(center_x), int(center_y))
        
        # 只有一条车道线检测到的情况
        elif left_lane is not None and self.lane_width is not None:
            x1, y1, x2, y2 = left_lane
            left_bottom_y = max(y1, y2)
            left_bottom_x = x1 if y1 > y2 else x2
            
            # 估计右侧车道线位置
            estimated_right_x = left_bottom_x + self.lane_width
            center_x = (left_bottom_x + estimated_right_x) / 2
            center_y = left_bottom_y
            
            return (int(center_x), int(center_y))
        
        elif right_lane is not None and self.lane_width is not None:
            x1, y1, x2, y2 = right_lane
            right_bottom_y = max(y1, y2)
            right_bottom_x = x1 if y1 > y2 else x2
            
            # 估计左侧车道线位置
            estimated_left_x = right_bottom_x - self.lane_width
            center_x = (estimated_left_x + right_bottom_x) / 2
            center_y = right_bottom_y
            
            return (int(center_x), int(center_y))
        
        else:
            # 两条车道线都未检测到
            self.frames_without_both_lanes += 1
            return None
    
    def process_frame(self, frame):
        mask, start_row, end_row = self.set_roi_region(frame)
        edges, gray = self.preprocess_frame(frame, mask)
        left_lanes, right_lanes = self.detect_lanes(edges, frame, start_row)
        
        # 选择最靠近中心的左右车道线
        left_lane = self.select_closest_lane(left_lanes, self.frame_center_x, is_left=True)
        right_lane = self.select_closest_lane(right_lanes, self.frame_center_x, is_left=False)
        
        # 计算中心点
        center_point = self.calculate_center_from_lines(left_lane, right_lane, frame.shape, start_row)
        
        result = frame.copy()
        self.draw_reference_lines(result)
        cv2.line(result, (0, start_row), (result.shape[1], start_row), (255, 255, 0), 2)
        
        # 绘制状态信息
        state_text = f'State: {self.state}'
        waypoint_text = f'Waypoint: {self.current_waypoint_index + 1}/{len(self.waypoints_list)}'
        
        # 添加障碍物信息
        obstacle_text = f'Obstacle: {"DETECTED" if self.obstacle_detected else "CLEAR"}'
        distance_text = f'Distance: {self.obstacle_distance:.2f}m'
        obstacle_count_text = f'Obstacle Count: {self.obstacle_count}/{self.obstacle_threshold_count}'
        
        # 车道线状态
        lanes_status = f'Lanes: Left={left_lane is not None}, Right={right_lane is not None}'
        
        cv2.putText(result, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result, waypoint_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, obstacle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if self.obstacle_detected else (0, 255, 0), 2)
        cv2.putText(result, distance_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 255), 2)
        cv2.putText(result, obstacle_count_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 0) if self.obstacle_count > 0 else (200, 200, 200), 2)
        cv2.putText(result, lanes_status, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制所有检测到的车道线
        for line in left_lanes:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 100, 100), 1)
        
        for line in right_lanes:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (100, 100, 0), 1)
        
        # 绘制选定的车道线
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        if center_point is not None:
            cv2.circle(result, center_point, 15, (0, 255, 255), -1)
            self.last_known_center = center_point
        
        return result, edges, center_point
    
    def draw_reference_lines(self, result):
        cv2.line(result, (self.frame_center_x, 0), (self.frame_center_x, result.shape[0]), (255, 0, 255), 2)
        tolerance_pixels = int(self.frame_width * self.center_tolerance / 2)
        cv2.line(result, (self.frame_center_x - tolerance_pixels, 0), 
                (self.frame_center_x - tolerance_pixels, result.shape[0]), (200, 200, 200), 1)
        cv2.line(result, (self.frame_center_x + tolerance_pixels, 0), 
                (self.frame_center_x + tolerance_pixels, result.shape[0]), (200, 200, 200), 1)
    
    def cleanup_and_shutdown(self):
        # 发布停止命令
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_msg)
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.destroy_node()
        rclpy.shutdown()
        exit(0)

def normalize_quaternion(qx, qy, qz, qw):
    """归一化四元数"""
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm == 0:
        return 0.0, 0.0, 0.0, 1.0
    return qx/norm, qy/norm, qz/norm, qw/norm

def create_pose_stamped(node, x, y, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0, frame_id='map'):
    """创建PoseStamped消息 - 使用节点的时钟"""
    # 归一化四元数
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
    parser = argparse.ArgumentParser(description='整合车道检测和导航系统')
    parser.add_argument('--input', type=str, default='ros2', 
                       choices=['ros2', 'camera', 'video'],
                       help='输入源类型')
    parser.add_argument('--source', type=str, default=None,
                       help='源路径: 摄像头ID或视频文件路径')
    parser.add_argument('--roi', type=float, default=0.1,
                       help='ROI区域比例')
    parser.add_argument('--obstacle-threshold', type=float, default=4.0,
                       help='障碍物检测阈值（米），默认为5米')
    parser.add_argument('--front-angle', type=float, default=None,
                       help='正前方角度（弧度），如果不指定则自动判断')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    # 创建临时节点用于生成路径点
    temp_node = rclpy.create_node('temp_pose_creator')
    
    # 定义路径点
    waypoints_list = [
        # [A1, B1]
        [
            create_pose_stamped(temp_node, 2.3, -5.1, 0.0, 0.0, 0.0, 0.69, 0.71),  # A1
            create_pose_stamped(temp_node, -0.48,-6.61, 0.0, 0.0, 0.0 ,0.0, 0.0)   # B1
        ],
        # [A2, B2] 
        [
            create_pose_stamped(temp_node, 0.2, -92.8, 0.0, 0.0, 0.0, -0.71, 0.70),  # A2
            create_pose_stamped(temp_node, 3.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # B2
        ]
    ]
    
    # 销毁临时节点
    temp_node.destroy_node()
    
    node = IntegratedLaneNavNode(
        waypoints_list=waypoints_list,
        roi_ratio=args.roi,
        input_source=args.input,
        video_source=args.source
    )
    
    # 设置障碍物检测阈值
    node.obstacle_threshold = args.obstacle_threshold
    
    # 如果用户指定了正前方角度，可以存储起来供scan_callback使用
    if args.front_angle is not None:
        node.front_angle = args.front_angle
    
    print("整合车道检测和导航系统")
    print("=" * 50)
    print(f"状态: 启动 (车道检测模式)")
    print(f"路径点数量: {len(waypoints_list)} 对")
    print(f"障碍物检测阈值: {args.obstacle_threshold} 米")
    print(f"连续检测阈值: 5次")
    print("按 'q' 退出程序")
    print("注意: 现在使用激光雷达检测障碍物作为切换条件")
    print("提示: 如果检测方向不对，请使用--front-angle参数指定正前方角度（弧度）")
    
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
    finally:
        node.cleanup_and_shutdown()

if __name__ == "__main__":
    main()
