import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math
import argparse
import os

class ChickenFarmLaneDetector(Node):
    def __init__(self, roi_ratio=0.6, input_source="ros2", video_source=None):
        # 初始化ROS2节点
        super().__init__('chicken_farm_lane_detector')
        
        # 创建CV桥接器，用于ROS2图像消息和OpenCV图像之间的转换
        self.bridge = CvBridge()
        
        # 创建Twist消息发布者
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 存储输入源配置
        self.input_source = input_source
        self.video_source = video_source
        self.cap = None
        
        # 控制参数
        self.previous_left_fit = None
        self.previous_right_fit = None
        self.smooth_factor = 0.7
        self.roi_ratio = roi_ratio
        
        # 新增：车道宽度和状态跟踪
        self.lane_width = None  # 存储车道宽度（像素）
        self.last_known_center = None  # 最后已知的中心点
        self.frames_without_both_lanes = 0  # 连续丢失双车道的帧数
        self.max_frames_without_lanes = 200  # 最大允许丢失帧数
        
        # 中心点控制参数
        self.center_tolerance = 0.1  # 容错距离（画面宽度的10%）
        self.linear_speed = 0.1      # 前进速度
        self.angular_speed = 0.1     # 转向速度
        
        # 根据输入源类型初始化
        self.setup_input_source()
        
        # 创建定时器用于定期发布控制命令
        self.control_timer = self.create_timer(0.1, self.control_timer_callback)
        
        # 存储当前检测结果
        self.current_center_point = None
        self.frame_width = 640  # 默认宽度，会在处理第一帧时更新
        self.frame_center_x = 0
        self.current_frame = None
        
        # 创建可调整大小的窗口
        cv2.namedWindow('Near Area Lane Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Edges (Near Area)', cv2.WINDOW_NORMAL)
        
        self.get_logger().info(f'养鸡场过道检测器已启动，输入源: {self.input_source}')
    
    def setup_input_source(self):
        """根据输入源类型设置数据源"""
        if self.input_source == "ros2":
            # ROS2话题订阅模式
            self.setup_ros2_subscriber()
        elif self.input_source == "camera":
            # 摄像头直接读取模式
            self.setup_camera_capture()
        elif self.input_source == "video":
            # 视频文件读取模式
            self.setup_video_capture()
        else:
            self.get_logger().error(f'不支持的输入源类型: {self.input_source}')
            return
    
    def setup_ros2_subscriber(self):
        """设置ROS2图像订阅者"""
        # 创建图像订阅者，订阅/camera/color/image_raw话题
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.ros2_image_callback,
            10  # 队列大小
        )
        self.get_logger().info('已订阅ROS2话题: /camera/color/image_raw')
    
    def setup_camera_capture(self):
        """设置摄像头捕获"""
        try:
            # 尝试解析摄像头ID
            if self.video_source is None:
                camera_id = 0
            else:
                camera_id = int(self.video_source)
            
            self.cap = cv2.VideoCapture(camera_id)
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                self.get_logger().error(f'无法打开摄像头 ID: {camera_id}')
                # 尝试其他可能的摄像头设备
                for device_id in [0, 1, 2]:
                    self.cap = cv2.VideoCapture(device_id)
                    if self.cap.isOpened():
                        camera_id = device_id
                        self.get_logger().info(f'成功打开摄像头 ID: {device_id}')
                        break
                else:
                    self.get_logger().error('无法找到可用的摄像头设备')
                    return
            
            self.get_logger().info(f'已打开摄像头: /dev/video{camera_id}')
            
            # 创建定时器用于定期从摄像头读取帧
            self.camera_timer = self.create_timer(0.033, self.camera_capture_callback)  # 约30fps
            
        except Exception as e:
            self.get_logger().error(f'摄像头初始化错误: {str(e)}')
    
    def setup_video_capture(self):
        """设置视频文件捕获"""
        if self.video_source is None:
            self.get_logger().error('视频模式需要指定视频文件路径')
            return
        
        if not os.path.exists(self.video_source):
            self.get_logger().error(f'视频文件不存在: {self.video_source}')
            return
        
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开视频文件: {self.video_source}')
            return
        
        self.get_logger().info(f'已打开视频文件: {self.video_source}')
        
        # 创建定时器用于定期从视频读取帧
        self.video_timer = self.create_timer(0.033, self.video_capture_callback)  # 约30fps
    
    def ros2_image_callback(self, msg):
        """ROS2图像消息回调函数"""
        try:
            # 将ROS2图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_and_display_frame(cv_image)
            
        except Exception as e:
            self.get_logger().error(f'ROS2图像处理错误: {str(e)}')
    
    def camera_capture_callback(self):
        """摄像头捕获回调函数"""
        try:
            # 从摄像头读取帧
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display_frame(frame)
            else:
                self.get_logger().warning('从摄像头读取帧失败')
                
        except Exception as e:
            self.get_logger().error(f'摄像头捕获错误: {str(e)}')
    
    def video_capture_callback(self):
        """视频文件捕获回调函数"""
        try:
            # 从视频文件读取帧
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display_frame(frame)
            else:
                # 视频播放结束，重新开始或退出
                self.get_logger().info('视频播放结束')
                # 可以选择重新播放
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
        except Exception as e:
            self.get_logger().error(f'视频捕获错误: {str(e)}')
    
    def process_and_display_frame(self, frame):
        """处理并显示帧（通用函数，适用于所有输入源）"""
        # 更新当前帧
        self.current_frame = frame
        
        # 处理当前帧
        result, edges, center_point = self.process_frame(frame)
        
        # 显示结果
        cv2.imshow('Near Area Lane Detection', result)
        cv2.imshow('Edges (Near Area)', edges)
        
        # 更新当前中心点
        self.current_center_point = center_point
        
        # 处理OpenCV窗口事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('检测到退出命令')
            self.cleanup_and_shutdown()
    
    def control_timer_callback(self):
        """定时发布控制命令"""
        if self.current_center_point is not None or self.frames_without_both_lanes > 0:
            self.publish_control_command()
    
    def publish_control_command(self):
        """根据中心点位置或丢失车道状态发布控制命令"""
        msg = Twist()
        
        if self.current_center_point is not None:
            # 正常情况：基于检测到的中心点控制
            offset = (self.current_center_point[0] - self.frame_center_x) / (self.frame_width / 2)
            tolerance = self.center_tolerance
            
            if abs(offset) < tolerance:
                # 在中心区域，直行
                msg.linear.x = self.linear_speed
                self.get_logger().info('Centered - Moving forward')
            elif offset < -tolerance:
                # 偏左，向右转
                msg.linear.x = self.linear_speed
                msg.linear.y = self.angular_speed
                self.get_logger().info('Left of center - Turning right')
            else:
                # 偏右，向左转
                msg.linear.x = self.linear_speed
                msg.linear.y = -self.angular_speed
                self.get_logger().info('Right of center - Turning left')
            
            # 重置丢失计数器
            self.frames_without_both_lanes = 0
            
        else:
            # 无法检测到中心点，根据丢失帧数采取不同策略
            if self.frames_without_both_lanes < self.max_frames_without_lanes:
                # 短暂丢失：保持直行
                msg.linear.x = self.linear_speed  # 降低速度
                self.get_logger().warn(f'Lanes lost - Keeping straight ({self.frames_without_both_lanes}/{self.max_frames_without_lanes})')
            else:
                # 长时间丢失：停止
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.get_logger().error('Lanes lost for too long - Stopping')
        
        # 发布控制消息
        self.cmd_vel_publisher.publish(msg)
    
    def estimate_missing_lane(self, detected_fit, is_left_lane, frame_shape):
        """估算缺失的车道线位置"""
        height, width = frame_shape[:2]
        
        if self.lane_width is None or detected_fit is None:
            return None
        
        # 计算检测到的车道线在底部的位置
        y_bottom = height - 1
        detected_x_bottom = detected_fit[0] * y_bottom + detected_fit[1]
        
        # 根据车道宽度估算缺失的车道线位置
        if is_left_lane:
            # 缺失的是右车道线，在左车道线右侧加上车道宽度
            estimated_x_bottom = detected_x_bottom + self.lane_width
        else:
            # 缺失的是左车道线，在右车道线左侧减去车道宽度
            estimated_x_bottom = detected_x_bottom - self.lane_width
        
        # 计算斜率（使用检测到的车道线的斜率，或者使用默认值）
        if self.previous_left_fit is not None and self.previous_right_fit is not None:
            # 使用历史斜率的平均值
            avg_slope = (self.previous_left_fit[0] + self.previous_right_fit[0]) / 2
        else:
            # 使用默认斜率（根据车道线通常的倾斜程度）
            avg_slope = 0.0 if is_left_lane else 0.0
        
        # 计算截距
        estimated_intercept = estimated_x_bottom - avg_slope * y_bottom
        
        return np.array([avg_slope, estimated_intercept])
    
    def calculate_single_center_point(self, left_fit, right_fit, frame_shape, start_row):
        """计算中心点，支持单车道线估算"""
        height, width = frame_shape[:2]
        
        # 更新车道宽度（当两条线都检测到时）
        if left_fit is not None and right_fit is not None:
            y_bottom = height - 1
            left_x_bottom = left_fit[0] * y_bottom + left_fit[1]
            right_x_bottom = right_fit[0] * y_bottom + right_fit[1]
            self.lane_width = abs(right_x_bottom - left_x_bottom)
            self.frames_without_both_lanes = 0  # 重置计数器
        
        # 处理不同类型的车道线检测情况
        if left_fit is not None and right_fit is not None:
            # 情况1：两条车道线都检测到
            self.frames_without_both_lanes = 0
            return self._calculate_center_from_both_lanes(left_fit, right_fit, frame_shape, start_row)
        
        elif left_fit is not None and self.lane_width is not None:
            # 情况2：只检测到左车道线，估算右车道线
            self.frames_without_both_lanes += 1
            estimated_right_fit = self.estimate_missing_lane(left_fit, True, frame_shape)
            if estimated_right_fit is not None:
                return self._calculate_center_from_both_lanes(left_fit, estimated_right_fit, frame_shape, start_row)
        
        elif right_fit is not None and self.lane_width is not None:
            # 情况3：只检测到右车道线，估算左车道线
            self.frames_without_both_lanes += 1
            estimated_left_fit = self.estimate_missing_lane(right_fit, False, frame_shape)
            if estimated_left_fit is not None:
                return self._calculate_center_from_both_lanes(estimated_left_fit, right_fit, frame_shape, start_row)
        else:
            # 情况4：两条车道线都未检测到，或没有足够的车道宽度信息
            self.frames_without_both_lanes += 1
            return None
        
        return None
    
    def _calculate_center_from_both_lanes(self, left_fit, right_fit, frame_shape, start_row):
        """从两条车道线计算中心点（内部方法）"""
        height, width = frame_shape[:2]
        
        # 在ROI区域的顶部和底部分别计算中点
        y_top = start_row
        y_bottom = height - 1
        
        # 计算左右直线在顶部和底部对应的x坐标
        left_x_top = left_fit[0] * y_top + left_fit[1]
        right_x_top = right_fit[0] * y_top + right_fit[1]
        left_x_bottom = left_fit[0] * y_bottom + left_fit[1]
        right_x_bottom = right_fit[0] * y_bottom + right_fit[1]
        
        # 计算顶部和底部的中点
        center_x_top = (left_x_top + right_x_top) / 2
        center_x_bottom = (left_x_bottom + right_x_bottom) / 2
        
        # 计算这两个中点连线的中心点
        center_x = (center_x_top + center_x_bottom) / 2
        center_y = (y_top + y_bottom) / 2
        
        center_point = (int(center_x), int(center_y))
        
        # 保存最后已知的中心点
        self.last_known_center = center_point
        
        return center_point
    
    def set_roi_region(self, frame):
        """设置感兴趣区域（近处）"""
        height, width = frame.shape[:2]
        # 更新帧宽度和中心点
        self.frame_width = width
        self.frame_center_x = width // 2
        
        # 只取图像下半部分，根据roi_ratio调整
        start_row = int(height * (1 - self.roi_ratio))
        end_row = height
        
        # 创建ROI掩码
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[start_row:end_row, :] = 255
        
        return mask, start_row, end_row
    
    def preprocess_frame(self, frame, mask):
        """图像预处理，只处理ROI区域"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_roi = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_and(edges_roi, edges_roi, mask=mask)
        return edges, gray
    
    def detect_lanes(self, edges, original_frame, start_row):
        """检测车道线"""
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
    
    def fit_lane_lines(self, left_lanes, right_lanes, frame_shape):
        """拟合车道线"""
        height, width = frame_shape[:2]
        
        left_fit = None
        right_fit = None
        
        # 拟合左侧直线
        if len(left_lanes) > 0:
            left_points = []
            for line in left_lanes:
                x1, y1, x2, y2 = line
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            
            left_points = np.array(left_points)
            if len(left_points) > 1:
                left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 1)
        
        # 拟合右侧直线
        if len(right_lanes) > 0:
            right_points = []
            for line in right_lanes:
                x1, y1, x2, y2 = line
                right_points.append([x1, y1])
                right_points.append([x2, y2])
            
            right_points = np.array(right_points)
            if len(right_points) > 1:
                right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 1)
        
        # 平滑处理
        if left_fit is not None:
            if self.previous_left_fit is not None:
                left_fit = (self.smooth_factor * np.array(self.previous_left_fit) + 
                          (1 - self.smooth_factor) * np.array(left_fit))
            self.previous_left_fit = left_fit
        elif self.previous_left_fit is not None:
            #left_fit = self.previous_left_fit
            pass
        
        if right_fit is not None:
            if self.previous_right_fit is not None:
                right_fit = (self.smooth_factor * np.array(self.previous_right_fit) + 
                           (1 - self.smooth_factor) * np.array(right_fit))
            self.previous_right_fit = right_fit
        elif self.previous_right_fit is not None:
            #right_fit = self.previous_right_fit
            pass
        
        return left_fit, right_fit
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 1. 设置ROI区域
        mask, start_row, end_row = self.set_roi_region(frame)
        
        # 2. 预处理
        edges, gray = self.preprocess_frame(frame, mask)
        
        # 3. 检测直线
        left_lanes, right_lanes = self.detect_lanes(edges, frame, start_row)
        
        # 4. 拟合直线
        left_fit, right_fit = self.fit_lane_lines(left_lanes, right_lanes, frame.shape)
        
        # 5. 计算中心点（支持单车道线估算）
        center_point = self.calculate_single_center_point(left_fit, right_fit, frame.shape, start_row)
        
        # 6. 创建结果图像
        result = frame.copy()
        
        # 绘制画面中心线和容错区域
        self.draw_reference_lines(result)
        
        # 绘制ROI区域边界
        cv2.line(result, (0, start_row), (result.shape[1], start_row), (255, 255, 0), 2)
        cv2.putText(result, 'NEAR AREA', (10, start_row - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 绘制检测到的线段和拟合的直线
        self.draw_detected_lanes(result, left_lanes, right_lanes, left_fit, right_fit, start_row)
        
        # 绘制中心点和状态信息
        self.draw_center_point_and_info(result, center_point, left_fit, right_fit, left_lanes, right_lanes)
        
        return result, edges, center_point
    
    def draw_reference_lines(self, result):
        """绘制参考线"""
        cv2.line(result, (self.frame_center_x, 0), (self.frame_center_x, result.shape[0]), (255, 0, 255), 2)
        tolerance_pixels = int(self.frame_width * self.center_tolerance / 2)
        cv2.line(result, (self.frame_center_x - tolerance_pixels, 0), 
                (self.frame_center_x - tolerance_pixels, result.shape[0]), (200, 200, 200), 1)
        cv2.line(result, (self.frame_center_x + tolerance_pixels, 0), 
                (self.frame_center_x + tolerance_pixels, result.shape[0]), (200, 200, 200), 1)
    
    def draw_detected_lanes(self, result, left_lanes, right_lanes, left_fit, right_fit, start_row):
        """绘制检测到的车道线"""
        # 绘制原始检测线段
        for line in left_lanes:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        for line in right_lanes:
            x1, y1, x2, y2 = line
            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制拟合的直线
        if left_fit is not None:
            y1 = start_row
            y2 = result.shape[0] - 1
            x1 = int(left_fit[0] * y1 + left_fit[1])
            x2 = int(left_fit[0] * y2 + left_fit[1])
            # 根据是否估算来改变颜色
            color = (0, 100, 255) if self.frames_without_both_lanes > 0 and right_fit is None else (0, 100, 255)
            cv2.line(result, (x1, y1), (x2, y2), color, 3)
        
        if right_fit is not None:
            y1 = start_row
            y2 = result.shape[0] - 1
            x1 = int(right_fit[0] * y1 + right_fit[1])
            x2 = int(right_fit[0] * y2 + right_fit[1])
            color = (255, 100, 0) if self.frames_without_both_lanes > 0 and left_fit is None else (255, 100, 0)
            cv2.line(result, (x1, y1), (x2, y2), color, 3)
    
    def draw_center_point_and_info(self, result, center_point, left_fit, right_fit, left_lanes, right_lanes):
        """绘制中心点和状态信息"""
        # 绘制中心点
        if center_point is not None:
            cv2.circle(result, center_point, 15, (0, 255, 255), -1)
            cv2.circle(result, center_point, 15, (0, 0, 0), 2)
            cv2.putText(result, 'CENTER POINT', 
                       (center_point[0] - 60, center_point[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示车道线检测状态
        lane_status = "BOTH LANES"
        status_color = (0, 255, 0)  # 绿色
        
        if left_fit is None and right_fit is None:
            lane_status = "NO LANES"
            status_color = (0, 0, 255)  # 红色
        elif left_fit is None:
            lane_status = "ESTIMATED LEFT"
            status_color = (0, 165, 255)  # 橙色
        elif right_fit is None:
            lane_status = "ESTIMATED RIGHT" 
            status_color = (0, 165, 255)  # 橙色
        
        cv2.putText(result, f'Lane Status: {lane_status}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 显示输入源信息
        source_text = f'Input: {self.input_source.upper()}'
        if self.input_source == "camera":
            source_text += f' (Camera {self.video_source if self.video_source else "0"})'
        elif self.input_source == "video":
            source_text += f' ({os.path.basename(self.video_source)})'
        else:
            source_text += ' (/camera/color/image_raw)'
        
        cv2.putText(result, source_text, (10, result.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示车道宽度信息
        if self.lane_width is not None:
            cv2.putText(result, f'Lane Width: {self.lane_width:.1f}px', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示丢失帧数
        if self.frames_without_both_lanes > 0:
            cv2.putText(result, f'Lost frames: {self.frames_without_both_lanes}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示中心点坐标
        if center_point is not None:
            cv2.putText(result, f'Center: {center_point}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def cleanup_and_shutdown(self):
        """清理资源并关闭程序"""
        # 发布停止命令
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
        
        # 释放摄像头资源
        if self.cap is not None:
            self.cap.release()
        
        # 关闭OpenCV窗口
        cv2.destroyAllWindows()
        
        # 关闭ROS2节点
        self.destroy_node()
        rclpy.shutdown()
        
        print("程序已安全退出")
        exit(0)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='养鸡场过道中线检测系统')
    parser.add_argument('--input', type=str, default='ros2', 
                       choices=['ros2', 'camera', 'video'],
                       help='输入源类型: ros2 (ROS2话题), camera (摄像头), video (视频文件)')
    parser.add_argument('--source', type=str, default='/home/sbiao/ros2_ws/src/example_python/example_python/3.mp4',
                       help='源路径: 摄像头ID (如0,1,2) 或视频文件路径')
    parser.add_argument('--roi', type=float, default=0.2,
                       help='ROI区域比例 (0.1-0.9)')
    
    args = parser.parse_args()
    
    # 初始化ROS2
    rclpy.init()
    
    # 初始化检测器
    detector = ChickenFarmLaneDetector(
        roi_ratio=args.roi,
        input_source=args.input,
        video_source=args.source
    )
    
    print("养鸡场过道中线检测系统 - 多输入源版本")
    print("=" * 60)
    print(f"输入源: {args.input.upper()}")
    
    if args.input == "camera":
        camera_id = args.source if args.source else "0"
        print(f"摄像头设备: /dev/video{camera_id}")
    elif args.input == "video":
        video_path = args.source if args.source else "未指定"
        print(f"视频文件: {video_path}")
    else:
        print("ROS2话题: /camera/color/image_raw")
    
    print("控制命令: ROS2话题 /cmd_vel")
    print("按 'q' 退出程序")
    print("等待图像数据...")
    
    try:
        # 使用spin保持节点运行
        rclpy.spin(detector)
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {str(e)}")
    finally:
        detector.cleanup_and_shutdown()

if __name__ == "__main__":
    main()
