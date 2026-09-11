#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from collections import deque


class ChargingDockNode(Node):
    """融合控制：像素伺服 + 近距离稳定yaw修正"""

    def __init__(self):
        super().__init__('charging_dock_node')

        # ---------- 参数 ----------
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0.08)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('desired_distance', 0.40)

        # 速度
        self.declare_parameter('max_linear_x', 0.10)
        self.declare_parameter('min_linear_x', 0.03)
        self.declare_parameter('max_angular_z', 0.4)

        # 控制增益
        self.declare_parameter('k_pixel', 0.8)          # 像素偏差增益（全程）
        self.declare_parameter('k_yaw', 0.4)            # yaw修正增益（近距离）
        self.declare_parameter('angular_sign', -1.0)    # 旋转方向符号

        # 融合距离阈值
        self.declare_parameter('yaw_enable_distance', 0.8)  # 小于此距离启用yaw修正
        self.declare_parameter('yaw_stability_threshold', 0.05) # yaw标准差低于此值视为稳定

        # 停车阈值
        self.declare_parameter('z_tolerance', 0.03)
        self.declare_parameter('pixel_tolerance', 25)
        self.declare_parameter('yaw_tolerance', 0.10)       # ~5.7°

        # 滤波
        self.declare_parameter('yaw_filter_window', 15)     # 滤波窗口

        # 读取参数
        self.marker_id = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        self.desired_distance = self.get_parameter('desired_distance').value

        self.max_vx = self.get_parameter('max_linear_x').value
        self.min_vx = self.get_parameter('min_linear_x').value
        self.max_vw = self.get_parameter('max_angular_z').value

        self.k_pixel = self.get_parameter('k_pixel').value
        self.k_yaw = self.get_parameter('k_yaw').value
        self.angular_sign = self.get_parameter('angular_sign').value

        self.yaw_enable_dist = self.get_parameter('yaw_enable_distance').value
        self.yaw_stability_th = self.get_parameter('yaw_stability_threshold').value

        self.z_tol = self.get_parameter('z_tolerance').value
        self.pixel_tol = self.get_parameter('pixel_tolerance').value
        self.yaw_tol = self.get_parameter('yaw_tolerance').value

        filter_window = self.get_parameter('yaw_filter_window').value
        self.yaw_history = deque(maxlen=filter_window)

        # 相机
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cam_ok = False

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.bridge = CvBridge()

        self.docked = False

        # ROS2
        self.twist_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, 10)
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('camera_topic').value, self.image_callback, 10)

        self.get_logger().info('✅ 融合控制对接节点已启动 (像素+yaw)')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        if not self.cam_ok:
            self.get_logger().info('📸 相机内参已加载')
            self.cam_ok = True

    def get_yaw_stats(self):
        """返回滤波后的yaw及其标准差"""
        if len(self.yaw_history) < 3:
            return 0.0, 1.0  # 数据不足，认为不稳定
        arr = np.array(self.yaw_history)
        return float(np.mean(arr)), float(np.std(arr))

    def image_callback(self, msg: Image):
        if self.docked:
            self.twist_pub.publish(Twist())
            return
        if not self.cam_ok:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        twist = Twist()

        if ids is not None and self.marker_id in ids.flatten():
            idx = list(ids.flatten()).index(self.marker_id)
            marker_corners = corners[idx]

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], self.marker_size, self.camera_matrix, self.dist_coeffs)
            tvec = tvecs[0][0]
            current_distance = tvec[2]
            z_err = current_distance - self.desired_distance

            # 像素偏差
            cx = self.camera_matrix[0, 2]
            center_x = np.mean(marker_corners[0][:, 0])
            pixel_err = center_x - cx

            # 原始yaw并存入滤波队列
            R, _ = cv2.Rodrigues(rvecs[0][0])
            normal = R[:, 2]
            raw_yaw = math.atan2(normal[0], -normal[2])
            self.yaw_history.append(raw_yaw)
            filtered_yaw, yaw_std = self.get_yaw_stats()

            # 对接成功判定
            if (abs(z_err) < self.z_tol and
                abs(pixel_err) < self.pixel_tol and
                abs(filtered_yaw) < self.yaw_tol):
                self.get_logger().info('✅✅✅ 对接成功！')
                self.docked = True
                self.twist_pub.publish(Twist())
                return

            # ---------- 融合角速度计算 ----------
            # 1. 像素伺服项（全程）
            pixel_vw = self.k_pixel * pixel_err / cx

            # 2. yaw修正项（仅在近距离且yaw稳定时启用）
            yaw_vw = 0.0
            use_yaw = False
            if current_distance < self.yaw_enable_dist and yaw_std < self.yaw_stability_th:
                yaw_vw = -self.k_yaw * filtered_yaw   # 负反馈
                use_yaw = True

            # 3. 权重：yaw修正的权重随距离减小而增大，且当像素偏差大时降低
            distance_factor = max(0.0, 1.0 - current_distance / self.yaw_enable_dist)  # 0→1
            pixel_protection = max(0.0, 1.0 - abs(pixel_err) / 120.0)  # 像素偏差>120时权重降为0
            yaw_weight = distance_factor * pixel_protection

            # 4. 合成角速度
            vw = self.angular_sign * (pixel_vw + yaw_weight * yaw_vw)
            vw = max(-self.max_vw, min(self.max_vw, vw))

            # 前进速度：距离越近越慢
            vx = self.max_vx * min(1.0, abs(z_err) / 1.0)  # 1米时满速
            vx = max(self.min_vx, min(self.max_vx, vx))

            twist.linear.x = vx
            twist.linear.y = 0.0
            twist.angular.z = vw

            self.get_logger().info(
                f'距离:{current_distance:.2f}m  pix:{pixel_err:.0f}  '
                f'yaw(滤波/原始):{math.degrees(filtered_yaw):.1f}/{math.degrees(raw_yaw):.1f}°  '
                f'yaw_std:{math.degrees(yaw_std):.1f}°  '
                f'融合:{"是" if use_yaw else "否"}  vx:{vx:.2f}  vw:{vw:.2f}',
                throttle_duration_sec=0.5)
        else:
            # 搜索
            twist.angular.z = 0.3 * self.angular_sign
            twist.linear.x = 0.0

        self.twist_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ChargingDockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.twist_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
