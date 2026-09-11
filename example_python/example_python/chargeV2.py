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
    """融合控制 + 最终阶段强制yaw对齐"""

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
        self.declare_parameter('yaw_align_angular_z', 0.3)  # 最终yaw对齐时的旋转速度

        # 控制增益
        self.declare_parameter('k_pixel', 0.8)
        self.declare_parameter('k_yaw', 0.8)                # yaw修正增益（最终对齐时更大）
        self.declare_parameter('angular_sign', -1.0)

        # 距离阈值
        self.declare_parameter('yaw_enable_distance', 1.0)
        self.declare_parameter('final_align_distance', 0.6) # 进入强制yaw对齐的距离

        # 稳定性
        self.declare_parameter('yaw_stability_threshold', 0.08)  # 放宽一些
        self.declare_parameter('yaw_filter_window', 15)

        # 停车阈值
        self.declare_parameter('z_tolerance', 0.03)
        self.declare_parameter('pixel_tolerance', 25)
        self.declare_parameter('yaw_tolerance', 0.08)       # ~4.6°

        # 读取参数
        self.marker_id = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        self.desired_distance = self.get_parameter('desired_distance').value

        self.max_vx = self.get_parameter('max_linear_x').value
        self.min_vx = self.get_parameter('min_linear_x').value
        self.max_vw = self.get_parameter('max_angular_z').value
        self.yaw_align_vw = self.get_parameter('yaw_align_angular_z').value

        self.k_pixel = self.get_parameter('k_pixel').value
        self.k_yaw = self.get_parameter('k_yaw').value
        self.angular_sign = self.get_parameter('angular_sign').value

        self.yaw_enable_dist = self.get_parameter('yaw_enable_distance').value
        self.final_align_dist = self.get_parameter('final_align_distance').value
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
        self.final_yaw_alignment = False  # 是否处于最终yaw对齐阶段

        # ROS2
        self.twist_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, 10)
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('camera_topic').value, self.image_callback, 10)

        self.get_logger().info('✅ 融合+强制yaw对齐节点已启动')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        if not self.cam_ok:
            self.get_logger().info('📸 相机内参已加载')
            self.cam_ok = True

    def get_yaw_stats(self):
        if len(self.yaw_history) < 3:
            return 0.0, 1.0
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

            cx = self.camera_matrix[0, 2]
            center_x = np.mean(marker_corners[0][:, 0])
            pixel_err = center_x - cx

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

            # ---------- 最终yaw对齐逻辑 ----------
            # 如果距离已经小于final_align_distance，且yaw不达标，则进入强制对齐模式
            if (current_distance < self.final_align_dist and 
                abs(filtered_yaw) > self.yaw_tol and
                not self.final_yaw_alignment):
                self.final_yaw_alignment = True
                self.get_logger().info(f'⚠️ 进入最终yaw对齐模式 (距离{current_distance:.2f}m, yaw{math.degrees(filtered_yaw):.1f}°)')

            if self.final_yaw_alignment:
                # 如果yaw已经达标，退出强制对齐模式
                if abs(filtered_yaw) < self.yaw_tol:
                    self.final_yaw_alignment = False
                    self.get_logger().info('✅ 最终yaw对齐完成，继续接近')
                else:
                    # 强制对齐：停止前进，纯yaw旋转
                    vx = 0.0
                    # 仍然监控像素，如果偏太远（>100px）暂时切换为像素伺服防止丢失
                    if abs(pixel_err) > 100:
                        vw = self.angular_sign * (self.k_pixel * pixel_err / cx)
                        self.get_logger().info(f'[最终对齐-像素保护] pix:{pixel_err:.0f}', throttle_duration_sec=0.5)
                    else:
                        # 使用较大的yaw增益，并限制角速度
                        vw = self.angular_sign * (-self.k_yaw * filtered_yaw)
                        vw = max(-self.yaw_align_vw, min(self.yaw_align_vw, vw))
                        self.get_logger().info(
                            f'[最终对齐-旋转] yaw:{math.degrees(filtered_yaw):.1f}° vw:{vw:.2f}',
                            throttle_duration_sec=0.5)
                    twist.linear.x = vx
                    twist.linear.y = 0.0
                    twist.angular.z = vw
                    self.twist_pub.publish(twist)
                    return

            # ---------- 正常融合控制（非强制对齐阶段） ----------
            pixel_vw = self.k_pixel * pixel_err / cx
            yaw_vw = 0.0
            use_yaw = False
            if current_distance < self.yaw_enable_dist and yaw_std < self.yaw_stability_th:
                yaw_vw = -self.k_yaw * filtered_yaw
                use_yaw = True

            distance_factor = max(0.0, 1.0 - current_distance / self.yaw_enable_dist)
            pixel_protection = max(0.0, 1.0 - abs(pixel_err) / 120.0)
            yaw_weight = distance_factor * pixel_protection

            vw = self.angular_sign * (pixel_vw + yaw_weight * yaw_vw)
            vw = max(-self.max_vw, min(self.max_vw, vw))

            vx = self.max_vx * min(1.0, abs(z_err) / 1.0)
            vx = max(self.min_vx, min(self.max_vx, vx))

            twist.linear.x = vx
            twist.linear.y = 0.0
            twist.angular.z = vw

            self.get_logger().info(
                f'距离:{current_distance:.2f}m  pix:{pixel_err:.0f}  '
                f'yaw(滤波):{math.degrees(filtered_yaw):.1f}°  yaw_std:{math.degrees(yaw_std):.1f}°  '
                f'融合:{"是" if use_yaw else "否"}  vx:{vx:.2f}  vw:{vw:.2f}',
                throttle_duration_sec=0.5)
        else:
            # 搜索
            twist.angular.z = 0.3 * self.angular_sign
            twist.linear.x = 0.0
            self.final_yaw_alignment = False  # 丢失目标后重置

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
