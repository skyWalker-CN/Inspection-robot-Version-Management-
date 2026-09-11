#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from collections import deque
from tf2_ros import TransformListener
import tf2_geometry_msgs
# 注：为简化航向角计算，可直接从四元数转欧拉角


class ChargingDockNode(Node):
    """带尾部对接的充电节点：最终旋转180°后倒退"""

    def __init__(self):
        super().__init__('charging_dock_node')

        # ---------- 参数 ----------
        self.declare_parameter('marker_id', 0)
        self.declare_parameter('marker_size', 0.08)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('desired_distance', 0.40)      # 正常对接目标距离（不再使用）

        # 尾部对接参数
        self.declare_parameter('turn_distance', 0.70)         # 触发转向的距离 (m)
        self.declare_parameter('back_distance', 0.20)         # 倒退目标距离 (m)
        self.declare_parameter('turn_angle_tolerance', 0.1)   # 转向角度容差 (rad)
        self.declare_parameter('turn_angular_speed', 0.5)     # 转向角速度 (rad/s)
        self.declare_parameter('back_linear_speed', -0.05)    # 倒退速度 (m/s)

        # 原有融合控制参数
        self.declare_parameter('max_linear_x', 0.10)
        self.declare_parameter('min_linear_x', 0.03)
        self.declare_parameter('max_angular_z', 0.4)
        self.declare_parameter('k_pixel', 0.8)
        self.declare_parameter('k_yaw', 0.8)
        self.declare_parameter('angular_sign', -1.0)
        self.declare_parameter('yaw_enable_distance', 0.8)
        self.declare_parameter('yaw_stability_threshold', 0.05)
        self.declare_parameter('yaw_filter_window', 15)
        self.declare_parameter('z_tolerance', 0.03)
        self.declare_parameter('pixel_tolerance', 25)
        self.declare_parameter('yaw_tolerance', 0.08)

        # 读取参数
        self.marker_id = self.get_parameter('marker_id').value
        self.marker_size = self.get_parameter('marker_size').value
        self.desired_distance = self.get_parameter('desired_distance').value
        self.turn_distance = self.get_parameter('turn_distance').value
        self.back_distance = self.get_parameter('back_distance').value
        self.turn_angle_tol = self.get_parameter('turn_angle_tolerance').value
        self.turn_ang_speed = self.get_parameter('turn_angular_speed').value
        self.back_lin_speed = self.get_parameter('back_linear_speed').value

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

        # 状态机
        self.state = 'NORMAL'  # NORMAL, TURNING, BACKING, DOCKED
        self.turn_start_yaw = None
        self.turn_target_yaw = None
        self.back_start_pos = None   # 记录倒退起始位置 (x, y)
        self.back_accum_dist = 0.0

        # 里程计数据
        self.current_odom = None
        self.odom_received = False

        # 相机
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cam_ok = False

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.bridge = CvBridge()

        # ROS2
        self.twist_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.camera_info_callback, 10)
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('camera_topic').value, self.image_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)

        self.get_logger().info('✅ 尾部对接节点已启动（融合+180°转向+倒退）')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        if not self.cam_ok:
            self.get_logger().info('📸 相机内参已加载')
            self.cam_ok = True

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg
        self.odom_received = True

    def get_yaw_from_odom(self):
        """从里程计四元数提取航向角"""
        if self.current_odom is None:
            return 0.0
        q = self.current_odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def get_position_from_odom(self):
        """返回 (x, y) 位置"""
        if self.current_odom is None:
            return (0.0, 0.0)
        return (self.current_odom.pose.pose.position.x,
                self.current_odom.pose.pose.position.y)

    def normalize_angle(self, angle):
        """将角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_yaw_stats(self):
        if len(self.yaw_history) < 3:
            return 0.0, 1.0
        arr = np.array(self.yaw_history)
        return float(np.mean(arr)), float(np.std(arr))

    def image_callback(self, msg: Image):
        # 若已对接，保持停止
        if self.state == 'DOCKED':
            self.twist_pub.publish(Twist())
            return

        # 确保里程计已收到
        if not self.odom_received:
            self.get_logger().warn('等待里程计数据...', throttle_duration_sec=1.0)
            return

        twist = Twist()

        # ========== 状态机分支 ==========
        if self.state == 'TURNING':
            # 旋转180度阶段
            current_yaw = self.get_yaw_from_odom()
            angle_diff = self.normalize_angle(self.turn_target_yaw - current_yaw)
            if abs(angle_diff) < self.turn_angle_tol:
                # 转向完成，进入倒退
                self.state = 'BACKING'
                self.back_start_pos = self.get_position_from_odom()
                self.back_accum_dist = 0.0
                self.get_logger().info('✅ 转向180°完成，开始倒退')
            else:
                # 控制旋转方向
                vw = self.turn_ang_speed if angle_diff > 0 else -self.turn_ang_speed
                twist.angular.z = vw
                twist.linear.x = 0.0
                self.get_logger().info(
                    f'[TURNING] 当前yaw:{math.degrees(current_yaw):.1f}° 目标:{math.degrees(self.turn_target_yaw):.1f}° 差:{math.degrees(angle_diff):.1f}°',
                    throttle_duration_sec=0.5)
            self.twist_pub.publish(twist)
            return

        if self.state == 'BACKING':
            # 倒退阶段
            current_pos = self.get_position_from_odom()
            dx = current_pos[0] - self.back_start_pos[0]
            dy = current_pos[1] - self.back_start_pos[1]
            self.back_accum_dist = math.sqrt(dx*dx + dy*dy)
            if self.back_accum_dist >= self.back_distance:
                # 倒退完成，对接成功
                self.state = 'DOCKED'
                self.get_logger().info('✅✅✅ 尾部对接成功！')
                self.twist_pub.publish(Twist())
                return
            else:
                twist.linear.x = self.back_lin_speed   # 负值后退
                twist.angular.z = 0.0
                self.get_logger().info(
                    f'[BACKING] 已倒退:{self.back_accum_dist:.2f}m / {self.back_distance}m',
                    throttle_duration_sec=0.5)
            self.twist_pub.publish(twist)
            return

        # ========== NORMAL 状态：原有融合控制 + 触发转向检测 ==========
        if not self.cam_ok:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and self.marker_id in ids.flatten():
            idx = list(ids.flatten()).index(self.marker_id)
            marker_corners = corners[idx]

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners], self.marker_size, self.camera_matrix, self.dist_coeffs)
            tvec = tvecs[0][0]
            current_distance = tvec[2]
            z_err = current_distance - self.desired_distance  # 不再使用，仅用于日志

            cx = self.camera_matrix[0, 2]
            center_x = np.mean(marker_corners[0][:, 0])
            pixel_err = center_x - cx

            R, _ = cv2.Rodrigues(rvecs[0][0])
            normal = R[:, 2]
            raw_yaw = math.atan2(normal[0], -normal[2])
            self.yaw_history.append(raw_yaw)
            filtered_yaw, yaw_std = self.get_yaw_stats()

            # 检查是否触发180°转向
            if current_distance <= self.turn_distance:
                self.state = 'TURNING'
                self.turn_start_yaw = self.get_yaw_from_odom()
                # 目标航向：当前航向 + 180° (pi)，取最短路径已在控制中处理
                self.turn_target_yaw = self.normalize_angle(self.turn_start_yaw + math.pi)
                self.get_logger().info(f'🔁 触发180°转向 (距离{current_distance:.2f}m)')
                self.twist_pub.publish(Twist())  # 先停车
                return

            # 正常融合控制（同之前代码）
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
                f'[NORMAL] 距离:{current_distance:.2f}m  pix:{pixel_err:.0f}  '
                f'yaw(滤波):{math.degrees(filtered_yaw):.1f}°  std:{math.degrees(yaw_std):.1f}°  '
                f'vx:{vx:.2f}  vw:{vw:.2f}',
                throttle_duration_sec=0.5)
        else:
            # 未检测到二维码，但在NORMAL状态才搜索
            twist.angular.z = 0.3 * self.angular_sign
            twist.linear.x = 0.0
            self.get_logger().info('🔍 搜索...', throttle_duration_sec=1.0)

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
