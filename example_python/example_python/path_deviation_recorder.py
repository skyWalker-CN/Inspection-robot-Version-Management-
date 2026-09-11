#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径偏差记录节点
- 订阅 /plan 获取 Nav2 实时规划路径
- 通过 TF 同时记录：
    1. 定位偏差：map -> base_link（AMCL 定位后的位置）
    2. 物理偏差：odom -> base_link 再用首次 /plan 时的 odom->map 基准转到 map 系
- 保存 CSV，供任务结束后离线可视化
"""

import csv
import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped


class PathDeviationRecorder(Node):
    def __init__(self):
        super().__init__('path_deviation_recorder')

        # 输出目录：~/path_deviation_results/时间戳/
        self.out_dir = os.path.expanduser(
            os.path.join('~', 'path_deviation_results',
                         datetime.now().strftime('%Y%m%d_%H%M%S')))
        os.makedirs(self.out_dir, exist_ok=True)

        self.deviation_csv = os.path.join(self.out_dir, 'deviation.csv')
        self.plan_csv = os.path.join(self.out_dir, 'plan.csv')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 初始 odom->map 基准，只在第一次收到 /plan 时保存
        self.odom_to_map_ref = None

        # 当前最新规划路径
        self.plan = None
        self.plan_points = []
        self.plan_seq = 0
        self.first_plan_published = False

        # 订阅 Nav2 全局路径
        # Nav2 的 /plan 默认是 VOLATILE，订阅端也必须用 VOLATILE 才能收到
        plan_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, plan_qos
        )

        # 可选：实时发布，方便任务中也能看
        path_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.loc_path_pub = self.create_publisher(
            Path, '/deviation/actual_path_loc', path_qos)
        self.phys_path_pub = self.create_publisher(
            Path, '/deviation/actual_path_phys', path_qos)
        self.plan_path_pub = self.create_publisher(
            Path, '/deviation/planned_path', path_qos)
        self.initial_plan_path_pub = self.create_publisher(
            Path, '/deviation/planned_path_first', path_qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/deviation/error_markers', path_qos)
        self.loc_error_pub = self.create_publisher(
            Float32, '/deviation/loc_abs_error', 10)
        self.phys_error_pub = self.create_publisher(
            Float32, '/deviation/phys_abs_error', 10)

        # 10Hz 记录
        self.timer = self.create_timer(0.1, self.timer_callback)

        # 实际轨迹累积
        self.loc_path = Path()
        self.loc_path.header.frame_id = 'map'
        self.phys_path = Path()
        self.phys_path.header.frame_id = 'map'

        # CSV
        self.dev_file = open(self.deviation_csv, 'w', newline='')
        self.dev_writer = csv.writer(self.dev_file)
        self.dev_writer.writerow([
            'timestamp',
            'loc_x', 'loc_y', 'loc_yaw',
            'loc_abs_err', 'loc_lat_err',
            'phys_x', 'phys_y', 'phys_yaw',
            'phys_abs_err', 'phys_lat_err',
            'nearest_loc_x', 'nearest_loc_y',
            'nearest_phys_x', 'nearest_phys_y',
            'plan_seq'
        ])

        self.plan_file = open(self.plan_csv, 'w', newline='')
        self.plan_writer = csv.writer(self.plan_file)
        self.plan_writer.writerow([
            'plan_seq', 'plan_time', 'x', 'y'
        ])

        self.get_logger().info(
            f'CSV 保存目录: {self.out_dir}')

    # ------------------------------------------------------------
    # 路径回调
    # ------------------------------------------------------------
    def plan_callback(self, msg: Path):
        self.plan = msg
        self.plan_points = []
        self.plan_seq += 1

        stamp = msg.header.stamp
        stamp_sec = stamp.sec + stamp.nanosec * 1e-9

        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.plan_points.append([x, y])
            self.plan_writer.writerow([
                self.plan_seq, stamp_sec, x, y
            ])
        self.plan_file.flush()

        # 自动保存第一次的 odom -> map 基准
        if self.odom_to_map_ref is None:
            try:
                self.odom_to_map_ref = self.tf_buffer.lookup_transform(
                    'map', 'odom', rclpy.time.Time())
                self.get_logger().info('已自动保存初始 odom -> map 基准')
            except Exception:
                self.get_logger().warn(
                    '暂时无法获取 odom -> map TF，将在后续继续尝试')

        # 可选：转发当前规划路径
        self.plan_path_pub.publish(msg)

        # 第一条规划路径单独发布，方便在 RViz 里看到最初完整路径
        if not self.first_plan_published:
            self.initial_plan_path_pub.publish(msg)
            self.first_plan_published = True

    # ------------------------------------------------------------
    # 定时记录
    # ------------------------------------------------------------
    def timer_callback(self):
        if self.plan is None or len(self.plan_points) < 1:
            return

        # 1. 定位位姿：map -> base_link
        try:
            trans_loc = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception:
            return

        loc_x = trans_loc.transform.translation.x
        loc_y = trans_loc.transform.translation.y
        loc_yaw = self.yaw_from_quat(trans_loc.transform.rotation)

        # 2. 物理位姿：odom -> base_link
        try:
            trans_odom = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time())
        except Exception:
            return

        odom_x = trans_odom.transform.translation.x
        odom_y = trans_odom.transform.translation.y
        odom_yaw = self.yaw_from_quat(trans_odom.transform.rotation)

        # 3. 如果还没拿到 odom->map 基准，继续尝试
        if self.odom_to_map_ref is None:
            try:
                self.odom_to_map_ref = self.tf_buffer.lookup_transform(
                    'map', 'odom', rclpy.time.Time())
                self.get_logger().info('已自动保存初始 odom -> map 基准')
            except Exception:
                pass

        # 4. 用固定 odom->map 把物理轨迹转到 map
        phys_x, phys_y, phys_yaw = odom_x, odom_y, odom_yaw
        phys_orientation = trans_odom.transform.rotation
        if self.odom_to_map_ref is not None:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = odom_x
            pose.pose.position.y = odom_y
            pose.pose.orientation = trans_odom.transform.rotation

            phys = do_transform_pose_stamped(pose, self.odom_to_map_ref)
            phys_x = phys.pose.position.x
            phys_y = phys.pose.position.y
            phys_orientation = phys.pose.orientation
            phys_yaw = self.yaw_from_quat(phys.pose.orientation)

        # 4. 计算两种偏差
        loc_n = self.nearest_on_path(loc_x, loc_y, self.plan_points)
        phys_n = self.nearest_on_path(phys_x, phys_y, self.plan_points)

        loc_abs_err = math.hypot(loc_x - loc_n[0], loc_y - loc_n[1])
        phys_abs_err = math.hypot(phys_x - phys_n[0], phys_y - phys_n[1])

        # 5. 追加实际路径
        self.append_pose(self.loc_path, loc_x, loc_y, trans_loc.transform.rotation)
        self.append_pose(self.phys_path, phys_x, phys_y, phys_orientation)

        # 6. 写 CSV
        now = self.get_clock().now()
        self.dev_writer.writerow([
            now.to_msg().sec + now.to_msg().nanosec * 1e-9,
            loc_x, loc_y, loc_yaw,
            loc_abs_err, loc_n[3],
            phys_x, phys_y, phys_yaw,
            phys_abs_err, phys_n[3],
            loc_n[0], loc_n[1],
            phys_n[0], phys_n[1],
            self.plan_seq
        ])
        self.dev_file.flush()

        # 7. 实时发布（可选项）
        self.loc_path_pub.publish(self.loc_path)
        self.phys_path_pub.publish(self.phys_path)
        self.loc_error_pub.publish(Float32(data=loc_abs_err))
        self.phys_error_pub.publish(Float32(data=phys_abs_err))
        self.publish_error_markers(loc_x, loc_y, loc_n, phys_x, phys_y, phys_n)

    # ------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------
    def append_pose(self, path_msg: Path, x: float, y: float, quaternion):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation = quaternion
        path_msg.poses.append(pose)
        path_msg.header.stamp = pose.header.stamp

    @staticmethod
    def nearest_on_path(x, y, points):
        if len(points) == 1:
            return points[0][0], points[0][1], 0, 0.0

        best_dist = float('inf')
        best_point = (points[0][0], points[0][1])
        best_yaw = 0.0

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            vx = x2 - x1
            vy = y2 - y1
            wx = x - x1
            wy = y - y1
            seg_len2 = vx * vx + vy * vy

            if seg_len2 < 1e-9:
                t = 0.0
            else:
                t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))

            px = x1 + t * vx
            py = y1 + t * vy
            d = math.hypot(x - px, y - py)

            if d < best_dist:
                best_dist = d
                best_point = (px, py)
                best_yaw = math.atan2(vy, vx)

        ux = math.cos(best_yaw)
        uy = math.sin(best_yaw)
        # 左侧法向量
        nx = -uy
        ny = ux
        lat_err = (x - best_point[0]) * nx + (y - best_point[1]) * ny

        return best_point[0], best_point[1], 0, lat_err

    @staticmethod
    def yaw_from_quat(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def quat_from_yaw(yaw):
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def publish_error_markers(self, loc_x, loc_y, loc_n,
                              phys_x, phys_y, phys_n):
        marker_array = MarkerArray()

        loc_marker = Marker()
        loc_marker.header.frame_id = 'map'
        loc_marker.header.stamp = self.get_clock().now().to_msg()
        loc_marker.ns = 'loc_errors'
        loc_marker.id = 0
        loc_marker.type = Marker.LINE_LIST
        loc_marker.action = Marker.ADD
        loc_marker.scale.x = 0.03
        loc_marker.color.r = 1.0
        loc_marker.color.g = 0.0
        loc_marker.color.b = 0.0
        loc_marker.color.a = 1.0
        loc_marker.pose.orientation.w = 1.0
        from geometry_msgs.msg import Point
        p1 = Point(x=loc_x, y=loc_y)
        p2 = Point(x=loc_n[0], y=loc_n[1])
        loc_marker.points = [p1, p2]
        marker_array.markers.append(loc_marker)

        phys_marker = Marker()
        phys_marker.header.frame_id = 'map'
        phys_marker.header.stamp = self.get_clock().now().to_msg()
        phys_marker.ns = 'phys_errors'
        phys_marker.id = 1
        phys_marker.type = Marker.LINE_LIST
        phys_marker.action = Marker.ADD
        phys_marker.scale.x = 0.03
        phys_marker.color.r = 0.0
        phys_marker.color.g = 0.0
        phys_marker.color.b = 1.0
        phys_marker.color.a = 1.0
        phys_marker.pose.orientation.w = 1.0
        p3 = Point(x=phys_x, y=phys_y)
        p4 = Point(x=phys_n[0], y=phys_n[1])
        phys_marker.points = [p3, p4]
        marker_array.markers.append(phys_marker)

        self.marker_pub.publish(marker_array)

    def destroy_node(self):
        self.dev_file.close()
        self.plan_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathDeviationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
