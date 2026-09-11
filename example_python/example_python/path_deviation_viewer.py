#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径偏差离线可视化节点
- 读取 path_deviation_recorder.py 保存的 CSV
- 发布：
    /deviation/actual_path_loc     定位轨迹（AMCL）
    /deviation/actual_path_phys    物理轨迹（基于初始 odom->map 对齐）
    /deviation/planned_path        规划路径，默认为第一条/plan
    /deviation/error_markers       偏差线
- 打开 RViz 添加上述话题即可查看
"""

import argparse
import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray


class PathDeviationViewer(Node):
    def __init__(self, deviation_csv, plan_csv, rate, plan_mode='first', plan_merge_distance=0.30):
        super().__init__('path_deviation_viewer')

        self.rate = rate
        self.plan_mode = plan_mode
        self.plan_merge_distance = plan_merge_distance

        # 发布 QoS 使用 TRANSIENT_LOCAL，后打开 RViz 也能收到当前已发布的数据
        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.loc_path_pub = self.create_publisher(
            Path, '/deviation/actual_path_loc', qos)
        self.phys_path_pub = self.create_publisher(
            Path, '/deviation/actual_path_phys', qos)
        self.plan_path_pub = self.create_publisher(
            Path, '/deviation/planned_path', qos)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/deviation/error_markers', qos)
        self.plan_segment_pub = self.create_publisher(
            MarkerArray, '/deviation/planned_path_segments', qos)

        # 加载数据
        self.dev_rows = self.load_deviation(deviation_csv)
        self.plan_points = self.load_plan(plan_csv, plan_mode=plan_mode, merge_distance=plan_merge_distance)
        self.plan_segments = self.load_plan_segments(plan_csv)

        if not self.dev_rows:
            self.get_logger().error(f'没有读取到有效偏差数据: {deviation_csv}')
            raise SystemExit(1)

        # 构建可视化消息
        self.loc_path = self.build_path_from_rows(
            self.dev_rows, 'loc_x', 'loc_y')
        self.phys_path = self.build_path_from_rows(
            self.dev_rows, 'phys_x', 'phys_y')
        self.plan_path = self.build_plan_path()
        self.markers = self.build_markers()

        # 打印统计信息
        self.print_statistics()

        # 周期重发，方便任务结束后打开 RViz 也能看到
        self.timer = self.create_timer(1.0 / rate, self.publish_all)

    # ------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------
    @staticmethod
    def load_deviation(path):
        rows = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        'loc_x': float(row['loc_x']),
                        'loc_y': float(row['loc_y']),
                        'loc_yaw': float(row['loc_yaw']),
                        'loc_abs_err': float(row['loc_abs_err']),
                        'loc_lat_err': float(row['loc_lat_err']),
                        'phys_x': float(row['phys_x']),
                        'phys_y': float(row['phys_y']),
                        'phys_yaw': float(row['phys_yaw']),
                        'phys_abs_err': float(row['phys_abs_err']),
                        'phys_lat_err': float(row['phys_lat_err']),
                        'nearest_loc_x': float(row['nearest_loc_x']),
                        'nearest_loc_y': float(row['nearest_loc_y']),
                        'nearest_phys_x': float(row['nearest_phys_x']),
                        'nearest_phys_y': float(row['nearest_phys_y']),
                    })
                except (KeyError, ValueError):
                    continue
        return rows

    @staticmethod
    def load_plan(path, plan_mode='first', merge_distance=0.30):
        """读取 plan.csv，并返回规划路径点列表

        plan_mode:
            first: 第一次收到的 /plan，通常是第一段完整规划路径
            last: 最后一次收到的 /plan
            all: 把所有历史 /plan 按时间顺序拼接，并合并过近点
            segments: 返回所有历史 /plan 的拼接结果，同时保留分段信息
        """
        plan_dict = {}
        if not os.path.exists(path):
            return []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = int(float(row['plan_seq']))
                plan_dict.setdefault(seq, []).append([
                    float(row['x']),
                    float(row['y'])
                ])

        if not plan_dict:
            return []

        if plan_mode == 'last':
            return plan_dict[max(plan_dict.keys())]
        elif plan_mode in ('all', 'segments'):
            merged = []
            for seq in sorted(plan_dict.keys()):
                for x, y in plan_dict[seq]:
                    if not merged:
                        merged.append([x, y])
                        continue

                    last_x, last_y = merged[-1]
                    if math.hypot(x - last_x, y - last_y) >= merge_distance:
                        merged.append([x, y])

            return merged
        else:
            # default: first
            return plan_dict[min(plan_dict.keys())]

    @staticmethod
    def load_plan_segments(path):
        """读取 plan.csv，返回 {plan_seq: [[x,y], ...]}"""
        plan_dict = {}
        if not os.path.exists(path):
            return {}

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = int(float(row['plan_seq']))
                plan_dict.setdefault(seq, []).append([
                    float(row['x']),
                    float(row['y'])
                ])
        return plan_dict

    # ------------------------------------------------------------
    # 构建消息
    # ------------------------------------------------------------
    @staticmethod
    def build_path_from_rows(rows, x_key, y_key):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rclpy.time.Time().to_msg()

        for row in rows:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = row[x_key]
            pose.pose.position.y = row[y_key]
            # 只显示平面路径，z 置 0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        if path.poses:
            path.header.stamp = rclpy.time.Time().to_msg()
        return path

    def build_plan_path(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = rclpy.time.Time().to_msg()

        for x, y in self.plan_points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        if path.poses:
            path.header.stamp = rclpy.time.Time().to_msg()
        return path

    def build_markers(self):
        marker_array = MarkerArray()

        # 定位偏差线：红色
        loc_marker = self.base_marker('loc_errors', 0, (1.0, 0.0, 0.0))
        # 物理偏差线：蓝色
        phys_marker = self.base_marker('phys_errors', 1, (0.0, 0.0, 1.0))

        for row in self.dev_rows:
            loc_marker.points.append(Point(
                x=row['loc_x'], y=row['loc_y']))
            loc_marker.points.append(Point(
                x=row['nearest_loc_x'], y=row['nearest_loc_y']))

            phys_marker.points.append(Point(
                x=row['phys_x'], y=row['phys_y']))
            phys_marker.points.append(Point(
                x=row['nearest_phys_x'], y=row['nearest_phys_y']))

        marker_array.markers.append(loc_marker)
        marker_array.markers.append(phys_marker)
        marker_array.markers.extend(self.build_plan_segment_markers())
        return marker_array

    def build_plan_segment_markers(self):
        markers = []
        for marker_id, (seq, pts) in enumerate(sorted(self.plan_segments.items())):
            if len(pts) < 2:
                continue

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rclpy.time.Time().to_msg()
            marker.ns = 'planned_path_segments'
            marker.id = 1000 + marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.015

            # 每个规划版本颜色略有不同，避免完全重叠看不清
            hue = 0.15 + (marker_id % 10) * 0.07
            rgb = self.hsv_like_to_rgb(hue)
            marker.color.r = rgb[0]
            marker.color.g = rgb[1]
            marker.color.b = rgb[2]
            marker.color.a = 0.65
            marker.pose.orientation.w = 1.0

            for x, y in pts:
                marker.points.append(Point(x=x, y=y, z=0.0))

            markers.append(marker)
        return markers

    @staticmethod
    def hsv_like_to_rgb(h):
        h = h % 1.0
        i = int(h * 6.0)
        f = h * 6.0 - i
        q = 1.0 - f
        t = 1.0 - (1.0 - f)

        if i % 6 == 0:
            return 1.0, t, 0.0
        elif i % 6 == 1:
            return q, 1.0, 0.0
        elif i % 6 == 2:
            return 0.0, 1.0, t
        elif i % 6 == 3:
            return 0.0, q, 1.0
        elif i % 6 == 4:
            return t, 0.0, 1.0
        else:
            return 1.0, 0.0, q

    @staticmethod
    def base_marker(ns, id_, rgb):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.ns = ns
        marker.id = id_
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        marker.points = []
        return marker

    # ------------------------------------------------------------
    # 统计与发布
    # ------------------------------------------------------------
    def print_statistics(self):
        loc_abs = [r['loc_abs_err'] for r in self.dev_rows]
        phys_abs = [r['phys_abs_err'] for r in self.dev_rows]
        loc_lat = [r['loc_lat_err'] for r in self.dev_rows]
        phys_lat = [r['phys_lat_err'] for r in self.dev_rows]

        self.get_logger().info('========== 路径偏差统计 ==========')
        self.get_logger().info(
            f'定位偏差 loc_abs: mean={self.mean(loc_abs):.3f} m, '
            f'max={max(loc_abs):.3f} m')
        #self.get_logger().info(
            #f'物理偏差 phys_abs: mean={self.mean(phys_abs):.3f} m, '
            #f'max={max(phys_abs):.3f} m')
        self.get_logger().info(
            f'定位横向偏差 loc_lat: mean={self.mean(loc_lat):.3f} m')
        #self.get_logger().info(
            #f'物理横向偏差 phys_lat: mean={self.mean(phys_lat):.3f} m')
        #self.get_logger().info('===================================')

    @staticmethod
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    def publish_all(self):
        self.loc_path_pub.publish(self.loc_path)
        self.phys_path_pub.publish(self.phys_path)
        self.plan_path_pub.publish(self.plan_path)
        self.marker_pub.publish(self.markers)
        self.plan_segment_pub.publish(self.markers)

        now = self.get_clock().now().to_msg()
        self.loc_path.header.stamp = now
        self.phys_path.header.stamp = now
        self.plan_path.header.stamp = now
        for marker in self.markers.markers:
            marker.header.stamp = now


def parse_args():
    parser = argparse.ArgumentParser(description='离线查看路径偏差')
    parser.add_argument(
        '--deviation',
        default=None,
        help='deviation.csv 路径，默认找 ~/path_deviation_results 下最新一次'
    )
    parser.add_argument(
        '--plan',
        default=None,
        help='plan.csv 路径，默认根据 --deviation 自动推断'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=1.0,
        help='重新发布频率，默认 1Hz'
    )
    parser.add_argument(
        '--plan-mode',
        choices=['first', 'last', 'all', 'segments'],
        default='segments',
        help='first: 初始路径；last: 最后一条路径；all: 全部历史路径拼接；segments: 全部历史路径按版本分段显示。默认 segments'
    )
    parser.add_argument(
        '--plan-merge-distance',
        type=float,
        default=0.30,
        help='all 模式下合并相邻规划点的距离阈值，默认 0.30 m'
    )
    return parser.parse_args()


def find_latest_csv():
    base = os.path.expanduser('~/path_deviation_results')
    if not os.path.isdir(base):
        return None, None

    dirs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    if not dirs:
        return None, None

    latest = max(dirs, key=os.path.getmtime)
    dev = os.path.join(latest, 'deviation.csv')
    plan = os.path.join(latest, 'plan.csv')
    if not os.path.exists(dev):
        return None, None
    return dev, plan


def main(args=None):
    args = parse_args()
    rclpy.init()

    deviation_csv = args.deviation
    plan_csv = args.plan

    if deviation_csv is None:
        deviation_csv, plan_csv = find_latest_csv()
        if deviation_csv is None:
            print('未找到历史 CSV，请通过 --deviation 指定 deviation.csv 路径')
            rclpy.shutdown()
            return

    if plan_csv is None and deviation_csv is not None:
        plan_csv = os.path.join(
            os.path.dirname(deviation_csv), 'plan.csv')

    node = PathDeviationViewer(
        deviation_csv, plan_csv, args.rate,
        args.plan_mode, args.plan_merge_distance)

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
