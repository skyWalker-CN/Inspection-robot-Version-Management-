#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class VoxelDownsampler(Node):
    def __init__(self):
        super().__init__('voxel_downsampler')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(PointCloud2, '/rslidar_points', self.callback, qos)
        self.pub = self.create_publisher(PointCloud2, '/rslidar_points_filtered', qos)
        self.leaf_size = 0.03   # 5cm 体素

    def callback(self, msg):
        if msg.width == 0 or msg.height == 0:
            return

        # 提取 x, y, z 并直接构建普通 numpy 数组
        points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])   # 显式取出三个坐标值
        if len(points) == 0:
            return

        arr = np.array(points, dtype=np.float32)  # 此时形状 (N,3)

        # 体素降采样
        voxel_indices = np.floor(arr / self.leaf_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        filtered = arr[unique_indices]

        # 生成新点云，仅包含 xyz
        new_msg = pc2.create_cloud_xyz32(msg.header, filtered.tolist())
        self.pub.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoxelDownsampler()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
