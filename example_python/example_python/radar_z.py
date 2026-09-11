#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class FlattenRadar(Node):
    def __init__(self):
        super().__init__('flatten_radar')
        self.sub = self.create_subscription(PointCloud2, '/radarpointcloud', self.cb, 10)
        self.pub = self.create_publisher(PointCloud2, '/radarpointcloud_flat', 10)
        self.get_logger().info('雷达点云压平节点已启动')

    def cb(self, msg):
        points = list(pc2.read_points(msg, skip_nans=True))
        if not points:
            return
        # 把所有点的 Z 强制设为 0.0
        flat_points = [(p[0], p[1], 0.0) + tuple(p[3:]) for p in points]
        out = pc2.create_cloud(msg.header, msg.fields, flat_points)
        self.pub.publish(out)

def main():
    rclpy.init()
    node = FlattenRadar()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

