#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from builtin_interfaces.msg import Time
import time

class InitialPoseSender(Node):
    def __init__(self):
        super().__init__('initial_pose_sender')
        self.pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        # 延迟 3 秒等待 TF 稳定，然后发送
        self.timer = self.create_timer(3.0, self.send_pose)

    def send_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        # 使用当前时间减去 2 秒，让它一定在 TF 缓存窗口内
        now = self.get_clock().now()
        earlier = Time(sec=now.seconds_nanoseconds()[0] - 2,
                       nanosec=now.seconds_nanoseconds()[1])
        msg.header.stamp = earlier

        # 设置你的初始位姿（根据实际情况修改坐标）
        msg.pose.pose.position = Point(x=9.16, y=0.34, z=0.0)
        # 欧拉角 yaw = -0.044 rad，转为四元数
        import math
        yaw = -0.044
        msg.pose.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=math.sin(yaw/2), w=math.cos(yaw/2))
        # 协方差随意
        msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0685]

        self.pub.publish(msg)
        self.get_logger().info('Initial pose sent with past timestamp')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = InitialPoseSender()
    rclpy.spin_once(node, timeout_sec=5)
    node.destroy_node()

if __name__ == '__main__':
    main()
