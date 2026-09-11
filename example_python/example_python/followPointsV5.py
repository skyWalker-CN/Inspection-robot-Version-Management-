#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from rclpy.node import Node
import math
import copy
import time

class AsyncNavigator(Node):
    def __init__(self):
        super().__init__('async_navigator')
        self.loop_times = 2
        self.path_queue = []                # 每条路径是一个点列表
        self.current_waypoints = []         # 当前正在执行的路径点
        self.current_idx = 0
        self.goal_handle = None
        self.navigating = False

        # 语音发布器
        self.chat_pub = self.create_publisher(String, '/chat_message', 10)

        # 初始位姿发布器（用于 AMCL 初始化）
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # 里程计订阅
        self.current_odom = None
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10)

        # 导航 action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 等待 Nav2 激活并记录原点
        self._wait_for_nav2()
        self.origin_pose = self._get_origin_pose_with_fallback()
        self.get_logger().info(f"原点已记录: ({self.origin_pose.pose.position.x:.2f}, "
                               f"{self.origin_pose.pose.position.y:.2f})")

        # 自动发布初始位姿给 AMCL
        self._publish_initial_pose(self.origin_pose)

        # 路径订阅
        self.path_sub = self.create_subscription(
            Path, 'my_path', self._path_callback, 10)
        self.get_logger().info("节点初始化完成，等待 my_path 消息...")

        # 堵塞检测定时器（每秒一次）
        self.stuck_timer = self.create_timer(1.0, self._stuck_check)
        self.stuck_start_time = None
        self.stuck_start_pose = None

    def _wait_for_nav2(self):
        self.get_logger().info("等待 navigate_to_pose action server...")
        while not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("action server 未就绪，继续等待...")

    def _odom_callback(self, msg):
        self.current_odom = msg.pose.pose

    def _get_origin_pose_with_fallback(self):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        from tf2_ros import Buffer, TransformListener
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, self)
        try:
            transform = tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=2.0))
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            self.get_logger().info("通过 TF 获取原点成功")
            return pose
        except Exception as e:
            self.get_logger().warn(f"TF 获取原点失败: {e}，尝试使用里程计")

        # 等待里程计数据 (此时主循环未启动，安全使用 spin_once)
        start_time = time.time()
        while self.current_odom is None and time.time() - start_time < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.current_odom is not None:
            pose.pose.position.x = self.current_odom.position.x
            pose.pose.position.y = self.current_odom.position.y
            pose.pose.position.z = self.current_odom.position.z
            pose.pose.orientation = self.current_odom.orientation
            pose.header.frame_id = 'odom'
            self.get_logger().info("通过里程计获取原点成功")
            return pose

        self.get_logger().warn("完全无法获取位姿，原点设为(0,0)")
        pose.pose.orientation.w = 1.0
        return pose

    def _publish_initial_pose(self, pose_stamped):
        """向 /initialpose 发布初始位姿，让 AMCL 初始化"""
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.frame_id = 'map'            # AMCL 需要 map 坐标系
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.pose.pose = pose_stamped.pose
        # 设置协方差（表示不确定性，可根据实际情况调整）
        init_msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.068
        ]
        self.initial_pose_pub.publish(init_msg)
        self.get_logger().info(f"已发布初始位姿: ({pose_stamped.pose.position.x:.2f}, "
                               f"{pose_stamped.pose.position.y:.2f})")

    def set_loop_times(self, times):
        self.loop_times = max(1, times)

    # ---------- 路径消息回调 ----------
    def _path_callback(self, msg):
        self.get_logger().info(f"🚀 收到路径消息，包含 {len(msg.poses)} 个点")
        repeated = []
        for _ in range(self.loop_times):
            repeated.extend(msg.poses)
        self.path_queue.append(copy.deepcopy(repeated))
        self.get_logger().info(f"已加入队列 (长度 {len(self.path_queue)})")
        if not self.navigating:
            self._start_next_path()

    def _start_next_path(self):
        if not self.path_queue:
            self.navigating = False
            self.get_logger().info("所有路径执行完毕，等待下一条消息...")
            return
        self.navigating = True
        self.current_waypoints = self.path_queue.pop(0)
        self.current_idx = 0
        self.get_logger().info(f"开始执行新路径，共 {len(self.current_waypoints)} 个点")
        self._navigate_next()

    def _navigate_next(self):
        if self.current_idx >= len(self.current_waypoints):
            self.get_logger().info("本条路径所有点处理完毕")
            self._start_next_path()
            return
        pose = self.current_waypoints[self.current_idx]
        goal = PoseStamped()
        goal.header.frame_id = pose.header.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = pose.pose

        self.get_logger().info(f"前往路径点 {self.current_idx+1}/{len(self.current_waypoints)}")
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("目标被服务器拒绝，跳过该点")
            self.current_idx += 1
            self._navigate_next()
            return
        self.goal_handle = goal_handle
        # 重置堵塞计时基准
        self.stuck_start_time = None
        self.stuck_start_pose = None
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result()
        self.goal_handle = None
        if result.status == 4:   # SUCCEEDED
            self.get_logger().info(f"路径点 {self.current_idx+1} 到达成功")
        else:
            self.get_logger().warn(f"路径点 {self.current_idx+1} 失败，状态码: {result.status}")
        self.current_idx += 1
        self._navigate_next()

    # ---------- 堵塞检测定时器 ----------
    def _stuck_check(self):
        if self.goal_handle is None:
            self.stuck_start_time = None
            return
        if self.current_odom is None:
            return
        cur_pose = self.current_odom
        now = self.get_clock().now().nanoseconds / 1e9
        if self.stuck_start_time is None or self.stuck_start_pose is None:
            self.stuck_start_time = now
            self.stuck_start_pose = cur_pose
            return
        dx = cur_pose.position.x - self.stuck_start_pose.position.x
        dy = cur_pose.position.y - self.stuck_start_pose.position.y
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            if now - self.stuck_start_time > 5.0:
                msg = String()
                msg.data = "你好，请让一下，你好，请让一下"
                self.chat_pub.publish(msg)
                self.get_logger().info("检测到堵塞，已发送语音提示")
                self.stuck_start_time = now
                self.stuck_start_pose = cur_pose
        else:
            self.stuck_start_time = None
            self.stuck_start_pose = None

def main():
    rclpy.init()
    navigator = AsyncNavigator()
    navigator.set_loop_times(1)   # 可按需修改
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
