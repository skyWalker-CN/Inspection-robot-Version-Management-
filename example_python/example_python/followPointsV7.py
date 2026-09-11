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
import json
import os

class AsyncNavigator(Node):
    def __init__(self):
        super().__init__('async_navigator')
        self.loop_times = 2                     # 循环次数（新路径时使用）
        self.path_queue = []                    # 每条路径是一个点列表
        self.current_waypoints = []             # 当前正在执行的路径点
        self.current_idx = 0                    # 当前正在执行/已发送的点索引
        self.goal_handle = None
        self.navigating = False
        self.last_result = None                 # 当前路径最后一个点的结果

        # ---------- 状态监控与中断恢复 ----------
        self.robot_status = "0"                 # 默认正常
        self.task_interrupted = False           # 是否有未完成任务（实时内存标志）
        self.last_path_poses = []               # 用户原始路径点（深拷贝）
        self.last_path_progress = 0             # 恢复起点：下一个要执行的点在 last_path_poses 中的索引
        self.cancel_requested = False
        self._first_status_received = False

        # 持久化文件路径
        self.save_file = os.path.expanduser('~/.nav_recovery.json')

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

        # 订阅路径和状态
        self.path_sub = self.create_subscription(
            Path, 'my_path', self._path_callback, 10)
        self.status_sub = self.create_subscription(
            String, '/robot_status', self._status_callback, 10)

        # 堵塞检测定时器（每秒一次）
        self.stuck_timer = self.create_timer(1.0, self._stuck_check)
        self.stuck_start_time = None
        self.stuck_start_pose = None

        # ---------- 改进：延迟重复发布初始位姿，确保 AMCL 收到 ----------
        self._init_pose_count = 0
        self.initial_pose_timer = self.create_timer(1.0, self._try_send_initial_pose)

        # ---------- 启动时加载持久化任务 ----------
        self._load_recovery_state()

        self.get_logger().info("节点初始化完成，等待 my_path 消息...")

    # ---------- 基础方法 ----------
    def _wait_for_nav2(self):
        self.get_logger().info("等待 navigate_to_pose action server...")
        while not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().info("action server 未就绪，继续等待...")

    def _odom_callback(self, msg):
        self.current_odom = msg.pose.pose

    def _get_origin_pose_with_fallback(self):
        """返回固定的初始位姿"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = -0.5809604525566101
        pose.pose.position.y = -1.418220043182373
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.10710896326966343
        pose.pose.orientation.w = 0.9942472881468161
        self.get_logger().info("使用固定初始位姿")
        return pose

    def _publish_initial_pose(self, pose_stamped):
        """发布一次初始位姿到 /initialpose"""
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.frame_id = 'map'
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.pose.pose = pose_stamped.pose
        init_msg.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467
        ]
        self.initial_pose_pub.publish(init_msg)
        self.get_logger().info(f"已发布初始位姿: ({pose_stamped.pose.position.x:.2f}, "
                               f"{pose_stamped.pose.position.y:.2f})")

    def _try_send_initial_pose(self):
        """定时尝试发送初始位姿，发送3次后停止"""
        if self._init_pose_count >= 3:
            self.destroy_timer(self.initial_pose_timer)
            return
        self._publish_initial_pose(self.origin_pose)
        self._init_pose_count += 1
        self.get_logger().info(f"重发初始位姿 {self._init_pose_count}/3")

    def set_loop_times(self, times):
        self.loop_times = max(1, times)

    # ---------- 持久化 ----------
    def _save_recovery_state(self):
        """将最新路径、进度、未完成标志写入 JSON 文件"""
        data = {
            'task_interrupted': self.task_interrupted,
            'progress': self.last_path_progress,
            'path': []
        }
        for pose in self.last_path_poses:
            p = pose.pose.position
            o = pose.pose.orientation
            data['path'].append({
                'x': p.x, 'y': p.y, 'z': p.z,
                'qx': o.x, 'qy': o.y, 'qz': o.z, 'qw': o.w,
                'frame_id': pose.header.frame_id
            })
        try:
            with open(self.save_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.get_logger().error(f"保存恢复状态失败: {e}")

    def _load_recovery_state(self):
        """启动时从文件恢复未完成的任务及进度"""
        if not os.path.exists(self.save_file):
            return
        try:
            with open(self.save_file, 'r') as f:
                data = json.load(f)
            if not data.get('task_interrupted', False):
                return
            self.get_logger().info("发现未完成的导航任务，准备恢复")
            # 重建路径
            self.last_path_poses = []
            for wp in data.get('path', []):
                pose = PoseStamped()
                pose.header.frame_id = wp.get('frame_id', 'map')
                pose.pose.position.x = wp['x']
                pose.pose.position.y = wp['y']
                pose.pose.position.z = wp.get('z', 0.0)
                pose.pose.orientation.x = wp.get('qx', 0.0)
                pose.pose.orientation.y = wp.get('qy', 0.0)
                pose.pose.orientation.z = wp.get('qz', 0.0)
                pose.pose.orientation.w = wp.get('qw', 1.0)
                self.last_path_poses.append(pose)
            # 恢复进度，若无效则从0开始
            saved_progress = data.get('progress', 0)
            if 0 <= saved_progress <= len(self.last_path_poses):
                self.last_path_progress = saved_progress
            else:
                self.last_path_progress = 0
            self.task_interrupted = True
        except Exception as e:
            self.get_logger().error(f"加载恢复状态失败: {e}")
            self.task_interrupted = False
            self.last_path_poses = []
            self.last_path_progress = 0

    # ---------- 路径消息回调 ----------
    def _path_callback(self, msg):
        self.get_logger().info(f"🚀 收到路径消息，包含 {len(msg.poses)} 个点")
        self.last_path_poses = copy.deepcopy(msg.poses)
        self.last_path_progress = 0          # 新路径从起点开始
        self.task_interrupted = True         # 新任务标记为未完成
        self._save_recovery_state()

        # 如果当前状态正常，清除实时中断标志，准备执行
        if self.robot_status == "0":
            self.task_interrupted = False

        # 构建重复路径点（用于执行队列）
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
        self.last_result = None
        self.get_logger().info(f"开始执行新路径，共 {len(self.current_waypoints)} 个点")
        self._navigate_next()

    def _navigate_next(self):
        if self.current_idx >= len(self.current_waypoints):
            # 所有点处理完毕
            self.get_logger().info("本条路径所有点处理完毕")
            if self._is_origin_path():
                self.get_logger().info("原点路径执行完毕，停止导航")
                self._finish_navigation()
                return
            # 用户路径：最后一个点失败且无排队路径 → 返回原点
            if self.last_result is not None and not self.last_result and not self.path_queue:
                self.get_logger().warn("最后一个路径点失败，尝试返回原点")
                self._return_to_origin()
                return
            # 否则继续下一条路径
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

    def _is_origin_path(self):
        if len(self.current_waypoints) == 1:
            op = self.origin_pose.pose
            cp = self.current_waypoints[0].pose
            return (abs(cp.position.x - op.position.x) < 0.001 and
                    abs(cp.position.y - op.position.y) < 0.001)
        return False

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("目标被服务器拒绝，跳过该点")
            self.current_idx += 1
            self._navigate_next()
            return
        self.goal_handle = goal_handle
        self.stuck_start_time = None
        self.stuck_start_pose = None
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result()
        self.goal_handle = None

        # 主动取消（状态异常）
        if self.cancel_requested:
            self.cancel_requested = False
            self.get_logger().info("目标已取消（机器人状态异常），暂停导航")
            self.navigating = False
            self.current_waypoints = []
            self.task_interrupted = True
            # 进度保持当前点（未完成，下次重试该点）
            self._save_recovery_state()
            return

        success = (result.status == 4)
        if success:
            self.get_logger().info(f"路径点 {self.current_idx+1} 到达成功")
        else:
            self.get_logger().warn(f"路径点 {self.current_idx+1} 失败，状态码: {result.status}")

        # 更新进度：仅在执行用户路径且成功时推进
        if not self._is_origin_path() and success:
            # current_idx 即将自增，下一个点的索引是 current_idx+1
            self.last_path_progress = self.current_idx + 1

        # 用户路径最后一个点成功 → 任务完成，清除标志
        is_last_point_of_user_path = (
            not self._is_origin_path() and
            self.current_idx + 1 >= len(self.current_waypoints)
        )
        if success and is_last_point_of_user_path:
            self.get_logger().info("用户路径最后一个点成功到达，任务完成")
            self.task_interrupted = False
            self.last_path_progress = 0   # 任务完成，重置进度
            self._save_recovery_state()   # 标志变为 0

        self.last_result = success
        self.current_idx += 1
        self._navigate_next()

    # ---------- 返回原点 ----------
    def _return_to_origin(self):
        origin_waypoint = PoseStamped()
        origin_waypoint.header.frame_id = self.origin_pose.header.frame_id
        origin_waypoint.header.stamp = self.get_clock().now().to_msg()
        origin_waypoint.pose = self.origin_pose.pose
        self.path_queue.insert(0, [origin_waypoint])
        self.get_logger().info("已插入原点路径，准备返回")
        self._start_next_path()

    def _finish_navigation(self):
        self.navigating = False
        self.path_queue.clear()
        self.current_waypoints = []
        # 注意：这里不清除 task_interrupted 和进度，因为原点结束不代表任务完成
        self.get_logger().info("导航完全结束（原点到达），等待下一条路径消息...")

    # ---------- 机器人状态回调 ----------
    def _status_callback(self, msg):
        new_status = msg.data
        old_status = self.robot_status
        self.robot_status = new_status

        # 首次收到状态，若存在未完成任务且正常，立即恢复
        if not self._first_status_received:
            self._first_status_received = True
            if new_status == "0" and self.task_interrupted and self.last_path_poses:
                self.get_logger().info("首次收到正常状态，恢复中断的导航")
                self._restore_navigation()
                return

        if old_status == "0" and new_status == "1":
            self.get_logger().error("机器人状态异常！中断当前导航任务")
            self.task_interrupted = True
            self._save_recovery_state()
            if self.goal_handle is not None:
                self.cancel_requested = True
                # 修复：Humble 中方法名为 cancel_goal_async
                self.goal_handle.cancel_goal_async()
            else:
                if self.navigating:
                    self.navigating = False
                    self.current_waypoints = []
        elif old_status == "1" and new_status == "0":
            self.get_logger().info("机器人状态恢复正常")
            if self.task_interrupted and self.last_path_poses:
                self._restore_navigation()
            else:
                self.task_interrupted = False
                self._save_recovery_state()

    def _restore_navigation(self):
        """从中断点恢复用户路径"""
        if not self.last_path_poses:
            return
        start_idx = self.last_path_progress
        if start_idx >= len(self.last_path_poses):
            self.get_logger().warn("进度索引超出路径长度，重置为起点")
            start_idx = 0
            self.last_path_progress = 0
            self._save_recovery_state()
        remaining = self.last_path_poses[start_idx:]
        if not remaining:
            self.get_logger().info("没有剩余路径点，任务已完成？")
            self.task_interrupted = False
            self.last_path_progress = 0
            self._save_recovery_state()
            return
        self.get_logger().info(f"从中断点 {start_idx+1}/{len(self.last_path_poses)} 恢复，剩余 {len(remaining)} 个点")
        # 恢复时不再重复循环，直接走剩余点一次
        restored_path = copy.deepcopy(remaining)
        self.path_queue.insert(0, restored_path)
        self.task_interrupted = False
        self._save_recovery_state()
        self._start_next_path()

    # ---------- 堵塞检测 ----------
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
        if dist < 0.05:
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
