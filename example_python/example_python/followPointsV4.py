#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import threading
import time
import math
import copy

class AsyncNavigator(BasicNavigator):
    def __init__(self):
        super().__init__()
        self.task_in_progress = False
        self.loop_times = 2
        self.task_lock = threading.Lock()
        self.path_queue = []

        self.chat_pub = self.create_publisher(String, '/chat_message', 10)

        # 订阅里程计
        self.current_odom = None
        self.odom_lock = threading.Lock()
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10
        )

        self.waitUntilNav2Active()
        self.origin_pose = self._get_origin_pose_with_fallback()
        self.get_logger().info(f"原点已记录: ({self.origin_pose.pose.position.x:.2f}, "
                               f"{self.origin_pose.pose.position.y:.2f})")

    def _odom_callback(self, msg):
        with self.odom_lock:
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
                'map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=2.0)
            )
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            self.get_logger().info("通过 TF 获取原点成功")
            return pose
        except Exception as e:
            self.get_logger().warn(f"TF 获取原点失败: {e}，尝试使用里程计")

        start = time.time()
        while self.current_odom is None and time.time() - start < 5.0:
            time.sleep(0.1)
        if self.current_odom is not None:
            pose.pose.position.x = self.current_odom.position.x
            pose.pose.position.y = self.current_odom.position.y
            pose.pose.orientation = self.current_odom.orientation
            pose.header.frame_id = 'odom'
            self.get_logger().info("通过里程计获取原点成功")
            return pose

        self.get_logger().warn("完全无法获取位姿，原点设为(0,0)")
        pose.pose.orientation.w = 1.0
        return pose

    def set_loop_times(self, times):
        self.loop_times = max(1, times)

    def follow_waypoints_async(self, waypoints):
        """启动路径跟踪或加入队列（非阻塞）"""
        with self.task_lock:
            if self.task_in_progress:
                self.path_queue.extend(waypoints)
                self.get_logger().info(f"已有任务进行中，缓存路径点，队列长度: {len(self.path_queue)}")
                return
            # 无任务在跑，立即启动，并清空旧队列（防止残留）
            self.task_in_progress = True
            self.path_queue.clear()

        self.get_logger().info("开始新导航任务")
        threading.Thread(
            target=self._follow_waypoints_thread,
            args=(waypoints,),
            daemon=True
        ).start()

    def _wait_for_task_with_stuck_detection(self):
        """
        等待当前导航目标完成，同时基于里程计累计位移检测堵塞。
        若 5 秒内累计位移 < 0.05 m，则通过 /chat_message 发送语音提示。
        """
        stuck_start_time = None
        stuck_start_pose = None
        warn_printed = False   # 避免里程计缺失时日志刷屏

        while not self.isTaskComplete():
            with self.odom_lock:
                cur_pose = self.current_odom
            if cur_pose is None:
                if not warn_printed:
                    self.get_logger().warn("里程计数据不可用，堵塞检测暂不可用")
                    warn_printed = True
                time.sleep(0.2)
                continue

            warn_printed = False

            if stuck_start_time is None:
                stuck_start_time = time.time()
                stuck_start_pose = cur_pose
            else:
                dx = cur_pose.position.x - stuck_start_pose.position.x
                dy = cur_pose.position.y - stuck_start_pose.position.y
                dist = math.hypot(dx, dy)

                if dist < 0.5:   # 注意这里是 0.05，不是 0.5
                    if time.time() - stuck_start_time > 5.0:
                        msg = String()
                        msg.data = "你好，请让一下，你好，请让一下"
                        self.chat_pub.publish(msg)
                        self.get_logger().info("检测到堵塞，已发送语音提示")
                        # 重置起点，以便持续堵塞时每5秒发送一次
                        stuck_start_time = time.time()
                        stuck_start_pose = cur_pose
                else:
                    # 机器人确实移动了，清除堵塞状态
                    stuck_start_time = None
                    stuck_start_pose = None

            time.sleep(0.2)

    def _follow_waypoints_thread(self, waypoints):
        """核心导航线程，支持连续执行队列中的多条路径"""
        try:
            # 深拷贝，防止外部修改
            current_waypoints = copy.deepcopy(waypoints)
            last_result = None

            while current_waypoints:
                success_count = 0
                failed_count = 0
                total = len(current_waypoints)

                for idx, pose in enumerate(current_waypoints):
                    goal = PoseStamped()
                    goal.header.frame_id = pose.header.frame_id
                    goal.header.stamp = self.get_clock().now().to_msg()
                    goal.pose = pose.pose

                    self.get_logger().info(f"前往路径点 {idx+1}/{total}")
                    self.goToPose(goal)

                    self._wait_for_task_with_stuck_detection()

                    result = self.getResult()
                    last_result = result

                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info(f"路径点 {idx+1} 到达成功")
                        success_count += 1
                    else:
                        self.get_logger().warn(f"路径点 {idx+1} 导航失败 ({result})")
                        failed_count += 1
                        time.sleep(1.0)

                self.get_logger().info(f"本轮完成：成功 {success_count}，失败 {failed_count}")

                # 取出队列中的下一条路径（如果有）
                with self.task_lock:
                    if self.path_queue:
                        current_waypoints = self.path_queue.copy()
                        self.path_queue.clear()
                        self.get_logger().info(f"从队列中取出新路径，共 {len(current_waypoints)} 个点")
                    else:
                        current_waypoints = None

                # 所有路径（含队列）执行完毕，若最后失败则返回原点
                if current_waypoints is None:
                    if last_result is not None and last_result == TaskResult.FAILED:
                        self.get_logger().warn("末尾目标失败，正在返回原点")
                        self._return_to_origin()

        except Exception as e:
            self.get_logger().error(f"导航线程异常: {e}")
        finally:
            with self.task_lock:
                self.task_in_progress = False
            self.get_logger().info("所有路径点执行完毕，等待下一条路径消息...")

    def _return_to_origin(self):
        self.get_logger().info("正在返回原点...")
        goal = PoseStamped()
        goal.header.frame_id = self.origin_pose.header.frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose = self.origin_pose.pose

        self.goToPose(goal)
        self._wait_for_task_with_stuck_detection()

        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("成功返回原点")
        else:
            self.get_logger().error(f"返回原点失败: {result}")

def main():
    rclpy.init()
    navigator = AsyncNavigator()
    navigator.set_loop_times(1)   # 可根据需要修改

    def path_callback(msg):
        navigator.get_logger().info(f"收到路径消息，包含 {len(msg.poses)} 个点")
        repeated = []
        for _ in range(navigator.loop_times):
            repeated.extend(msg.poses)
        navigator.get_logger().info(f"扩展为 {len(repeated)} 个点，准备执行")
        navigator.follow_waypoints_async(repeated)

    subscription = navigator.create_subscription(
        Path, 'my_path', path_callback, 10
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(navigator)

    try:
        navigator.get_logger().info("节点已启动，等待 my_path 消息...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        navigator.get_logger().error(f"执行器异常: {e}")
    finally:
        navigator.destroyNode()
        rclpy.shutdown()
        print("节点已关闭")

if __name__ == '__main__':
    main()
