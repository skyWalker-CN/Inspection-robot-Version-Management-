#!/usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import threading
import time

class AsyncNavigator(BasicNavigator):
    def __init__(self):
        super().__init__()
        self.task_in_progress = False
        self.loop_times = 2  # 默认循环次数
        self.task_lock = threading.Lock()
        self.path_queue = []  # 存储待执行的路径点队列

        # 等待导航激活并记录原点
        self.waitUntilNav2Active()
        self.origin_pose = self._get_current_pose()
        self.get_logger().info(f"原点已记录: ({self.origin_pose.pose.position.x:.2f}, "
                               f"{self.origin_pose.pose.position.y:.2f})")

    def _get_current_pose(self):
        """获取机器人当前位姿，返回 PoseStamped 格式"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        # 使用 TF 获取当前位姿（需要 tf2_ros 可用）
        from tf2_ros import Buffer, TransformListener
        tf_buffer = Buffer()
        tf_listener = TransformListener(tf_buffer, self)
        time.sleep(1.0)  # 给 TF 一点时间
        try:
            transform = tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
        except Exception as e:
            self.get_logger().warn(f"获取当前位姿失败，使用默认原点(0,0): {e}")
            pose.pose.position.x = 0.0
            pose.pose.position.y = 0.0
            pose.pose.orientation.w = 1.0
        return pose

    def set_loop_times(self, times):
        """设置路径循环次数"""
        self.loop_times = max(1, times)  # 确保至少执行1次

    def follow_waypoints_async(self, waypoints):
        """在后台线程中执行路径跟随"""
        with self.task_lock:
            if self.task_in_progress:
                # 将新路径加入队列等待执行
                self.path_queue.extend(waypoints)
                self.get_logger().info(f"任务进行中，已缓存路径点，当前队列长度: {len(self.path_queue)}")
                return
            self.task_in_progress = True

        # 启动后台线程
        threading.Thread(
            target=self._follow_waypoints_thread,
            args=(waypoints,),
            daemon=True
        ).start()

    def _follow_waypoints_thread(self, waypoints):
        """后台线程：逐个尝试路径点，失败则跳过，全失败则返回原点"""
        try:
            current_waypoints = waypoints
            while current_waypoints:
                success_count = 0
                failed_count = 0
                total_points = len(current_waypoints)

                for idx, pose in enumerate(current_waypoints):
                    # 构建带时间戳的导航目标
                    goal_pose = PoseStamped()
                    goal_pose.header.frame_id = pose.header.frame_id
                    goal_pose.header.stamp = self.get_clock().now().to_msg()
                    goal_pose.pose = pose.pose

                    self.get_logger().info(f"尝试导航至路径点 {idx+1}/{total_points}")
                    self.goToPose(goal_pose)

                    # 等待导航完成
                    while not self.isTaskComplete():
                        time.sleep(0.5)

                    result = self.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info(f"路径点 {idx+1} 到达成功")
                        success_count += 1
                    else:
                        self.get_logger().warn(f"路径点 {idx+1} 导航失败，原因: {result}，跳过该点")
                        failed_count += 1
                        time.sleep(1.0)  # 短暂等待，避免规划器拥堵

                self.get_logger().info(f"本轮路径处理完成：成功 {success_count} 个，失败 {failed_count} 个")

                # 检查队列中是否有新任务
                with self.task_lock:
                    if self.path_queue:
                        current_waypoints = self.path_queue
                        self.path_queue = []
                        self.get_logger().info("开始执行队列中的下一条路径")
                    else:
                        current_waypoints = None

            # 所有路径点（包括队列中的）都已尝试完毕
            # 如果没有任何点成功到达，则返回原点
            if success_count == 0 and failed_count > 0:
                self.get_logger().warn("所有路径点均无法到达，尝试返回原点")
                self._return_to_origin()

        except Exception as e:
            self.get_logger().error(f'导航线程异常: {str(e)}')
        finally:
            with self.task_lock:
                self.task_in_progress = False

    def _return_to_origin(self):
        """导航回记录的原点位置"""
        self.get_logger().info("正在返回原点...")
        origin_goal = PoseStamped()
        origin_goal.header.frame_id = self.origin_pose.header.frame_id
        origin_goal.header.stamp = self.get_clock().now().to_msg()
        origin_goal.pose = self.origin_pose.pose

        self.goToPose(origin_goal)
        while not self.isTaskComplete():
            time.sleep(0.5)

        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("成功返回原点")
        else:
            self.get_logger().error(f"返回原点失败: {result}")

def main():
    rclpy.init()

    # 创建节点
    navigator = AsyncNavigator()
    navigator.set_loop_times(1)  # 设置路径循环次数

    # 等待导航系统激活（已在 __init__ 中调用，此处可省略）
    # navigator.waitUntilNav2Active()
    navigator.get_logger().info("Nav2 已激活，等待路径消息...")

    def path_callback(msg):
        """接收路径回调并生成循环路径"""
        navigator.get_logger().info(f"收到路径消息，包含 {len(msg.poses)} 个点")

        # 根据循环次数生成重复路径
        repeated_poses = []
        for _ in range(navigator.loop_times):
            repeated_poses.extend(msg.poses)

        navigator.get_logger().info(f"生成循环路径，总路径点: {len(repeated_poses)}")
        navigator.follow_waypoints_async(repeated_poses)

    # 订阅路径话题
    subscription = navigator.create_subscription(
        Path,
        'my_path',
        path_callback,
        10
    )

    # 使用多线程执行器
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(navigator)
    try:
        navigator.get_logger().info("节点已启动，等待消息...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        navigator.get_logger().error(f"执行器异常: {str(e)}")
    finally:
        navigator.destroyNode()
        rclpy.shutdown()
        print("节点已关闭")

if __name__ == '__main__':
    main()
