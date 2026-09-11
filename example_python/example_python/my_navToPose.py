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
        """后台线程中执行导航任务"""
        try:
            while waypoints:
                # 更新时间戳
                current_time = self.get_clock().now()
                updated_waypoints = []

                for i, pose in enumerate(waypoints):
                    # 深拷贝避免修改原始消息
                    new_pose = PoseStamped()
                    new_pose.header.frame_id = pose.header.frame_id
                    new_pose.header.stamp = (current_time + Duration(seconds=i)).to_msg()
                    new_pose.pose = pose.pose
                    updated_waypoints.append(new_pose)

                # 启动导航任务
                self.get_logger().info(f"启动导航任务，路径点数量: {len(updated_waypoints)}")
                self.followWaypoints(updated_waypoints)

                # 监控导航进度
                while not self.isTaskComplete():
                    time.sleep(0.5)
                    feedback = self.getFeedback()
                    if feedback and feedback.current_waypoint % 2 == 0:
                        self.get_logger().info(
                            f'导航进度: {feedback.current_waypoint+1}/{len(waypoints)}'
                        )

                # 处理结果
                result = self.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().info('导航成功!')
                elif result == TaskResult.CANCELED:
                    self.get_logger().warn('导航被取消')
                elif result == TaskResult.FAILED:
                    self.get_logger().error('导航失败')
                else:
                    self.get_logger().error('未知错误')
                
                # 检查是否有后续路径
                with self.task_lock:
                    if self.path_queue:
                        self.get_logger().info("执行队列中的下一条路径")
                        waypoints = self.path_queue
                        self.path_queue = []  # 清空队列
                    else:
                        waypoints = []  # 结束导航循环

        except Exception as e:
            self.get_logger().error(f'导航线程异常: {str(e)}')
        finally:
            # 标记任务完成
            with self.task_lock:
                self.task_in_progress = False

def main():
    rclpy.init()

    # 创建节点
    navigator = AsyncNavigator()
    navigator.set_loop_times(1)  # 设置路径循环次数为5

    # 等待导航系统激活
    navigator.waitUntilNav2Active()
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
