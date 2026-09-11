#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 + Nav2 路径跟随与路点任务节点（升级版）

支持的路点任务类型（4 类）：
  1. wait     停车等待
  2. spin     原地旋转（相对角度，正=逆时针/左转，负=顺时针/右转）
  3. cmd_vel  自定义速度指令（前进/后退/左转/右转/右前/右后等任意组合）
  4. gimbal   云台控制（向 /controlCam 发送 String，与 new_hksdk.py 对接）

任务通过 String 话题动态配置（在开始追点前发布即可）：
  订阅话题: waypoint_tasks
  消息格式: 用分号 ";" 分隔多个任务，每个任务用逗号 "," 分隔字段
            <路点序号>,<类型>[,参数...]

  路点序号从 1 开始（第 1 个点 = 1，即 Path 消息里的第 1 个 pose）。

  类型与参数：
    wait     ->  wait,<等待秒数>
    spin     ->  spin,<相对角度(度)>         例如 spin,90 表示原地左转 90°
    cmd_vel  ->  cmd_vel,<线速度>,<角速度>,<持续秒数>
    gimbal   ->  gimbal,<动作码>,<数值>       动作码/数值与 new_hksdk.py 的 /controlCam 一致

  cmd_vel 参数说明：
    line_x    前进为正，后退为负（m/s）
    angular_z 逆时针(左转)为正，顺时针(右转)为负（rad/s）
    例：右前行驶 3 秒 -> cmd_vel,0.2,-0.3,3

  gimbal 动作码（与 /controlCam 一致）：
    0-右, 1-左, 2-上, 3-下, 4-放大, 5-缩小, 6-雨刷, 7-补光,
    8-设置预置位(数值为编号), 9-调用预置位(数值为编号)
    例：云台左转 90° -> gimbal,1,90

  完整示例（一条 String 消息）：
    2,wait,5;5,spin,90;8,cmd_vel,0.2,-0.3,3;10,gimbal,1,90
    含义：
      第 2 个点 停车等待 5 秒
      第 5 个点 原地左转 90°
      第 8 个点 右前行驶 3 秒
      第 10 个点 云台左转 90°
"""

import math
import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


class AsyncNavigator(BasicNavigator):
    def __init__(self):
        super().__init__()

        # ------- 参数 -------
        self.declare_parameter("path_topic", "my_path")
        self.declare_parameter("tasks_topic", "waypoint_tasks")
        self.declare_parameter("gimbal_topic", "controlCam")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("loop_times", 1)
        self.declare_parameter("spin_angular_speed", 0.5)  # 原地旋转角速度 rad/s
        self.declare_parameter("cmd_vel_rate_hz", 10.0)    # 速度指令发布频率

        path_topic = self.get_parameter("path_topic").value
        tasks_topic = self.get_parameter("tasks_topic").value
        gimbal_topic = self.get_parameter("gimbal_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.loop_times = max(1, int(self.get_parameter("loop_times").value))
        self.spin_angular_speed = float(self.get_parameter("spin_angular_speed").value)
        self.cmd_vel_rate_hz = float(self.get_parameter("cmd_vel_rate_hz").value)

        # ------- 状态 -------
        self.task_in_progress = False
        self.task_lock = threading.Lock()
        self.path_queue = []           # 排队的路径
        self.waypoint_tasks = {}       # {路点序号(1开始): (类型, 参数列表)}

        # ------- 发布器 / 订阅器 -------
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.gimbal_pub = self.create_publisher(String, gimbal_topic, 10)

        self.path_sub = self.create_subscription(Path, path_topic, self._path_callback, 10)
        self.tasks_sub = self.create_subscription(String, tasks_topic, self._tasks_callback, 10)

        self.get_logger().info(
            f"路径话题: {path_topic} | 任务话题: {tasks_topic} | "
            f"云台话题: {gimbal_topic} | cmd_vel 话题: {cmd_vel_topic}"
        )

    # ==================================================================
    # 任务配置：通过 String 话题动态接收
    # ==================================================================
    def _tasks_callback(self, msg: String):
        """解析任务配置消息，替换当前任务表。"""
        tasks = {}
        # 兼容分号与换行分隔
        text = msg.data.replace("\r", "").replace("\n", ";")

        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            fields = [f.strip() for f in chunk.split(",")]
            if len(fields) < 2:
                self.get_logger().warn(f"忽略无效任务项: {chunk!r}")
                continue
            try:
                waypoint_no = int(fields[0])
            except ValueError:
                self.get_logger().warn(f"路点序号无法解析: {fields[0]!r}")
                continue
            if waypoint_no < 1:
                self.get_logger().warn(f"路点序号需从 1 开始，忽略: {waypoint_no}")
                continue

            task_type = fields[1].lower()
            params = fields[2:]
            tasks[waypoint_no] = (task_type, params)
            self.get_logger().info(
                f"注册任务: 路点 {waypoint_no} -> {task_type} {params}"
            )

        self.waypoint_tasks = tasks
        self.get_logger().info(f"任务配置更新完成，共 {len(tasks)} 个路点任务")

    def _path_callback(self, msg: Path):
        """收到路径后按 loop_times 复制，再交给后台线程执行。"""
        poses = list(msg.poses)
        if not poses:
            self.get_logger().warn("收到空路径，忽略")
            return

        repeated = []
        for _ in range(self.loop_times):
            repeated.extend(poses)

        self.get_logger().info(
            f"收到路径 {len(poses)} 个点，循环 {self.loop_times} 次，共 {len(repeated)} 个点"
        )
        self.follow_waypoints_async(repeated)

    # ==================================================================
    # 四类路点任务的具体执行
    # ==================================================================
    def _stop(self):
        """发布停止指令。"""
        self.cmd_vel_pub.publish(Twist())

    def _task_wait(self, params):
        """停车等待。"""
        seconds = self._to_float(params, 0, default=5.0)
        self.get_logger().info(f"[wait] 停车等待 {seconds} 秒")
        self._stop()
        time.sleep(seconds)
        return True

    def _task_spin(self, params):
        """原地旋转相对角度。"""
        angle_deg = self._to_float(params, 0, default=90.0)
        if abs(self.spin_angular_speed) < 1e-6:
            self.get_logger().error("[spin] spin_angular_speed 不能为 0")
            return False

        angular_z = self.spin_angular_speed if angle_deg >= 0 else -self.spin_angular_speed
        duration = abs(math.radians(angle_deg)) / abs(self.spin_angular_speed)
        self.get_logger().info(f"[spin] 原地旋转 {angle_deg}°，持续 {duration:.2f} 秒")
        self._publish_twist(angular_z=angular_z, duration=duration)
        return True

    def _task_cmd_vel(self, params):
        """自定义速度指令。"""
        if len(params) < 3:
            self.get_logger().warn("[cmd_vel] 需要 3 个参数: 线速度,角速度,持续时间")
            return False
        linear_x = self._to_float(params, 0, default=0.0)
        angular_z = self._to_float(params, 1, default=0.0)
        duration = self._to_float(params, 2, default=1.0)
        self.get_logger().info(
            f"[cmd_vel] linear.x={linear_x}, angular.z={angular_z}, 持续 {duration} 秒"
        )
        self._publish_twist(linear_x=linear_x, angular_z=angular_z, duration=duration)
        return True

    def _task_gimbal(self, params):
        """云台控制：转发到 /controlCam 的 String。"""
        if len(params) < 2:
            self.get_logger().warn("[gimbal] 需要 2 个参数: 动作码,数值")
            return False
        action, value = params[0], params[1]
        msg = String()
        msg.data = f"{action} {value}"
        self.gimbal_pub.publish(msg)
        self.get_logger().info(f"[gimbal] 发送云台指令: {msg.data!r}")
        return True

    def _execute_task(self, task_type, params):
        try:
            if task_type == "wait":
                return self._task_wait(params)
            if task_type == "spin":
                return self._task_spin(params)
            if task_type == "cmd_vel":
                return self._task_cmd_vel(params)
            if task_type == "gimbal":
                return self._task_gimbal(params)
            self.get_logger().warn(f"未知任务类型: {task_type}")
            return False
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"任务执行异常 [{task_type} {params}]: {e}")
            return False

    @staticmethod
    def _to_float(params, index, default):
        try:
            return float(params[index])
        except (IndexError, ValueError):
            return default

    def _publish_twist(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        """按固定频率发布速度指令，结束补发停止。"""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)

        # 注意：这里用 time.sleep 而不是 create_rate()/rate.sleep()。
        # create_rate() 在非执行器线程里调用时，部分 rclpy 版本会行为异常，
        # 导致只发一帧就退出，机器人表现为“不动”。
        rate_hz = max(1.0, self.cmd_vel_rate_hz)
        period = 1.0 / rate_hz
        start = time.time()
        while rclpy.ok() and (time.time() - start) < duration:
            self.cmd_vel_pub.publish(twist)
            time.sleep(period)

        self._stop()

    # ==================================================================
    # 路径跟随（后台线程，逐点导航 + 到点执行任务）
    # ==================================================================
    def follow_waypoints_async(self, waypoints):
        with self.task_lock:
            if self.task_in_progress:
                self.path_queue.append(list(waypoints))
                self.get_logger().info(
                    f"导航进行中，已缓存新路径，当前队列长度 {len(self.path_queue)}"
                )
                return
            self.task_in_progress = True

        threading.Thread(
            target=self._follow_waypoints_thread,
            args=(waypoints,),
            daemon=True,
        ).start()

    def _follow_waypoints_thread(self, waypoints):
        try:
            current = list(waypoints)
            while current and rclpy.ok():
                # 逐点导航：保证到达第 N 个点后，再执行该点的任务
                for index, pose in enumerate(current):
                    if not rclpy.ok():
                        return

                    waypoint_no = index + 1  # 1 开始，用于匹配任务配置
                    stamped = self._restamp_pose(pose)

                    self.get_logger().info(
                        f"导航到路点 {waypoint_no}/{len(current)}"
                    )
                    if not self.followWaypoints([stamped]):
                        self.get_logger().warn(f"路点 {waypoint_no} 导航目标发送失败，跳过")
                        continue

                    while not self.isTaskComplete():
                        if not rclpy.ok():
                            return
                        time.sleep(0.2)

                    result = self.getResult()
                    if result != TaskResult.SUCCEEDED:
                        self.get_logger().warn(
                            f"路点 {waypoint_no} 未成功到达（结果={result}），仍尝试执行其任务"
                        )
                    else:
                        self.get_logger().info(f"已到达路点 {waypoint_no}")

                    task = self.waypoint_tasks.get(waypoint_no)
                    if task:
                        task_type, params = task
                        self.get_logger().info(
                            f"路点 {waypoint_no} 执行任务: {task_type} {params}"
                        )
                        ok = self._execute_task(task_type, params)
                        if ok:
                            self.get_logger().info(f"路点 {waypoint_no} 任务执行完成")
                        else:
                            self.get_logger().warn(f"路点 {waypoint_no} 任务执行失败")

                # 当前路径走完，检查是否有排队的新路径
                with self.task_lock:
                    if self.path_queue:
                        current = self.path_queue.pop(0)
                        self.get_logger().info(
                            f"执行队列中的下一条路径（{len(current)} 个点）"
                        )
                    else:
                        current = []
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"导航线程异常: {e}")
        finally:
            with self.task_lock:
                self.task_in_progress = False
            self.get_logger().info("路径跟随线程结束")

    def _restamp_pose(self, pose):
        """重打时间戳，避免复用 Path 消息时因旧时间戳导致异常。"""
        new_pose = PoseStamped()
        new_pose.header.frame_id = pose.header.frame_id
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.pose = pose.pose
        return new_pose


def main(args=None):
    rclpy.init(args=args)
    navigator = AsyncNavigator()
    navigator.waitUntilNav2Active()
    navigator.get_logger().info("Nav2 已激活，等待路径与任务配置消息...")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(navigator)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
