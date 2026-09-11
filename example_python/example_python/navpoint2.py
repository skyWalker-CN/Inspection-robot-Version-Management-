#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import os
import re
import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav2_msgs.action import Spin
from rclpy.action import ActionClient
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from openai import OpenAI

# ====================== 仪表识别配置 ======================
DASHSCOPE_API_KEY = "sk-ws-H.REXXLHL.96hG.MEYCIQDGmf1zYin-HUK5W2peKnH1LXVTdijFo3-ZQ-ROnutNxwIhANJ0-36E9KKDSgWv8qX86pLWRiCgsVmEMvl9AOgNnoIK"  # 请替换为你的API Key
MODEL_NAME = "qwen3.7-plus"

PROMPT_TEMPLATE = """
你是一名工业仪表计量专家，请严格按以下步骤分析图片中的仪表：
1. 检测画面中是否存在**仪表/表盘**（指针式压力表、温度表、流量表、电表等）
2. 若存在仪表，请分辨类型：
   - 指针式仪表：观察指针指向的刻度，结合量程范围读数
   - 数字式仪表：直接读取显示屏上的数字
3. 读取读数时注意：
   - 先确认量程（表盘上的最大值、最小值和单位）
   - 指针式仪表读数需判断指针指向的最近刻度线
   - 若表盘模糊、指针遮挡或反光导致无法确认，请如实降低置信度
4. 输出JSON格式结果：
{
  "has_meter": bool,
  "meter_type": str,  // "pointer"/"digital"/"none" 指针式/数字式
  "meter_name": str,  // 仪表名称，如"压力表"/"温度表"/"电压表"
  "reading_value": float,  // 读数数值，如 0.45（无法读取时为 null）
  "unit": str,  // 单位，如 "MPa"/"℃"/"V"/"A"/"kW"
  "range_min": float,  // 量程下限，如 0
  "range_max": float,  // 量程上限，如 1.6
  "is_normal": bool,  // 读数是否在正常范围内
  "confidence": float,  // 读数置信度 0-1，越接近1越确定
  "reading_advice": str  // 1句读数确认/复核建议
}
"""

def analyze_image_from_path(image_path: str, is_local: bool = True) -> dict:
    """从文件路径或URL分析图像（兼容旧功能）"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    content = []
    if is_local:
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_str}"
    else:
        image_url = image_path

    content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": PROMPT_TEMPLATE})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        raw_content = response.choices[0].message.content
        clean_content = re.sub(r'^```json\n|\n```$|^```|```$', '', raw_content, flags=re.MULTILINE).strip()
        return json.loads(clean_content)
    except json.JSONDecodeError as e:
        print(f"[JSON解析错误] 模型返回的内容不是标准JSON: {raw_content}")
        return {"error": "JSON解析失败", "raw_data": raw_content}
    except Exception as e:
        print(f"[API错误] {str(e)}")
        return {"error": str(e)}

def analyze_image_from_cv2(image_cv2) -> dict:
    """从OpenCV图像（numpy数组）分析仪表"""
    success, encoded_image = cv2.imencode('.jpg', image_cv2)
    if not success:
        return {"error": "图像编码失败"}
    base64_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{base64_str}"

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": PROMPT_TEMPLATE}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        raw_content = response.choices[0].message.content
        clean_content = re.sub(r'^```json\n|\n```$|^```|```$', '', raw_content, flags=re.MULTILINE).strip()
        return json.loads(clean_content)
    except json.JSONDecodeError as e:
        print(f"[JSON解析错误] 模型返回的内容不是标准JSON: {raw_content}")
        return {"error": "JSON解析失败", "raw_data": raw_content}
    except Exception as e:
        print(f"[API错误] {str(e)}")
        return {"error": str(e)}


class AsyncNavigator(BasicNavigator):
    """支持按路径点动态配置任务的导航节点。"""

    def __init__(self):
        super().__init__()
        self.task_in_progress = False
        self.loop_times = 1
        self.task_lock = threading.Lock()
        self.path_queue = []

        # key: 0-based 全局路径点索引
        # value: [(task_func, kwargs), ...]
        self.waypoint_tasks = {}

        # 原地旋转动作客户端
        self.spin_client = ActionClient(self, Spin, 'spin')

        # 底盘速度发布
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 云台控制话题
        self.gimbal_pub = self.create_publisher(String, '/controlCam', 10)

        # 图像订阅（用于仪表识别）
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.image_sub = self.create_subscription(
            Image,
            '/hik_camera/image_raw',
            self._image_callback,
            10
        )
        self.get_logger().info("已订阅相机话题: /hik_camera/image_raw")

    def _image_callback(self, msg: Image):
        """接收相机图像并缓存最新一帧"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.image_lock:
                self.latest_image = cv_image
                self.latest_image_time = time.time()
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    # ------------------------------------------------------------------
    # 对外配置接口
    # ------------------------------------------------------------------
    def set_loop_times(self, times):
        self.loop_times = max(1, times)

    def add_waypoint_task(self, waypoint_index, task_type, **kwargs):
        task_func = self._build_task_func(task_type, kwargs)
        if task_func is None:
            self.get_logger().error(f"未知任务类型: {task_type}")
            return
        with self.task_lock:
            self.waypoint_tasks.setdefault(waypoint_index, []).append(task_func)

    def load_task_config(self, text):
        try:
            data = json.loads(text)
        except Exception as e:
            self.get_logger().error(f"任务配置 JSON 解析失败: {e}")
            return

        if not isinstance(data, dict):
            self.get_logger().error("任务配置必须是 JSON 对象")
            return

        new_tasks = {}
        for key, value in data.items():
            try:
                point_one_based = int(key)
            except (ValueError, TypeError):
                self.get_logger().warn(f"忽略无效路径点序号: {key}")
                continue

            if point_one_based < 1:
                self.get_logger().warn(f"路径点序号应从 1 开始，忽略: {point_one_based}")
                continue

            waypoint_index = point_one_based - 1
            task_list = value if isinstance(value, list) else [value]
            parsed_tasks = []
            for task in task_list:
                if not isinstance(task, dict):
                    self.get_logger().warn(f"路径点 {point_one_based} 中存在非对象任务，已忽略: {task}")
                    continue
                task_func = self._parse_task_dict(task)
                if task_func is not None:
                    parsed_tasks.append(task_func)
                else:
                    self.get_logger().warn(f"路径点 {point_one_based} 的任务无法识别，已忽略: {task}")

            if parsed_tasks:
                new_tasks[waypoint_index] = parsed_tasks
            else:
                self.get_logger().warn(f"路径点 {point_one_based} 没有有效任务，已忽略")

        with self.task_lock:
            self.waypoint_tasks = new_tasks

        self.get_logger().info(f"已更新路径点任务配置，共 {len(new_tasks)} 个路径点配置了任务")

    # ------------------------------------------------------------------
    # 任务解析
    # ------------------------------------------------------------------
    def _build_task_func(self, task_type, kwargs):
        t = str(task_type).strip().lower()
        if t in ('wait', '停车等待', 'stop', 'park'):
            return (self._perform_wait_task,
                    {'wait_duration': float(kwargs.get('wait_duration', kwargs.get('duration', 5)))})
        if t in ('spin', '原地旋转', 'rotate'):
            return (self._perform_spin_task,
                    {'angle_degrees': float(kwargs.get('angle_degrees', kwargs.get('angle', 360))),
                     'time_allowance': float(kwargs.get('time_allowance', 30))})
        if t in ('cmd_vel', '速度'):
            return (self._perform_cmd_vel_task,
                    {'linear_x': float(kwargs.get('linear_x', 0.0)),
                     'linear_y': float(kwargs.get('linear_y', 0.0)),
                     'linear_z': float(kwargs.get('linear_z', 0.0)),
                     'angular_x': float(kwargs.get('angular_x', 0.0)),
                     'angular_y': float(kwargs.get('angular_y', 0.0)),
                     'angular_z': float(kwargs.get('angular_z', 0.0)),
                     'duration': float(kwargs.get('duration', 1.0))})
        if t in ('gimbal', '云台', 'ptz', 'camera'):
            return (self._perform_gimbal_task,
                    {'action': int(kwargs.get('action', 0)),
                     'value': float(kwargs.get('value', 0.0)),
                     'command': kwargs.get('command', kwargs.get('cmd'))})
        if t in ('meter', '仪表识别', 'inspect'):
            return (self._perform_meter_task,
                    {'image_path': kwargs.get('image_path'),
                     'is_local': bool(kwargs.get('is_local', True))})
        return None

    def _parse_task_dict(self, task):
        task_type = str(task.get('type', task.get('action_type', ''))).strip().lower()
        if not task_type:
            self.get_logger().warn(f"任务缺少 type 字段: {task}")
            return None

        if task_type in ('wait', '停车等待', 'stop', 'park'):
            return (self._perform_wait_task,
                    {'wait_duration': float(task.get('duration', task.get('wait_duration', 5)))})

        if task_type in ('spin', '原地旋转', 'rotate'):
            return (self._perform_spin_task,
                    {'angle_degrees': float(task.get('angle', task.get('angle_degrees', 360))),
                     'time_allowance': float(task.get('time_allowance', 30))})

        if task_type in ('cmd_vel', '速度', '直行'):
            return (self._perform_cmd_vel_task,
                    {'linear_x': float(task.get('linear_x', task.get('linear', 0.0))),
                     'linear_y': float(task.get('linear_y', 0.0)),
                     'linear_z': float(task.get('linear_z', 0.0)),
                     'angular_x': float(task.get('angular_x', 0.0)),
                     'angular_y': float(task.get('angular_y', 0.0)),
                     'angular_z': float(task.get('angular_z', task.get('angular', 0.0))),
                     'duration': float(task.get('duration', 1.0))})

        if task_type in ('gimbal', '云台', 'ptz', 'camera'):
            command = task.get('command', task.get('cmd'))
            if command is not None:
                return (self._perform_gimbal_task,
                        {'command': str(command), 'hold': float(task.get('hold', 0.0))})
            try:
                action = int(task.get('action'))
                value = float(task.get('value', 0.0))
            except (ValueError, TypeError):
                self.get_logger().warn(f"云台任务需要 action 和 value 字段，收到: {task}")
                return None
            return (self._perform_gimbal_task,
                    {'action': action, 'value': value, 'hold': float(task.get('hold', 0.0))})

        if task_type in ('meter', '仪表识别', 'inspect'):
            image_path = task.get('image_path') or task.get('url')
            return (self._perform_meter_task,
                    {'image_path': image_path,
                     'is_local': bool(task.get('is_local', True))})

        self.get_logger().warn(f"未知任务类型: {task_type}")
        return None

    # ------------------------------------------------------------------
    # 具体任务执行函数
    # ------------------------------------------------------------------
    def _perform_wait_task(self, wait_duration=5.0):
        try:
            self.get_logger().info(f'开始停车等待: {wait_duration} 秒')
            time.sleep(wait_duration)
            self.get_logger().info('停车等待完成')
            return True
        except Exception as e:
            self.get_logger().error(f'停车等待任务异常: {str(e)}')
            return False

    def _perform_spin_task(self, angle_degrees=360.0, time_allowance=30.0):
        try:
            self.get_logger().info(f'开始原地旋转: {angle_degrees} 度')
            if not self.spin_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().warn('Spin 动作服务器不可用，跳过原地旋转')
                return False

            goal_msg = Spin.Goal()
            goal_msg.target_yaw = angle_degrees * 3.141592653589793 / 180.0
            goal_msg.time_allowance = Duration(seconds=time_allowance).to_msg()

            goal_future = self.spin_client.send_goal_async(goal_msg)
            start_time = time.time()
            while not goal_future.done() and (time.time() - start_time) < time_allowance:
                time.sleep(0.05)

            if not goal_future.done():
                self.get_logger().warn('原地旋转超时')
                return False

            goal_handle = goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('原地旋转目标被拒绝')
                return False

            result_future = goal_handle.get_result_async()
            while not result_future.done() and (time.time() - start_time) < time_allowance:
                time.sleep(0.05)

            if not result_future.done():
                self.get_logger().warn('原地旋转结果等待超时')
                return False

            result = result_future.result()
            if result and result.status == 4:  # STATUS_SUCCEEDED
                self.get_logger().info('原地旋转完成')
                return True
            self.get_logger().error('原地旋转失败')
            return False
        except Exception as e:
            self.get_logger().error(f'原地旋转任务异常: {str(e)}')
            return False

    def _perform_cmd_vel_task(self, linear_x=0.0, linear_y=0.0, linear_z=0.0,
                              angular_x=0.0, angular_y=0.0, angular_z=0.0, duration=1.0):
        try:
            self.get_logger().info(
                f'开始 cmd_vel: linear=({linear_x}, {linear_y}, {linear_z}), '
                f'angular=({angular_x}, {angular_y}, {angular_z}), 持续 {duration} 秒')

            twist_msg = Twist()
            twist_msg.linear.x = float(linear_x)
            twist_msg.linear.y = float(linear_y)
            twist_msg.linear.z = float(linear_z)
            twist_msg.angular.x = float(angular_x)
            twist_msg.angular.y = float(angular_y)
            twist_msg.angular.z = float(angular_z)

            end_time = time.time() + float(duration)
            rate_hz = 10.0
            period = 1.0 / rate_hz

            while rclpy.ok() and time.time() < end_time:
                self.cmd_vel_pub.publish(twist_msg)
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                time.sleep(min(period, remaining))

            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            self.get_logger().info('cmd_vel 任务完成')
            return True
        except Exception as e:
            self.get_logger().error(f'cmd_vel 任务异常: {str(e)}')
            return False

    def _perform_gimbal_task(self, action=None, value=0.0, command=None, hold=0.0):
        try:
            if command is not None:
                cmd_str = str(command).strip()
            else:
                cmd_str = f"{int(action)} {float(value)}"

            msg = String()
            msg.data = cmd_str
            self.gimbal_pub.publish(msg)
            self.get_logger().info(f'云台命令已发送: {cmd_str}')

            if hold and hold > 0:
                time.sleep(float(hold))
            return True
        except Exception as e:
            self.get_logger().error(f'云台任务异常: {str(e)}')
            return False

    # ================== 仪表识别（异步） ==================
    def _perform_meter_task(self, image_path=None, is_local=True):
        """
        启动异步仪表识别任务。
        - 如果提供 image_path，则从文件或URL读取；
        - 否则使用最新缓存图像。
        该函数会立即返回，实际识别在后台线程中进行。
        """
        try:
            if image_path:
                self.get_logger().info(f'启动异步仪表识别任务（文件）: {image_path}')
                threading.Thread(
                    target=self._async_meter_from_file,
                    args=(image_path, is_local),
                    daemon=True
                ).start()
                return True
            else:
                with self.image_lock:
                    cv_image = self.latest_image
                if cv_image is None:
                    self.get_logger().error("没有可用的相机图像，无法执行仪表识别")
                    return False
                self.get_logger().info('启动异步仪表识别任务（实时相机）')
                threading.Thread(
                    target=self._async_meter_from_cv2,
                    args=(cv_image,),
                    daemon=True
                ).start()
                return True
        except Exception as e:
            self.get_logger().error(f'启动仪表识别任务异常: {str(e)}')
            return False

    def _async_meter_from_file(self, image_path, is_local):
        """后台线程：从文件分析仪表"""
        result = analyze_image_from_path(image_path, is_local)
        self._log_meter_result(result)

    def _async_meter_from_cv2(self, cv_image):
        """后台线程：从实时图像分析仪表"""
        result = analyze_image_from_cv2(cv_image)
        self._log_meter_result(result)

    def _log_meter_result(self, result):
        """输出仪表识别结果到日志"""
        if "error" in result:
            self.get_logger().error(f"仪表识别失败: {result['error']}")
            if "raw_data" in result:
                self.get_logger().error(f"原始返回: {result['raw_data']}")
            return

        if not result.get("has_meter"):
            self.get_logger().warn("画面中未检测到仪表")
            return

        meter_type = result.get('meter_type', '未知')
        meter_name = result.get('meter_name', '未知')
        reading = result.get('reading_value')
        unit = result.get('unit', '')
        range_min = result.get('range_min')
        range_max = result.get('range_max')
        is_normal = result.get('is_normal')
        confidence = result.get('confidence', 0.0)
        advice = result.get('reading_advice', '无')

        self.get_logger().info(
            f"✅ 仪表识别完成: 类型={meter_type}，名称={meter_name}，"
            f"读数={reading} {unit}，量程=[{range_min}, {range_max}] {unit}，"
            f"状态={'正常' if is_normal else '异常/越限'}，置信度={confidence:.0%}，"
            f"建议={advice}"
        )

        if is_normal is False:
            self.get_logger().warn(f"⚠️ 仪表读数异常！{meter_name}: {reading} {unit}")
        if confidence < 0.8:
            self.get_logger().warn(f"⚠️ 仪表读数置信度较低 ({confidence:.0%})，建议人工复核")

    # ------------------------------------------------------------------
    # 导航流程（逐点导航）
    # ------------------------------------------------------------------
    def _get_tasks_for_waypoint(self, waypoint_index):
        with self.task_lock:
            return list(self.waypoint_tasks.get(waypoint_index, []))

    def follow_waypoints_async(self, waypoints):
        with self.task_lock:
            if self.task_in_progress:
                self.path_queue.extend(waypoints)
                self.get_logger().info(
                    f"任务进行中，已缓存路径点，当前队列长度: {len(self.path_queue)}")
                return

            self.task_in_progress = True

        threading.Thread(
            target=self._follow_waypoints_thread,
            args=(waypoints,),
            daemon=True
        ).start()

    def _follow_waypoints_thread(self, waypoints):
        try:
            current_waypoint_set = list(waypoints)
            global_offset = 0

            while current_waypoint_set and rclpy.ok():
                for local_index, pose in enumerate(current_waypoint_set):
                    if not rclpy.ok():
                        return

                    global_index = global_offset + local_index
                    total_global_count = len(waypoints)

                    stamped = self._restamp_pose(pose)

                    self.get_logger().info(
                        f"导航到路点 [{global_index + 1}/{total_global_count}]")

                    if not self.followWaypoints([stamped]):
                        self.get_logger().warn(f"路点 {global_index + 1} 导航目标发送失败，跳过")
                        continue

                    while not self.isTaskComplete():
                        if not rclpy.ok():
                            return
                        time.sleep(0.1)

                    result = self.getResult()
                    if result != TaskResult.SUCCEEDED:
                        self.get_logger().warn(
                            f"路点 {global_index + 1} 未成功到达（结果={result}），仍尝试执行其任务")

                    tasks = self._get_tasks_for_waypoint(global_index)
                    if tasks:
                        self.get_logger().info(
                            f"路点 {global_index + 1} 执行 {len(tasks)} 个任务")

                        # 先停车
                        self.cmd_vel_pub.publish(Twist())

                        all_success = True
                        for task_func, kwargs in tasks:
                            try:
                                ok = task_func(**kwargs)
                                all_success = all_success and ok
                            except Exception as e:
                                self.get_logger().error(f'任务执行异常: {e}')
                                all_success = False

                        if all_success:
                            self.get_logger().info(f"路点 {global_index + 1} 的所有任务执行成功")
                        else:
                            self.get_logger().warn(f"路点 {global_index + 1} 的部分任务执行失败")

                with self.task_lock:
                    if self.path_queue:
                        self.get_logger().info("执行队列中的下一条路径")
                        next_waypoints = self.path_queue
                        self.path_queue = []
                        global_offset += len(current_waypoint_set)
                        current_waypoint_set = list(next_waypoints)
                    else:
                        current_waypoint_set = []

        except Exception as e:
            self.get_logger().error(f'导航线程异常: {e}')
        finally:
            with self.task_lock:
                self.task_in_progress = False
            self.get_logger().info("路径跟随线程结束")

    def _restamp_pose(self, pose):
        new_pose = PoseStamped()
        new_pose.header.frame_id = pose.header.frame_id
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.pose = pose.pose
        return new_pose


def main():
    rclpy.init()

    navigator = AsyncNavigator()
    navigator.set_loop_times(1)

    navigator.waitUntilNav2Active()
    navigator.get_logger().info("Nav2 已激活，等待路径和任务配置...")

    def task_config_callback(msg):
        navigator.load_task_config(msg.data)

    navigator.create_subscription(
        String,
        '/waypoint_tasks',
        task_config_callback,
        10
    )

    def path_callback(msg):
        navigator.get_logger().info(f"收到路径消息，包含 {len(msg.poses)} 个点")

        repeated_poses = []
        for _ in range(navigator.loop_times):
            repeated_poses.extend(msg.poses)

        navigator.get_logger().info(f"生成循环路径，总路径点: {len(repeated_poses)}")
        navigator.follow_waypoints_async(repeated_poses)

    navigator.create_subscription(
        Path,
        '/my_path',
        path_callback,
        10
    )

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
        navigator.destroy_node()
        rclpy.shutdown()
        print("节点已关闭")


if __name__ == '__main__':
    main()
