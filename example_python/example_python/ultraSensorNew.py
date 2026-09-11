#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import struct
import math
import threading
import time
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class UltraSensor(Node):
    def __init__(self):
        super().__init__('ultra_sensor')

        # 初始化串口连接
        try:
            self.ser = serial.Serial(
                port='/dev/ttyCH341USB0',
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            self.get_logger().info("Connected to ultra_sensor at /dev/ttyUSB0")
        except serial.SerialException as e:
            self.get_logger().error(f"Serial connection error: {str(e)}")
            raise SystemExit

        # 发布者
        self.sensor_pub = self.create_publisher(String, 'ultarSensor', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 数据存储
        self.group3_data = []      # 存储 (timestamp, distance)
        self.data_lock = threading.Lock()
        self.is_stopped = False    # 当前是否处于停车状态
        self.stop_timer = None     # 定时器句柄

        # 启动串口读取线程
        self.serial_thread = threading.Thread(target=self.read_serial)
        self.serial_thread.daemon = True
        self.serial_thread.start()

    def _clean_old_data(self, current_time):
        """移除超过3秒的数据"""
        cutoff = current_time - 1.0
        self.group3_data = [(t, v) for t, v in self.group3_data if t > cutoff]

    def _get_ratios(self):
        """计算最近3秒内小于800和大于800的比例"""
        with self.data_lock:
            now = self.get_clock().now().nanoseconds / 1e9
            self._clean_old_data(now)
            total = len(self.group3_data)
            if total == 0:
                return 0.0, 0.0
            count_less = sum(1 for _, v in self.group3_data if v < 800)
            count_more = sum(1 for _, v in self.group3_data if v > 800)
            return count_less / total, count_more / total

    def _update_stop_state(self):
        """根据最近数据更新停车状态"""
        ratio_less, ratio_more = self._get_ratios()

        if not self.is_stopped:
            # 当前未停车，检查是否需要停车
            if ratio_less >= 1.0:
                self.get_logger().info(f"Stop condition met: {ratio_less:.2%} of data < 800. Entering STOP state.")
                self.is_stopped = True
                # 创建定时器，每0.1秒发布一次停止命令
                self.stop_timer = self.create_timer(0.01, self.publish_stop_cmd)
        else:
            # 当前已停车，检查是否需要恢复
            if ratio_more >= 1.0:
                self.get_logger().info(f"Recovery condition met: {ratio_more:.2%} of data > 800. Exiting STOP state.")
                self.is_stopped = False
                if self.stop_timer is not None:
                    self.stop_timer.cancel()
                    self.stop_timer = None

    def publish_stop_cmd(self):
        """发布停止命令"""
        twist = Twist()
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)
        # 可选：减少日志频率，避免刷屏
        # self.get_logger().debug("Published stop command")

    def read_serial(self):
        """串口数据读取线程"""
        while True:
            raw_data = self.ser.read(10)
            print(raw_data)

            if len(raw_data) != 10:
                print(f"数据长度不足: 期望10字节, 实际收到 {len(raw_data)} 字节。继续等待...")
                continue

            if raw_data[0] != 0xFF:
                print(f"帧头错误: 期望 0xFF, 实际收到 {hex(raw_data[0])}。跳过此帧。")
                continue

            try:
                group1, group2, group3, group4 = struct.unpack('>4H', raw_data[1:9])
            except struct.error as e:
                print(f"解析数据时发生错误: {e}")
                continue

            print(f"解析成功 -> 组1: {group1}, 组2: {group2}, 组3: {group3}, 组4: {group4}")

            # 发布原始数据（原功能）
            ultra_sensor = (f"{group1/1000:.3f}m {group2/1000:.3f}m "
                            f"{group3/1000:.3f}m {group4/1000:.3f}m")
            ultra_msg = String()
            ultra_msg.data = ultra_sensor
            self.sensor_pub.publish(ultra_msg)

            # 存储组3数据
            now = self.get_clock().now().nanoseconds / 1e9
            with self.data_lock:
                self.group3_data.append((now, group3))

            # 更新停车状态
            self._update_stop_state()

    def destroy_node(self):
        """重写销毁方法，确保资源释放"""
        if self.stop_timer is not None:
            self.stop_timer.cancel()
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = UltraSensor()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
