#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import sys
import select
import termios
import tty
import os

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        # 创建cmd_vel发布者
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 保存终端设置
        self.settings = termios.tcgetattr(sys.stdin)
        
        # 控制参数
        self.linear_x = 0.5
        self.linear_y = 0.5
        self.angular_z = 0.5
        # 当前运动状态
        self.current_cmd = Twist()
        
        # 设置终端为非阻塞模式
        tty.setraw(sys.stdin.fileno())
        
        self.get_logger().info("键盘控制已启动!")
        self.get_logger().info("使用以下按键控制机器人运动:")
        self.get_logger().info("  i: 前进")
        self.get_logger().info("  j: 左转")
        self.get_logger().info("  l: 右转")
        self.get_logger().info("  m: 后退")
        self.get_logger().info("  u: 左前")
        self.get_logger().info("  o: 右前")
        self.get_logger().info("  n: 左后")
        self.get_logger().info("  ,: 右后")
        self.get_logger().info("  空格: 停止")
        self.get_logger().info("  Ctrl+C: 退出")
    
    def get_key(self):
        """非阻塞地获取键盘输入"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == 'i':    # 前进
                    self.current_cmd.linear.x = self.linear_x
                elif key == 'j':  # 左转
                    self.current_cmd.angular.z = self.angular_z
                elif key == 'l':  # 右转
                    self.current_cmd.angular.z = -self.angular_z
                elif key == 'm':  # 后退
                    self.current_cmd.linear.x = -self.linear_x
                elif key == 'u':  # 左前
                    self.current_cmd.linear.x = self.linear_x
                    self.current_cmd.angular.z = self.angular_z
                elif key == 'o':  # 右前
                    self.current_cmd.linear.x = self.linear_x
                    self.current_cmd.angular.z = -self.angular_z
                elif key == 'n':  # 左后
                    self.current_cmd.linear.x = -self.linear_x
                    self.current_cmd.angular.z = -self.angular_z
                elif key == ',':  # 右后
                    self.current_cmd.linear.x = -self.linear_x
                    self.current_cmd.angular.z = self.angular_z
                elif key == ' ':  # 停止
                    self.current_cmd = Twist()
                
                # 无论是否按键，都持续发布当前指令
                self.publisher.publish(self.current_cmd)
                
                # 退出程序
                if key == '\x03':  # Ctrl+C
                    break

        except Exception as e:
            self.get_logger().error(f"发生错误: {e}")
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            
            # 发布停止指令
            stop_cmd = Twist()
            self.publisher.publish(stop_cmd)
            self.get_logger().info("已停止机器人并退出")

def main():
    rclpy.init()
    node = KeyboardTeleop()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
