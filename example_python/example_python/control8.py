# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# import math

# class FigureEightController(Node):
#     def __init__(self):
#         super().__init__('figure_eight_controller')
#         self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
#         # 8字形参数
#         self.declare_parameters(
#             namespace='',
#             parameters=[
#                 ('linear_speed', 0.1),      # 线速度 (m/s)
#                 ('angular_speed', 0.05),     # 角速度 (rad/s)
#                 ('transition_time', 2.0),   # 每个半圆的时间 (秒)
#             ]
#         )
        
#         self.linear_speed = self.get_parameter('linear_speed').value
#         self.angular_speed = self.get_parameter('angular_speed').value
#         self.transition_time = self.get_parameter('transition_time').value
        
#         # 控制变量
#         self.current_direction = 1  # 1表示顺时针，-1表示逆时针
#         self.last_switch_time = self.get_clock().now()
        
#         # 创建定时器，每0.1秒发布一次速度指令
#         self.timer = self.create_timer(0.1, self.publish_cmd_vel)
#         self.get_logger().info("Figure eight controller started. Robot will trace an ∞ pattern.")
        
#     def publish_cmd_vel(self):
#         """发布速度指令控制机器人走8字形"""
#         current_time = self.get_clock().now()
#         elapsed = (current_time - self.last_switch_time).nanoseconds / 1e9
        
#         # 检查是否需要切换方向
#         if elapsed > self.transition_time:
#             self.current_direction *= -1
#             self.last_switch_time = current_time
#             self.get_logger().info(f"Switching direction. Now moving {'clockwise' if self.current_direction == 1 else 'counter-clockwise'}")
        
#         # 创建Twist消息
#         twist = Twist()
#         twist.linear.x = self.linear_speed
#         twist.angular.z = self.current_direction * self.angular_speed
        
#         self.publisher.publish(twist)

# def main(args=None):
#     rclpy.init(args=args)
#     controller = FigureEightController()
    
#     try:
#         rclpy.spin(controller)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         controller.destroy_node()
#         rclpy.shutdown()
#         controller.get_logger().info("Figure eight controller shut down.")

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class FigureEightController(Node):
    def __init__(self):
        super().__init__('figure_eight_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 控制序列参数
        self.sequence = [
            (0.1, 0.05, 3.0),   # linear.x, angular.z, duration
            (0.1, -0.08, 6.0),
            (0.1, -0.05, 6.0),
            (0.1, 0.08, 6.0)
        ]
        
        # 控制变量
        self.current_step = 0
        self.start_time = self.get_clock().now()
        self.step_start_time = self.get_clock().now()
        
        # 创建定时器，每0.1秒发布一次速度指令
        self.timer = self.create_timer(0.1, self.publish_cmd_vel)
        self.get_logger().info("Figure eight controller started with custom sequence.")
        self.get_logger().info(f"Starting step {self.current_step+1}/{len(self.sequence)}")
        
    def publish_cmd_vel(self):
        """发布速度指令控制机器人执行指定序列"""
        current_time = self.get_clock().now()
        step_elapsed = (current_time - self.step_start_time).nanoseconds / 1e9
        
        # 获取当前步骤的参数
        linear, angular, duration = self.sequence[self.current_step]
        
        # 检查当前步骤是否完成
        if step_elapsed >= duration:
            self.current_step += 1
            self.step_start_time = current_time
            
            if self.current_step < len(self.sequence):
                self.get_logger().info(f"Starting step {self.current_step+1}/{len(self.sequence)}")
            else:
                # 所有步骤完成，停止机器人
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.publisher.publish(twist)
                self.get_logger().info("Sequence completed. Stopping robot.")
                self.timer.cancel()  # 停止定时器
                rclpy.shutdown()     # 关闭节点
                return
        
        # 获取当前步骤的参数
        linear, angular, _ = self.sequence[self.current_step]
        
        # 创建Twist消息
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        
        self.publisher.publish(twist)
        
        # 记录状态
        total_elapsed = (current_time - self.start_time).nanoseconds / 1e9
        if int(total_elapsed) % 1 == 0:  # 每秒记录一次
            self.get_logger().info(
                f"Step {self.current_step+1}/{len(self.sequence)}: "
                f"Time: {total_elapsed:.1f}s, "
                f"Linear: {linear:.2f} m/s, "
                f"Angular: {angular:.2f} rad/s"
            )

def main(args=None):
    rclpy.init(args=args)
    controller = FigureEightController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # 如果用户中断，停止机器人
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        controller.publisher.publish(twist)
        controller.get_logger().info("Interrupted. Stopping robot.")
    finally:
        controller.destroy_node()
        rclpy.try_shutdown()
        controller.get_logger().info("Controller shut down.")

if __name__ == '__main__':
    main()
