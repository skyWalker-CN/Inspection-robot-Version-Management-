#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')  # 节点名为'simple_publisher'
        
        # 创建发布者，发布到'hello_topic'话题，使用String消息类型，队列大小为10
        self.publisher_ = self.create_publisher(String, 'hello_topic', 10)
        
        # 创建定时器：每秒调用一次publish_message
        self.timer = self.create_timer(1.0, self.publish_message)
        
        # 在终端输出提示信息
        self.get_logger().info('简单发布者节点已启动，每秒发布一条消息')

    def publish_message(self):
        msg = String()  # 创建消息对象
        msg.data = 'Hello World!'  # 设置消息内容
        
        # 发布消息
        self.publisher_.publish(msg)
        
        # 在终端显示发布状态
        self.get_logger().info(f'发布: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)  # 初始化ROS2
    
    simple_publisher = SimplePublisher()  # 创建节点
    
    try:
        rclpy.spin(simple_publisher)  # 保持节点运行
    except KeyboardInterrupt:  # 处理Ctrl+C中断
        pass
    
    # 清理工作
    simple_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()