#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class VideoSaverNode(Node):
    def __init__(self):
        super().__init__('video_saver_node')
        
        # 参数配置
        self.topic_name = '/camera/color/image_raw'
        self.fps = 30.0 # 录制的帧率，根据你相机的实际帧率调整
        self.output_dir = os.getcwd() # 默认保存到当前运行目录
        
        # 初始化 CvBridge
        self.bridge = CvBridge()
        
        # 订阅图像话题
        # 提示：如果你的相机发布的是 Best Effort (例如某些RealSense配置)，可能需要配置QoS
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.listener_callback,
            10)
        
        self.video_writer = None
        self.frame_size = None
        self.is_recording = False
        
        self.get_logger().info(f'已启动视频录制节点，正在等待话题: {self.topic_name} ...')

    def listener_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV图像 (BGR格式)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
            return

        # 如果是第一帧，初始化视频写入器
        if self.video_writer is None:
            height, width, _ = cv_image.shape
            self.frame_size = (width, height)
            self.init_writer(width, height)
        
        # 写入视频帧
        if self.video_writer:
            self.video_writer.write(cv_image)
            # 可选：显示实时画面 (如果在无头服务器上运行请注释掉下面两行)
            cv2.imshow("Real-time Monitor", cv_image)
            cv2.waitKey(1)

    def init_writer(self, width, height):
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ros2_video_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # 定义编码格式 (mp4v 兼容性较好)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.video_writer = cv2.VideoWriter(
            filepath, 
            fourcc, 
            self.fps, 
            (width, height)
        )
        self.is_recording = True
        self.get_logger().info(f'开始录制视频，分辨率: {width}x{height}, 保存路径: {filepath}')

    def release_writer(self):
        if self.video_writer:
            self.video_writer.release()
            self.get_logger().info('视频文件已保存并关闭。')
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = VideoSaverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C
        pass
    finally:
        # 确保节点关闭时保存视频
        node.release_writer()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()