#!/usr/bin/env python3

import cv2
import time
import rclpy
import threading
from rclpy.node import Node
from onvif import ONVIFCamera
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image




class CameraController(Node):
    def __init__(self,name):
        super().__init__(name)
        
        # 创建视频帧发布
        self.image_pub = self.create_publisher(Image, '/camera/image', 10)
        self.bridge = CvBridge()
        
        self.camera_ip = "192.168.0.102"  
        self.camera_port = 80
        self.camera_username = "admin" 
        self.camera_password = "88888888"
        
        self.ptz_move = self.create_subscription(String,'/ptz/move',self.ptz_callback,10)
        
        try:
            # 创建ONVIF相机对象
            self.camera = ONVIFCamera(self.camera_ip, self.camera_port, self.camera_username, self.camera_password)
            # 创建媒体和PTZ服务
            self.media = self.camera.create_media_service()
            self.ptz = self.camera.create_ptz_service()

            # 获取媒体配置文件，通常第一个配置文件用于PTZ控制
            self.media_profile = self.media.GetProfiles()[0]

            # 获取RTSP流地址
            self.rtsp_url = self.get_rtsp_url()
            
            self.cap = None
                       
        except Exception as e:
            self.get_logger().error(f"相机连接失败: {e}")
            raise
            
        self.camera_thread = threading.Thread(target=self.start_video_stream)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        

    def ptz_callback(self,msg):
        moveStr = msg.data
        moveStr_split = moveStr.split(";")
        if moveStr_split[0] == "pan":
            self.continuous_move(pan = float(moveStr_split[1]), timeout = int(moveStr_split[2]))
        elif moveStr_split[0] == "tilt":
            self.continuous_move(tilt = float(moveStr_split[1]), timeout = int(moveStr_split[2]))
        else:
            self.get_logger().error("参数错误")
        
        
    def continuous_move(self, pan=0.0, tilt=0.0, timeout=0):
        """
                  连续移动PTZ
        pan: 水平速度 (-1.0~1.0, 负左正右)
        tilt: 垂直速度 (-1.0~1.0, 负下正上)
        timeout: 移动持续时间(秒)，0表示持续移动直到调用stop
        """
        try:
            # 创建连续移动请求
            move_request = self.ptz.create_type('ContinuousMove')
            move_request.ProfileToken = self.media_profile.token

            # 设置速度参数
            move_request.Velocity = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': 0.0}
            }

            # 执行移动
            self.ptz.ContinuousMove(move_request)

            # 如果设置了超时，在指定时间后停止
            if timeout > 0:
                time.sleep(timeout)
                self.stop()

        except Exception as e:
            self.get_logger().error(f"连续移动失败: {e}")
            
    def stop(self):
        """停止所有PTZ运动"""
        try:
            stop_request = self.ptz.create_type('Stop')
            stop_request.ProfileToken = self.media_profile.token
            stop_request.PanTilt = True
            stop_request.Zoom = True

            self.ptz.Stop(stop_request)
            self.get_logger().info("PTZ运动已停止")

        except Exception as e:
            self.get_logger().error(f"停止命令失败: {e}")
            
    def get_rtsp_url(self):
        """动态获取高质量的RTSP流地址"""
        try:
            stream_uri = self.media.GetStreamUri({
                'StreamSetup': {
                    'Stream': 'RTP-Unicast',
                    'Transport': {'Protocol': 'RTSP'}
                },
                'ProfileToken': self.media_profile.token
            })
            print(stream_uri.Uri)
            return stream_uri.Uri
        except Exception as e:
            self.get_logger().warning(f"动态获取流地址失败，使用默认: {e}")
            return "rtsp://10.0.1.137:554/stream2"  # 尝试主码流
            
    def start_video_stream(self):
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                self.get_logger().error("无法打开RTSP视频流")
                return
            
            self.get_logger().info("视频流已启动")
            
            while rclpy.ok():
                ret, frame = self.cap.read()
                #cv2.imshow("123",frame)
                #cv2.waitKey(1)
                
                if not ret:
                    self.get_logger().warning("无法读取视频帧")
                    continue
                    
                # 发布到ros2话题
                try:
                    ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    ros_image.header.stamp = self.get_clock().now().to_msg()
                    ros_image.header.frame_id = "camera"
                    self.image_pub.publish(ros_image)
                except Exception as e:
                    self.get_logger().error(f"图像转换/发布失败: {e}")
                                                    
        except Exception as e:
            print(f"视频流显示错误: {e}")
        finally:
            self.stop_video_stream()
            
    def stop_video_stream(self):
        if self.cap:
            self.cap.release()
            self.get_logger().info("视频流已停止")
        
    def destroy_node(self):
        self.stop_video_stream()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    controller = CameraController("camera_controller")
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

