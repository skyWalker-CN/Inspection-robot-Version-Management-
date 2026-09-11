#!/usr/bin/env python3
import asyncio
import websockets
import json
import numpy as np
import cv2
import sys
import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ThermalCameraNode(Node):
    def __init__(self, esp32_ip='gacook.local'):
        super().__init__('thermal_camera_node')
        self.publisher_ = self.create_publisher(String, '/temperature', 10)
        
        self.websocket_url = f"ws://{esp32_ip}:81/"
        self.frame_count = 0
        self.last_output_time = 0
        
        # 温度范围设置
        self.min_temp = 0.0   # 蓝色对应的温度
        self.max_temp = 40.0  # 红色对应的温度
        
        # 创建窗口
        #self.window_name = "热成像"
        #cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        #cv2.resizeWindow(self.window_name, 320, 240)
        
        # 启动异步任务
        self.running = True
        self.thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        self.thread.start()
        
        self.get_logger().info(f'热成像节点已启动')
        self.get_logger().info(f"连接到: {self.websocket_url}")
    
    def publish_temperature(self, temperature):
        """发布温度到ROS2话题"""
        msg = String()
        msg.data = f"{temperature:.2f}"
        self.publisher_.publish(msg)
        self.get_logger().debug(f'发布温度: {temperature:.2f}°C', throttle_duration_sec=1.0)
    
    async def async_websocket_task(self):
        """异步WebSocket任务"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.get_logger().info("WebSocket连接成功!")
                
                while self.running and rclpy.ok():
                    try:
                        # 接收数据
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if 'thermal' in data:
                            # 解码温度
                            thermal_array = np.array(data['thermal'], dtype=np.float32) / 10.0
                            if len(thermal_array) == 768:
                                thermal_matrix = thermal_array.reshape(24, 32)
                                
                                # 计算当前帧的实际温度范围
                                frame_min_temp = np.min(thermal_matrix)
                                frame_max_temp = np.max(thermal_matrix)
                                
                                # 发布最高温度到ROS2
                                self.publish_temperature(frame_max_temp)
                                
                                # 每0.5秒输出一次温度信息
                                current_time = time.time()
                                if current_time - self.last_output_time > 0.5:
                                    self.get_logger().info(
                                        f"帧: {self.frame_count:4d} | "
                                        f"温度范围: {frame_min_temp:5.1f}°C ~ {frame_max_temp:5.1f}°C | "
                                        f"显示范围: {self.min_temp:5.1f}°C ~ {self.max_temp:5.1f}°C"
                                    )
                                    self.last_output_time = current_time
                                
                                # 快速归一化
                                normalized = ((thermal_matrix - self.min_temp) / 
                                            (self.max_temp - self.min_temp) * 255)
                                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
                                
                                # 应用彩虹调色板
                                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                                
                                # 放大显示
                                display_img = cv2.resize(colored, (320, 240), 
                                                        interpolation=cv2.INTER_NEAREST)
                                
                                # 显示图像
                                # cv2.imshow(self.window_name, display_img)
                                
                                self.frame_count += 1
                        
                        # 检查按键
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.running = False
                            break
                        elif key == ord('+'):
                            self.max_temp += 5
                            self.get_logger().info(f"调整显示范围: {self.min_temp}°C - {self.max_temp}°C")
                        elif key == ord('-'):
                            self.max_temp = max(self.min_temp + 5, self.max_temp - 5)
                            self.get_logger().info(f"调整显示范围: {self.min_temp}°C - {self.max_temp}°C")
                        elif key == ord('r'):
                            self.min_temp = 0.0
                            self.max_temp = 50.0
                            self.get_logger().info(f"重置显示范围: {self.min_temp}°C - {self.max_temp}°C")
                            
                    except Exception as e:
                        self.get_logger().error(f"数据处理错误: {e}")
                        continue
                        
        except Exception as e:
            self.get_logger().error(f"WebSocket连接错误: {e}")
        finally:
            cv2.destroyAllWindows()
            self.get_logger().info(f"总共处理 {self.frame_count} 帧")
    
    def run_asyncio_loop(self):
        """在新的线程中运行asyncio事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_websocket_task())
    
    def destroy_node(self):
        """清理资源"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    # 解析命令行参数
    ip = sys.argv[1] if len(sys.argv) > 1 else 'gacook.local'
    
    rclpy.init(args=args)
    
    try:
        node = ThermalCameraNode(ip)
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("接收到中断信号")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
