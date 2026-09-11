#!/usr/bin/env python3

import time
import rclpy
import pyaudio
import threading
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from vosk import Model, KaldiRecognizer



class VoskController(Node):
    def __init__(self):
        super().__init__('vosk_controller')

        # 创建cmd_vel_1发布者
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 创建/getMSG发布者
        self.reco_publisher = self.create_publisher(String, '/getMSG', 10)
        
        # 创建//chat_message发布者
        self.dsk_publisher = self.create_publisher(String, '/chat_message', 10)

        # 设置模型
        self.model = Model("/home/jetson/vosk-model-cn-0.22")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        # 动态添加自定义词汇，这些词在后续识别中会被优先考虑
        self.custom_words = ["小智小智", "小智同学","请告诉我什么是永磁同步电机"]
        self.recognizer.SetWords(self.custom_words)

        # 设置麦克风输入
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = pyaudio.paInt16,  # 16位深度
            channels=1,              # 单声道
            rate=16000,              # 采样率16kHz
            input=True,              # 输入流
            frames_per_buffer=4000   # 缓冲区大小
        )

        # 控制线程运行的标志位
        self.is_listening = False
        self.listening_thread = None
        
        self.command_thread = None

        self.current_cmd = Twist()

        self.recoStr = String()
        
        self.dskStr = String()

        # 状态变量
        self.current_state = "STATE_STOP"  # 初始状态为停止
        self.state_lock = threading.Lock()  # 确保状态变量的线程安全

        
    def start_listening(self):
        """启动语音监听线程"""
        if self.is_listening:
            self.get_logger().info('语音监听已在运行')
            return
            
        self.is_listening = True

        # 创建并启动守护线程，主程序退出时该线程会自动结束
        self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listening_thread.start()
        self.get_logger().info('语音监听线程已启动，等待唤醒词"小智"...')

        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        self.get_logger().info('命令发布线程已启动...')

    def _listen_loop(self):
        """在独立线程中运行的监听循环"""
        try:
            while self.is_listening and rclpy.ok():
                data = self.stream.read(4000, exception_on_overflow=False)
                if len(data) == 0:
                    break
                    
                if self.recognizer.AcceptWaveform(data):
                    result = self.recognizer.Result()
                    try:
                        result_dict = eval(result)
                        text = result_dict.get("text", "")
                        print(text)
                        if text and (text.startswith('小智') or text.startswith('小字') or text.startswith('小志')):  # 检查是否以"小智"开头
                            self.get_logger().info(f'识别到指令: {text}')
                            text = text.replace(" ","")
                
                            self._process_voice_command(text)
                    except Exception as e:
                        self.get_logger().error(f'处理识别结果时出错: {e}')
        except Exception as e:
            self.get_logger().error(f'监听循环发生错误: {e}')



    def _process_voice_command(self, text):
        """根据识别到的文本更新状态机"""
        if len(text) == 4:
            self.recoStr.data = '我在，请问我能为您做什么？'
            self.reco_publisher.publish(self.recoStr)
        else:
            # 因为要修改状态，故使用线程锁确保状态变更的原子性
            with self.state_lock:
                if "停止" in text:
                    self.current_state = "STATE_STOP"
                    self.get_logger().info('状态转换为: 停止')
                    
                    self.recoStr.data = '好的，已经停止。'
                    self.reco_publisher.publish(self.recoStr)
                    
                    # 发布一次停止命令
                    self.current_cmd.linear.x = 0.0
                    self.current_cmd.angular.z = 0.0
                    self.cmd_publisher.publish(self.current_cmd)
                    self.get_logger().info('已发布命令: 停止')
                elif "前进" in text and self.current_state != "STATE_FORWARD":
                    self.current_state = "STATE_FORWARD"
                    self.get_logger().info('状态转换为: 前进')
                    
                    self.recoStr.data = '好的，已经前进。'
                    self.reco_publisher.publish(self.recoStr)
                    
                elif "后退" in text and self.current_state != "STATE_BACK":
                    self.current_state = "STATE_BACK"
                    self.get_logger().info('状态转换为: 后退')
                    
                    self.recoStr.data = '好的，已经后退。'
                    self.reco_publisher.publish(self.recoStr)
                    
                elif "告诉" in text:
                    self.dskStr.data = text[4:]
                    self.dsk_publisher.publish(self.dskStr)
                   
    def _command_loop(self):
        """独立线程，根据当前状态循环发布命令"""
        while self.is_listening and rclpy.ok():
            # 检查是否处于前进状态
            with self.state_lock:
                if self.current_state == "STATE_FORWARD":
                    self.current_cmd.linear.x = 0.1
                    self.current_cmd.angular.z = 0.0
                    self.cmd_publisher.publish(self.current_cmd)
                    self.get_logger().info('已发布命令: 前进')
                elif self.current_state == "STATE_BACK":
                    self.current_cmd.linear.x = -0.1
                    self.current_cmd.angular.z = 0.0
                    self.cmd_publisher.publish(self.current_cmd)
                    self.get_logger().info('已发布命令: 后退')

            time.sleep(0.2)


    def stop_listening(self):
        """停止语音监听"""
        self.is_listening = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)  # 等待线程结束，最多1秒
        self.get_logger().info('语音监听已停止')

    def destroy_node(self):
        """重写销毁节点的方法，确保资源被正确清理"""
        self.stop_listening()
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    controller = VoskController()
    controller.start_listening()  # 启动监听线程
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
