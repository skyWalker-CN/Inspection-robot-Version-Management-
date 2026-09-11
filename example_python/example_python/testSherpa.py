#!/usr/bin/env python3

import time
import rclpy
import threading
import numpy as np
import sherpa_onnx
import sounddevice as sd
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class SherpaController(Node):
    def __init__(self):
        super().__init__('sherpa_controller')

        # 创建发布者
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reco_publisher = self.create_publisher(String, '/chat_message', 10)
        self.dsk_publisher = self.create_publisher(String, '/getMSG', 10)

        # 模型路径（请根据实际修改）
        model_dir = "/home/jetson/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        vad_model_path = "/home/jetson/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/silero_vad.onnx"

        # 初始化 VAD 配置
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.8   # 静音0.8秒切分
        vad_config.silero_vad.min_speech_duration = 0.3
        vad_config.silero_vad.window_size = 512
        self.vad_config = vad_config

        # 初始化 ASR 识别器
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
            decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
        )
        self.get_logger().info("✅ Sherpa-ONNX 模型初始化完成")

        # 控制线程标志
        self.is_listening = False
        self.listening_thread = None
        self.command_thread = None
        self.current_cmd = Twist()
        self.recoStr = String()
        self.dskStr = String()

        # 状态变量
        self.current_state = "STATE_STOP"
        self.state_lock = threading.Lock()

        # 语音回复冷却：发布 /getMSG 后的2秒内忽略语音
        self.last_reco_time = 0.0
        self.reco_cooldown = 5.0
        self.reco_cooldown_lock = threading.Lock()

        # ---------- 新增：左转/右转计时 ----------
        self.turn_start_time = None      # 记录左转或右转开始的时刻
        self.turn_duration = 3.0         # 持续3秒后自动停止

    def start_listening(self):
        """启动语音监听线程"""
        if self.is_listening:
            self.get_logger().info('语音监听已在运行')
            return
        self.is_listening = True

        # 监听线程
        self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listening_thread.start()
        self.get_logger().info('语音监听线程已启动，等待唤醒词"小智"...')

        # 命令发布线程
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        self.get_logger().info('命令发布线程已启动')

    def _listen_loop(self):
        """在独立线程中运行 VAD + ASR，识别完整句子"""
        try:
            # 使用 sounddevice 采集音频
            with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
                while self.is_listening and rclpy.ok():
                    # 每次识别一个语音段，新建 VAD 实例
                    vad = sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=30)
                    while self.is_listening and rclpy.ok():
                        data, overflowed = stream.read(int(16000 * 0.1))  # 每帧100ms
                        if overflowed:
                            self.get_logger().warn('音频缓冲区溢出')
                        vad.accept_waveform(data.flatten())

                        if not vad.empty():
                            segment = vad.front
                            # 识别该段音频
                            text = self._transcribe(segment.samples)
                            if text:
                                # 检查是否在语音回复冷却期内
                                with self.reco_cooldown_lock:
                                    cooldown_remaining = time.time() - self.last_reco_time
                                if cooldown_remaining < self.reco_cooldown:
                                    self.get_logger().info(f'识别到: {text} (忽略，距上次发布/getMSG仅{cooldown_remaining:.1f}秒)')
                                else:
                                    self.get_logger().info(f'识别到: {text}')
                                    # 调用命令处理（内部会判断唤醒词）
                                    self._process_voice_command(text)
                            vad.pop()
                            break  # 处理完一段后重新新建 VAD
        except Exception as e:
            self.get_logger().error(f'监听循环发生错误: {e}')

    def _transcribe(self, samples):
        """使用 sherpa-onnx 识别音频片段"""
        if len(samples) == 0:
            return ""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        text = result.text if hasattr(result, 'text') else str(result)
        return text.strip()

    def _process_voice_command(self, text):
        """根据识别到的文本更新状态机"""
        # 检查唤醒词
        if not ('小智' in text or '小字' in text or '小志' in text or '同学' in text):
            return

        # 如果只有唤醒词（简短回复）
        if len(text) == 4:
            self.recoStr.data = '我在，请问我能为您做什么？'
            self.reco_publisher.publish(self.recoStr)
            # 更新最后回复时间
            with self.reco_cooldown_lock:
                self.last_reco_time = time.time()
            return

        # 处理具体命令
        with self.state_lock:
            if ("停" in text or "运动" in text or "只" in text or "止" in text or "STOP" in text or "stop" in text):
                self.current_state = "STATE_STOP"
                self.turn_start_time = None          # 清除转向计时
                self.get_logger().info('状态转换为: 停止')
                self.recoStr.data = '好的，已经停止。'
                self.reco_publisher.publish(self.recoStr)
                with self.reco_cooldown_lock:
                    self.last_reco_time = time.time()
                # 立即发布停止命令
                self.current_cmd.linear.x = 0.0
                self.current_cmd.angular.z = 0.0
                self.cmd_publisher.publish(self.current_cmd)

            elif "前" in text and self.current_state != "STATE_FORWARD":
                self.current_state = "STATE_FORWARD"
                self.turn_start_time = None          # 清除转向计时
                self.get_logger().info('状态转换为: 前进')
                self.recoStr.data = '好的，已经前进。'
                self.reco_publisher.publish(self.recoStr)
                with self.reco_cooldown_lock:
                    self.last_reco_time = time.time()

            elif "后" in text and self.current_state != "STATE_BACK":
                self.current_state = "STATE_BACK"
                self.turn_start_time = None          # 清除转向计时
                self.get_logger().info('状态转换为: 后退')
                self.recoStr.data = '好的，已经后退。'
                self.reco_publisher.publish(self.recoStr)
                with self.reco_cooldown_lock:
                    self.last_reco_time = time.time()

            elif "告诉" in text:
                cmd_part = text.split("告诉", 1)[-1].strip()
                self.dskStr.data = cmd_part
                self.dsk_publisher.publish(self.dskStr)
                # “告诉”命令不发布语音回复，因此不更新冷却时间

            elif "左" in text and self.current_state != "STATE_LEFT":
                self.current_state = "STATE_LEFT"
                self.turn_start_time = time.time()   # 记录转向开始时间
                self.get_logger().info('状态转换为: 左转 (将持续3秒)')
                self.recoStr.data = '好的，正在左转。'
                self.reco_publisher.publish(self.recoStr)
                with self.reco_cooldown_lock:
                    self.last_reco_time = time.time()

            elif "右" in text and self.current_state != "STATE_RIGHT":
                self.current_state = "STATE_RIGHT"
                self.turn_start_time = time.time()   # 记录转向开始时间
                self.get_logger().info('状态转换为: 右转 (将持续3秒)')
                self.recoStr.data = '好的，正在右转。'
                self.reco_publisher.publish(self.recoStr)
                with self.reco_cooldown_lock:
                    self.last_reco_time = time.time()

    def _command_loop(self):
        """独立线程，根据当前状态循环发布命令"""
        while self.is_listening and rclpy.ok():
            with self.state_lock:
                # 如果是左转或右转，检查是否超时
                if self.current_state in ("STATE_LEFT", "STATE_RIGHT"):
                    if self.turn_start_time is not None:
                        elapsed = time.time() - self.turn_start_time
                        if elapsed >= self.turn_duration:
                            # 超时，自动停止
                            self.get_logger().info(f'转向已持续{elapsed:.1f}秒，自动停止')
                            self.current_state = "STATE_STOP"
                            self.turn_start_time = None
                            # 发布停止命令
                            self.current_cmd.linear.x = 0.0
                            self.current_cmd.angular.z = 0.0
                            self.cmd_publisher.publish(self.current_cmd)
                            # 跳过本次循环后续的速度发布
                            time.sleep(0.2)
                            continue

                # 根据当前状态发布速度指令
                if self.current_state == "STATE_FORWARD":
                    self.current_cmd.linear.x = 0.1
                    self.current_cmd.angular.z = 0.0
                    self.cmd_publisher.publish(self.current_cmd)
                elif self.current_state == "STATE_BACK":
                    self.current_cmd.linear.x = -0.1
                    self.current_cmd.angular.z = 0.0
                    self.cmd_publisher.publish(self.current_cmd)
                elif self.current_state == "STATE_LEFT":
                    self.current_cmd.linear.x = 0.0
                    self.current_cmd.angular.z = 0.5
                    self.cmd_publisher.publish(self.current_cmd)
                elif self.current_state == "STATE_RIGHT":
                    self.current_cmd.linear.x = 0.0
                    self.current_cmd.angular.z = -0.5
                    self.cmd_publisher.publish(self.current_cmd)
                # 停止状态不发布速度

            time.sleep(0.2)

    def stop_listening(self):
        """停止语音监听"""
        self.is_listening = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
        self.get_logger().info('语音监听已停止')

    def destroy_node(self):
        self.stop_listening()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    controller = SherpaController()
    controller.start_listening()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
