#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 异常警报语音播报节点
订阅 /anomaly_detection/alerts 和 /temperature，根据温度决定播报内容
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import re
import urllib.request
import json
import time
import threading
import tempfile
import traceback
import wave
from pathlib import Path

import piper
import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------- 配置 ----------
MODEL_DIR = Path.home() / ".cache" / "piper_models"
MODEL_URLS = {
    "zh": {
        "onnx": "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx",
        "json": "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"
    },
    "en": {
        "onnx": "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://hf-mirror.com/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    }
}

class AudioPlayer:
    """音频播放器类（与原代码相同）"""
    def __init__(self, logger):
        self.logger = logger
        self.play_queue = []
        self.currently_playing = None
        self.playing_thread = None
        pygame.mixer.init()
        self.logger.info("音频播放器初始化完成")

    def add_to_queue(self, audio_file):
        self.play_queue.append(audio_file)
        self.logger.info(f"已加入播放队列: {audio_file} (队列长度: {len(self.play_queue)})")
        if not self.is_playing():
            self._play_next()

    def is_playing(self):
        return self.currently_playing is not None

    def _play_next(self):
        if not self.play_queue or self.is_playing():
            return
        audio_file = self.play_queue.pop(0)
        self.currently_playing = audio_file
        self.playing_thread = threading.Thread(target=self._play_audio, args=(audio_file,))
        self.playing_thread.daemon = True
        self.playing_thread.start()

    def _play_audio(self, audio_file):
        try:
            self.logger.info(f"开始播放: {audio_file}")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"播放失败: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info(f"播放完成: {audio_file}")
            self.currently_playing = None
            try:
                os.remove(audio_file)
                self.logger.debug(f"已清理临时文件: {audio_file}")
            except:
                self.logger.warning(f"清理文件失败: {audio_file}")
            if self.play_queue:
                self.logger.info("开始播放下一个音频")
                self._play_next()

class AlertTtsNode(Node):
    def __init__(self):
        super().__init__('alert_tts_node')
        
        # 订阅警报话题
        self.subscription = self.create_subscription(
            String,
            '/anomaly_detection/alerts',
            self.alert_callback,
            10
        )
        
        # 订阅温度话题
        self._temSubscription = self.create_subscription(
            String,
            '/temperature',
            self.tem_callback,
            10
        )

        self._msgSubscription = self.create_subscription(
            String,
            '/chat_message',
            self.msg_callback,
            10
        )
        
        self.tem = '0.0'                     # 当前温度
        self.last_fire_alert_time = 0        # 最近一次火灾警报消息到达时间
        self.last_high_temp_reminder = 0     # 最近一次高温提醒时间
        
        # 配置参数
        self.temp_threshold = 100.0          # 温度阈值（℃）
        self.fire_alert_window = 5.0         # 火灾警报有效窗口（秒）
        self.high_temp_cooldown = 30.0       # 高温提醒冷却时间（秒）
        self.cooldown_sec = 5.0              # 同一类型警报最小间隔（秒）
        
        # 缓存语音模型
        self.voice_cache = {}
        # 警报去重记录
        self.last_alert_time = {}
        
        # 音频播放器
        self.audio_player = AudioPlayer(self.get_logger())
        
        self.get_logger().info('警报语音播报节点已启动，等待消息...')
        
    def msg_callback(self, msg):
        self.synthesize(msg.data)

    def tem_callback(self, msg):
        """温度更新回调：检查高温提醒"""
        self.tem = msg.data
        current_temp = float(self.tem)
        current_time = time.time()
        
        self.get_logger().debug(f"当前温度: {current_temp}℃")
        
        # 温度超过阈值且未在火灾警报窗口内
        if current_temp > self.temp_threshold:
            has_recent_fire = (current_time - self.last_fire_alert_time) < self.fire_alert_window
            if not has_recent_fire:
                # 检查高温提醒冷却
                if (current_time - self.last_high_temp_reminder) > self.high_temp_cooldown:
                    self.get_logger().info("温度超过100℃且无火灾警报，播报高温提醒")
                    self.last_high_temp_reminder = current_time
                    self.synthesize("发现异常高温物体，请尽快前往查看")
                else:
                    self.get_logger().debug("高温提醒冷却中，跳过")
            else:
                self.get_logger().debug(f"最近{self.fire_alert_window}秒内有火灾警报，不重复提醒")

    def detect_language(self, text):
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        else:
            return "en"

    def download_model(self, lang):
        self.get_logger().info(f'正在下载 {lang} 语音模型...')
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ["onnx", "json"]:
            url = MODEL_URLS[lang][ext]
            local_path = MODEL_DIR / f"{lang}.{ext}"
            if local_path.exists():
                self.get_logger().info(f'{local_path} 已存在，跳过下载')
                continue
            self.get_logger().info(f'下载 {url} -> {local_path}')
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                self.get_logger().error(f'下载失败：{e}')
                sys.exit(1)
        self.get_logger().info(f'{lang} 模型下载完成')

    def get_model_path(self, lang):
        model_onnx = MODEL_DIR / f"{lang}.onnx"
        model_json = MODEL_DIR / f"{lang}.json"
        if not model_onnx.exists() or not model_json.exists():
            self.get_logger().info(f'未找到 {lang} 语音模型，将自动下载...')
            self.download_model(lang)
        return model_onnx, model_json

    def synthesize(self, text, lang=None):
        """合成并播放语音（写入临时文件后交给播放器）"""
        if not text.strip():
            self.get_logger().warning('文本为空，跳过合成')
            return

        if lang is None:
            lang = self.detect_language(text)
        self.get_logger().info(f'检测到语言: {lang}')

        if lang not in self.voice_cache:
            model_onnx, model_json = self.get_model_path(lang)
            self.get_logger().info('加载语音模型...')
            self.voice_cache[lang] = piper.PiperVoice.load(model_onnx, model_json)

        voice = self.voice_cache[lang]

        self.get_logger().info('合成语音中...')
        audio_data = b""
        sample_rate = None
        sample_width = None
        channels = None

        for chunk in voice.synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate
                sample_width = chunk.sample_width
                channels = chunk.sample_channels
            audio_data += chunk.audio_int16_bytes

        # 保存为临时WAV文件
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                tmp_path = tmpfile.name
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            self.get_logger().info(f'临时音频文件已保存: {tmp_path}')
        except Exception as e:
            self.get_logger().error(f'保存临时文件失败: {e}')
            return

        self.audio_player.add_to_queue(tmp_path)

    def alert_callback(self, msg):
        """处理警报消息"""
        try:
            data = json.loads(msg.data)
            alert_type = data.get('type', 'unknown')
            description = data.get('description', '')
            current_time = time.time()
            current_temp = float(self.tem)
            
            # 记录火灾警报消息时间（无论是否播报）
            if alert_type == 'fire':
                self.last_fire_alert_time = current_time
            
            # 去重检查（所有类型都适用）
            last_time = self.last_alert_time.get(alert_type, 0)
            if current_time - last_time < self.cooldown_sec:
                self.get_logger().info(f'跳过重复警报（{alert_type}）: {description[:30]}...')
                return
            self.last_alert_time[alert_type] = current_time
            
            # 根据温度和警报类型决定是否播报
            if alert_type == 'fire' or alert_type == 'smoke':
                # 火灾警报：仅当温度超过阈值时播报
                if current_temp > self.temp_threshold:
                    if not description:
                        self.get_logger().warning('火灾警报消息中没有 description 字段，跳过')
                        return
                    self.get_logger().info(f'播报火灾警报: {description}')
                    self.synthesize(description)
                else:
                    self.get_logger().info(f'温度{current_temp}℃≤{self.temp_threshold}℃，忽略火灾警报')
            else:
                # 其他类型警报：始终播报（去重已处理）
                if not description:
                    self.get_logger().warning(f'{alert_type}警报消息中没有 description 字段，跳过')
                    return
                self.get_logger().info(f'播报{alert_type}警报: {description}')
                self.synthesize(description)
            
        except json.JSONDecodeError:
            self.get_logger().error('收到的消息不是有效的JSON')
        except Exception as e:
            self.get_logger().error(f'处理警报消息时出错: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = AlertTtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('收到Ctrl+C，关闭节点...')
    finally:
        pygame.mixer.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
