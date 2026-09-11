#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 异常警报语音播报节点（优化版）
- 异步合成队列，避免阻塞主线程
- 文本音频缓存（完整 WAV 字节），减少重复推理
- 串行合成，降低 CPU 峰值
"""

import os
import sys
import re
import urllib.request
import json
import time
import threading
import queue
import tempfile
import traceback
import wave
import io
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

# ---------- 音频播放器 ----------
class AudioPlayer:
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
            except Exception as e:
                self.logger.warning(f"清理文件失败: {audio_file} - {e}")
            if self.play_queue:
                self.logger.info("开始播放下一个音频")
                self._play_next()


# ---------- ROS2 节点 ----------
class AlertTtsNode(Node):
    def __init__(self):
        super().__init__('alert_tts_node')

        # 订阅
        self.subscription = self.create_subscription(
            String, '/anomaly_detection/alerts', self.alert_callback, 10)
        self._temSubscription = self.create_subscription(
            String, '/temperature', self.tem_callback, 10)
        self._msgSubscription = self.create_subscription(
            String, '/chat_message', self.msg_callback, 10)

        # 状态变量
        self.tem = '0.0'
        self.last_fire_alert_time = 0
        self.last_high_temp_reminder = 0
        self.temp_threshold = 100.0
        self.fire_alert_window = 5.0
        self.high_temp_cooldown = 30.0
        self.cooldown_sec = 15.0

        # 模型缓存
        self.voice_cache = {}
        self.last_alert_time = {}

        # 播放器
        self.audio_player = AudioPlayer(self.get_logger())

        # 异步合成队列
        self.synth_queue = queue.Queue()
        self.synth_worker = threading.Thread(target=self._synth_worker, daemon=True)
        self.synth_worker.start()

        # 音频缓存 (text -> 完整 WAV 字节)
        self.audio_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 20

        self.get_logger().info('警报语音播报节点（优化版）已启动')

    # ---------- 工作线程 ----------
    def _synth_worker(self):
        while True:
            text, lang = self.synth_queue.get()
            try:
                self._synthesize_and_play(text, lang)
            except Exception as e:
                self.get_logger().error(f"合成线程异常: {e}")
                self.get_logger().error(traceback.format_exc())
            finally:
                self.synth_queue.task_done()

    # ---------- 外部接口 ----------
    def synthesize(self, text, lang=None):
        if not text.strip():
            return
        if lang is None:
            lang = self.detect_language(text)
        self.synth_queue.put((text, lang))

    # ---------- 实际合成与播放 ----------
    def _synthesize_and_play(self, text, lang):
        # 检查缓存（完整 WAV 字节）
        with self.cache_lock:
            if text in self.audio_cache:
                wav_bytes = self.audio_cache[text]
                self.get_logger().info(f"使用缓存音频: {text[:30]}...")
                tmp_path = self._write_wav_bytes_to_temp(wav_bytes)
                self.audio_player.add_to_queue(tmp_path)
                return

        # 合成新音频
        self.get_logger().info(f"合成语音: {text[:30]}... (语言: {lang})")
        if lang not in self.voice_cache:
            model_onnx, model_json = self.get_model_path(lang)
            self.voice_cache[lang] = piper.PiperVoice.load(model_onnx, model_json)
        voice = self.voice_cache[lang]

        # 收集 PCM 数据
        audio_pcm = b""
        sample_rate = sample_width = channels = None
        for chunk in voice.synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate
                sample_width = chunk.sample_width
                channels = chunk.sample_channels
            audio_pcm += chunk.audio_int16_bytes

        # 构建完整 WAV 字节
        wav_bytes = self._build_wav_bytes(audio_pcm, sample_rate, sample_width, channels)

        # 写入缓存
        with self.cache_lock:
            if len(self.audio_cache) >= self.max_cache_size:
                oldest = next(iter(self.audio_cache))
                del self.audio_cache[oldest]
            self.audio_cache[text] = wav_bytes
            self.get_logger().debug(f"缓存: {text[:30]}... (size={len(self.audio_cache)})")

        # 写入临时文件并播放
        tmp_path = self._write_wav_bytes_to_temp(wav_bytes)
        self.audio_player.add_to_queue(tmp_path)

    # ---------- 辅助方法 ----------
    def _build_wav_bytes(self, pcm_data, sample_rate, sample_width, channels):
        """将 PCM 数据构建为完整的 WAV 文件字节"""
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data)
            return wav_io.getvalue()

    def _write_wav_bytes_to_temp(self, wav_bytes):
        """将 WAV 字节写入临时文件，返回文件路径"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        self.get_logger().debug(f"临时文件已创建: {tmp_path}")
        return tmp_path

    # ---------- 模型管理 ----------
    def detect_language(self, text):
        return "zh" if re.search(r'[\u4e00-\u9fff]', text) else "en"

    def download_model(self, lang):
        self.get_logger().info(f'下载 {lang} 语音模型...')
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ["onnx", "json"]:
            url = MODEL_URLS[lang][ext]
            local_path = MODEL_DIR / f"{lang}.{ext}"
            if local_path.exists():
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
            self.download_model(lang)
        return model_onnx, model_json

    # ---------- 回调 ----------
    def msg_callback(self, msg):
        self.synthesize(msg.data)

    def tem_callback(self, msg):
        self.tem = msg.data
        current_temp = float(self.tem)
        current_time = time.time()
        if current_temp > self.temp_threshold:
            has_recent_fire = (current_time - self.last_fire_alert_time) < self.fire_alert_window
            if not has_recent_fire and (current_time - self.last_high_temp_reminder) > self.high_temp_cooldown:
                self.last_high_temp_reminder = current_time
                self.synthesize("发现异常高温物体，请尽快前往查看")

    def alert_callback(self, msg):
        try:
            data = json.loads(msg.data)
            alert_type = data.get('type', 'unknown')
            description = data.get('description', '')
            if description.startswith('lb'):
                description = '李彪未佩戴安全帽,李彪未佩戴安全帽'
            elif description.startswith('feng'):
                description = '冯家硕未佩戴安全帽,冯家硕未佩戴安全帽'
            current_time = time.time()
            current_temp = float(self.tem)

            if alert_type == 'fire':
                self.last_fire_alert_time = current_time

            last_time = self.last_alert_time.get(alert_type, 0)
            if current_time - last_time < self.cooldown_sec:
                self.get_logger().info(f'跳过重复警报（{alert_type}）')
                return
            self.last_alert_time[alert_type] = current_time

            if alert_type in ('fire', 'smoke'):
                if current_temp > self.temp_threshold and description:
                    self.synthesize(description)
            else:
                if description:
                    self.synthesize(description)
        except Exception as e:
            self.get_logger().error(f'处理警报出错: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = AlertTtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('关闭节点...')
    finally:
        pygame.mixer.quit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
