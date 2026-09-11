# -*- coding:utf-8 -*-
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime, sleep, time
import _thread as thread
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import traceback
import pygame  # 添加pygame用于音频播放
import threading

class AudioPlayer:
    """音频播放器类，管理播放队列和状态"""
    def __init__(self, logger):
        self.logger = logger
        self.play_queue = []
        self.currently_playing = None
        self.playing_thread = None
        pygame.mixer.init()
        self.logger.info("音频播放器初始化完成")

    def add_to_queue(self, audio_file):
        """添加音频文件到播放队列"""
        self.play_queue.append(audio_file)
        self.logger.info(f"已加入播放队列: {audio_file} (队列长度: {len(self.play_queue)})")
        
        # 如果没有正在播放，立即播放
        if not self.is_playing():
            self._play_next()

    def is_playing(self):
        """检查是否有音频正在播放"""
        return self.currently_playing is not None

    def _play_next(self):
        """开始播放队列中的下一个音频"""
        if not self.play_queue or self.is_playing():
            return
            
        # 获取并移除队列中的第一个文件
        audio_file = self.play_queue.pop(0)
        self.currently_playing = audio_file
        
        # 创建播放线程
        self.playing_thread = threading.Thread(target=self._play_audio, args=(audio_file,))
        self.playing_thread.daemon = True
        self.playing_thread.start()

    def _play_audio(self, audio_file):
        """实际播放音频的方法"""
        try:
            self.logger.info(f"开始播放: {audio_file}")
            
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # 等待播放完成
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # 每100ms检查一次
                sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"播放失败: {str(e)}")
            self.logger.error(traceback.format_exc())
        finally:
            self.logger.info(f"播放完成: {audio_file}")
            self.currently_playing = None
            
            # 清理文件
            try:
                os.remove(audio_file)
                self.logger.debug(f"已清理临时文件: {audio_file}")
            except:
                self.logger.warning(f"清理文件失败: {audio_file}")
            
            # 检查是否有下一个要播放
            if self.play_queue:
                self.logger.info("开始播放下一个音频")
                self._play_next()

class Ws_Param:
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {
            "aue": "lame", 
            "auf": "audio/L16;rate=16000", 
            "vcn": "xiaoyan",
            "tte": "utf8"
        }
        # 修复文本编码
        encoded_text = base64.b64encode(self.Text.encode('utf-8')).decode('utf-8')
        self.Data = {"status": 2, "text": encoded_text}

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        
        signature_origin = f"host: ws-api.xfyun.cn\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(
            self.APISecret.encode('utf-8'),
            signature_origin.encode('utf-8'),
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode('utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode('utf-8')
        return url + '?' + urlencode({
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        })

class TTSNode(Node):
    def __init__(self):
        super().__init__('xfyun_tts_node')
        self.subscription = self.create_subscription(
            String,
            '/anomaly_detection/alerts',
            self.chat_callback,
            10
        )
        self.subscription = self.create_subscription(
            String,
            '/getMSG',
            self.deepseek_callback,
            10
        )
        self.subscription = self.create_subscription(
            String,
            '/environment/alerts',
            self.env_callback,
            10
        )
        self.get_logger().info("TTS 节点初始化完成")
        
        # 创建音频播放器
        self.audio_player = AudioPlayer(self.get_logger())
        
        # 消息历史记录字典，避免10秒内重复处理相同消息
        # 格式: {hash: (last_processed_time, file_path)}
        self.message_history = {}
        
        # 历史记录清理时间间隔（秒）
        self.history_clean_interval = 60
        self.last_clean_time = time()
        
        # 重复消息时间窗口（秒）
        self.repeat_threshold = 10
        
        # 请替换为您的实际凭证
        self.APPID = '00f3be54'
        self.APIKey = '254ef13b3cd006601bdddaa0c9cdd2ac'
        self.APISecret = 'YWRmYTE1NGNiMDQ0NzM3MTlkNDJlNjE4'

    def deepseek_callback(self, msg):
        text = msg.data
        self.get_logger().info(f'收到文本: "{text[:50]}"...')
        
        # 清理过期的历史记录
        current_time = time()
        if current_time - self.last_clean_time > self.history_clean_interval:
            self.clean_message_history()
            self.last_clean_time = current_time
        
        # 生成消息的唯一哈希（使用SHA-256）
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.get_logger().debug(f"消息哈希: {text_hash}")
        
        # 检查是否在最近10秒内处理过相同的消息
        if text_hash in self.message_history:
            last_time, file_path = self.message_history[text_hash]
            
            if current_time - last_time < self.repeat_threshold:
                self.get_logger().info(f"10秒内已收到相同消息，忽略: '{text[:20]}...'")
                return
            else:
                self.get_logger().info(f"相同消息但超过10秒间隔，重新处理: '{text[:20]}...'")
        
        try:
            # 使用线程处理合成
            thread.start_new_thread(self.synthesize_speech, (text, text_hash))
            # 记录处理时间但不记录文件路径（等合成完成再记录）
            self.message_history[text_hash] = (current_time, None)
            
        except Exception as e:
            self.get_logger().error(f"启动合成线程失败: {str(e)}")
            self.get_logger().error(traceback.format_exc())
        
    def chat_callback(self, msg):
        text = msg.data
        text = "警告！警告！" + json.loads(msg.data)['description']
        self.get_logger().info(f'收到文本: "{text[:50]}"...')
        
        # 清理过期的历史记录
        current_time = time()
        if current_time - self.last_clean_time > self.history_clean_interval:
            self.clean_message_history()
            self.last_clean_time = current_time
        
        # 生成消息的唯一哈希（使用SHA-256）
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.get_logger().debug(f"消息哈希: {text_hash}")
        
        # 检查是否在最近10秒内处理过相同的消息
        if text_hash in self.message_history:
            last_time, file_path = self.message_history[text_hash]
            
            if current_time - last_time < self.repeat_threshold:
                self.get_logger().info(f"10秒内已收到相同消息，忽略: '{text[:20]}...'")
                return
            else:
                self.get_logger().info(f"相同消息但超过10秒间隔，重新处理: '{text[:20]}...'")
        
        try:
            # 使用线程处理合成
            thread.start_new_thread(self.synthesize_speech, (text, text_hash))
            # 记录处理时间但不记录文件路径（等合成完成再记录）
            self.message_history[text_hash] = (current_time, None)
            
        except Exception as e:
            self.get_logger().error(f"启动合成线程失败: {str(e)}")
            self.get_logger().error(traceback.format_exc())
            
    def env_callback(self, msg):
        text = msg.data
        self.get_logger().info(f'收到文本: "{text[:50]}"...')
        
        # 清理过期的历史记录
        current_time = time()
        if current_time - self.last_clean_time > self.history_clean_interval:
            self.clean_message_history()
            self.last_clean_time = current_time
        
        # 生成消息的唯一哈希（使用SHA-256）
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.get_logger().debug(f"消息哈希: {text_hash}")
        
        # 检查是否在最近10秒内处理过相同的消息
        if text_hash in self.message_history:
            last_time, file_path = self.message_history[text_hash]
            
            if current_time - last_time < self.repeat_threshold:
                self.get_logger().info(f"10秒内已收到相同消息，忽略: '{text[:20]}...'")
                return
            else:
                self.get_logger().info(f"相同消息但超过10秒间隔，重新处理: '{text[:20]}...'")
        
        try:
            # 使用线程处理合成
            thread.start_new_thread(self.synthesize_speech, (text, text_hash))
            # 记录处理时间但不记录文件路径（等合成完成再记录）
            self.message_history[text_hash] = (current_time, None)
            
        except Exception as e:
            self.get_logger().error(f"启动合成线程失败: {str(e)}")
            self.get_logger().error(traceback.format_exc())

    def synthesize_speech(self, text, text_hash):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(os.path.expanduser('~'), f"tts_output_{timestamp}.mp3")
            self.get_logger().info(f"目标输出文件: {output_file}")
            
            wsParam = Ws_Param(self.APPID, self.APIKey, self.APISecret, text)
            ws_url = wsParam.create_url()
            self.get_logger().debug(f"WebSocket URL: {ws_url}")
            
            # 打开文件用于写入
            out_file = open(output_file, 'wb')
            self.get_logger().info("文件成功打开")
            
            def on_message(ws, message):
                try:
                    msg_data = json.loads(message)
                    code = msg_data.get("code", 0)
                    
                    # 错误处理
                    if code != 0:
                        err_msg = msg_data.get("message", "未知错误")
                        self.get_logger().error(f"合成失败 (code={code}): {err_msg}")
                        ws.close()
                        return
                    
                    # 检查音频数据
                    if "data" in msg_data and "audio" in msg_data["data"]:
                        audio_data = base64.b64decode(msg_data["data"]["audio"])
                        if audio_data:
                            out_file.write(audio_data)
                            self.get_logger().debug(f"写入 {len(audio_data)} 字节音频数据")
                    
                    # 检查是否完成
                    if msg_data.get("data", {}).get("status") == 2:
                        self.get_logger().info("合成完成")
                        ws.close()
                        
                except Exception as e:
                    self.get_logger().error(f"消息处理出错: {str(e)}")
                    self.get_logger().error(traceback.format_exc())

            def on_error(ws, error):
                self.get_logger().error(f"WebSocket错误: {str(error)}")
                # 从历史记录中移除失败的条目
                if text_hash in self.message_history:
                    del self.message_history[text_hash]

            def on_close(ws, status_code, close_msg):
                self.get_logger().info(f"连接关闭 (状态码={status_code}, 消息={close_msg})")
                out_file.close()
                
                # 检查文件大小
                file_size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
                self.get_logger().info(f"最终文件大小: {file_size} 字节")
                
                if file_size == 0:
                    self.get_logger().error("错误: 生成的文件为空!")
                    # 清理无效的历史记录
                    if text_hash in self.message_history:
                        del self.message_history[text_hash]
                else:
                    self.get_logger().info(f"音频文件已生成: {output_file}")
                    # 更新历史记录
                    self.message_history[text_hash] = (time(), output_file)
                    # 添加到播放队列
                    self.audio_player.add_to_queue(output_file)

            def on_open(ws):
                try:
                    data = {
                        "common": wsParam.CommonArgs,
                        "business": wsParam.BusinessArgs,
                        "data": wsParam.Data
                    }
                    self.get_logger().debug(f"发送数据: {json.dumps(data)}")
                    ws.send(json.dumps(data))
                    self.get_logger().info("连接成功，开始发送数据")
                except Exception as e:
                    self.get_logger().error(f"发送数据出错: {str(e)}")
                    self.get_logger().error(traceback.format_exc())
                    ws.close()

            # 修复ping设置问题 (ping_interval > ping_timeout)
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # 正确的ping设置
            self.get_logger().info("启动WebSocket连接...")
            ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=15,  # 间隔15秒发送一次ping
                ping_timeout=10    # 等待pong响应超时10秒
            )
            
        except Exception as e:
            self.get_logger().error(f"合成过程中出错: {str(e)}")
            self.get_logger().error(traceback.format_exc())
            # 清理无效的历史记录
            if text_hash in self.message_history:
                del self.message_history[text_hash]
            if 'out_file' in locals() and not out_file.closed:
                out_file.close()

    def clean_message_history(self):
        """清理过期消息历史记录"""
        current_time = time()
        keys_to_remove = []
        
        for text_hash, (last_time, file_path) in self.message_history.items():
            # 保留最近1小时内的记录（考虑10秒去重窗口）
            if current_time - last_time > 3600:
                keys_to_remove.append(text_hash)
                # 清理对应的音频文件
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.get_logger().debug(f"清理旧音频文件: {file_path}")
                    except:
                        pass
        
        for key in keys_to_remove:
            del self.message_history[key]
        
        self.get_logger().info(f"清理了 {len(keys_to_remove)} 条旧记录")
        self.get_logger().debug(f"当前历史记录数量: {len(self.message_history)}")

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到Ctrl+C，关闭节点...")
    except Exception as e:
        node.get_logger().error(f"节点运行错误: {str(e)}")
    finally:
        # 清理所有残留文件
        node.get_logger().info("清理残留资源...")
        for _, file_path in node.message_history.values():
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    node.get_logger().info(f"清理残留文件: {file_path}")
                except:
                    node.get_logger().warning(f"清理文件失败: {file_path}")
        
        # 关闭pygame
        pygame.mixer.quit()
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
