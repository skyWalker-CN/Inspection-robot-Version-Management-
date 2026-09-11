#!/usr/bin/env python3

import sys
import re
import json
import asyncio
import threading
import traceback
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")

from gi.repository import Gst, GstWebRTC, GstSdp, GLib

import cv2
import numpy as np
import rclpy
import websockets

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

# ============================================================
# 全局配置
# ============================================================

WIDTH = 480
HEIGHT = 270
FPS = 30
BITRATE = 4_000_000

ROS_TOPIC = "/anomaly_detection/processed"
WS_HOST = "0.0.0.0"
WS_PORT = 8080

# 调试开关
DEBUG_TIMING = True
DEBUG_ROS_FREQ = True
DEBUG_PUSH_RESULT = True

Gst.init(None)

# ============================================================
# 管线管理器：支持多浏览器
# ============================================================

class PipelineManager:
    def __init__(self):
        self.pipelines = {}          # key: websocket, value: WebRTCPipeline
        self.lock = threading.Lock()

    def add(self, websocket, pipeline):
        with self.lock:
            self.pipelines[websocket] = pipeline
            print(f"[Manager] Added pipeline for {websocket.remote_address}, total={len(self.pipelines)}")

    def remove(self, websocket):
        with self.lock:
            pipeline = self.pipelines.pop(websocket, None)
        if pipeline:
            pipeline.stop()
            print(f"[Manager] Removed pipeline for {websocket.remote_address}, total={len(self.pipelines)}")
        return pipeline

    def get_all(self):
        with self.lock:
            return list(self.pipelines.values())

    def stop_all(self):
        with self.lock:
            pipelines = list(self.pipelines.values())
            self.pipelines.clear()
        for p in pipelines:
            p.stop()
        print("[Manager] All pipelines stopped")


# ============================================================
# WebRTC 管线类（保持原有逻辑，仅适配多管线）
# ============================================================

class WebRTCPipeline:
    def __init__(self, asyncio_loop):
        self.asyncio_loop = asyncio_loop

        self.pipeline = None
        self.appsrc = None
        self.rtph264pay = None
        self.webrtc = None
        self.webrtc_sink_pad = None

        self.ws = None

        self.frame_count = 0
        self.frame_duration = Gst.SECOND // FPS

        self.remote_description_set = False
        self.local_description_set = False
        self.ready_to_push = False

        self.pending_remote_candidates = []

        self.selected_payload_type = None
        self.selected_profile_level_id = None

        self.lock = threading.Lock()

        # 调试统计
        self.push_error_count = 0
        self.push_time_total = 0.0
        self.push_time_count = 0

    # ========================================================
    # GStreamer 元素工具
    # ========================================================
    @staticmethod
    def create_element(factory, name):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"无法创建 GStreamer 元素: {factory}\n请执行: gst-inspect-1.0 {factory}")
        return element

    @staticmethod
    def set_property_if_supported(element, property_name, value):
        prop = element.find_property(property_name)
        if prop is None:
            print(f"[GStreamer] 属性不支持，跳过: {element.get_name()}.{property_name}")
            return False
        try:
            element.set_property(property_name, value)
            print(f"[GStreamer] 设置属性: {element.get_name()}.{property_name}={value}")
            return True
        except Exception as error:
            print(f"[GStreamer] 设置属性失败: {element.get_name()}.{property_name}: {error}")
            return False

    # ========================================================
    # 构建 GStreamer 管线
    # ========================================================
    def build_pipeline(self):
        self.pipeline = Gst.Pipeline.new("jetson-webrtc-pipeline")
        if self.pipeline is None:
            raise RuntimeError("无法创建 GStreamer Pipeline")

        # 元素
        self.appsrc = self.create_element("appsrc", "source")
        queue_input = self.create_element("queue", "queue_input")
        nvvidconv = self.create_element("nvvidconv", "nvvidconv")
        caps_nvmm = self.create_element("capsfilter", "caps_nvmm")
        encoder = self.create_element("nvv4l2h264enc", "h264_encoder")
        h264parse = self.create_element("h264parse", "h264parse")
        caps_h264 = self.create_element("capsfilter", "caps_h264")
        queue_rtp = self.create_element("queue", "queue_rtp")
        self.rtph264pay = self.create_element("rtph264pay", "rtph264pay")
        caps_rtp = self.create_element("capsfilter", "caps_rtp")
        self.webrtc = self.create_element("webrtcbin", "webrtc")

        # appsrc 设置
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("do-timestamp", False)
        self.appsrc.set_property("block", False)
        if self.appsrc.find_property("emit-signals") is not None:
            self.appsrc.set_property("emit-signals", False)

        appsrc_caps = Gst.Caps.from_string(
            "video/x-raw,"
            "format=(string)BGRx,"
            f"width=(int){WIDTH},"
            f"height=(int){HEIGHT},"
            f"framerate=(fraction){FPS}/1"
        )
        self.appsrc.set_property("caps", appsrc_caps)

        # 队列设置（保持原样）
        queue_input.set_property("max-size-buffers", 2)
        queue_input.set_property("max-size-bytes", 0)
        queue_input.set_property("max-size-time", 0)
        queue_input.set_property("leaky", 2)

        queue_rtp.set_property("max-size-buffers", 10)
        queue_rtp.set_property("max-size-bytes", 0)
        queue_rtp.set_property("max-size-time", 0)

        # NVMM 转换
        caps_nvmm.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),"
                "format=(string)NV12,"
                f"width=(int){WIDTH},"
                f"height=(int){HEIGHT},"
                f"framerate=(fraction){FPS}/1"
            ),
        )

        # 编码器
        self.set_property_if_supported(encoder, "bitrate", BITRATE)
        self.set_property_if_supported(encoder, "iframeinterval", FPS)
        self.set_property_if_supported(encoder, "idrinterval", FPS)
        self.set_property_if_supported(encoder, "insert-sps-pps", True)
        self.set_property_if_supported(encoder, "maxperf-enable", True)
        self.set_property_if_supported(encoder, "profile", 0)
        self.set_property_if_supported(encoder, "preset-level", 1)

        # H264 parser
        h264parse.set_property("config-interval", -1)
        caps_h264.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-h264,"
                "stream-format=(string)byte-stream,"
                "alignment=(string)au"
            ),
        )

        # RTP payloader
        self.rtph264pay.set_property("pt", 96)
        self.rtph264pay.set_property("config-interval", -1)
        self.rtph264pay.set_property("mtu", 1200)
        if self.rtph264pay.find_property("aggregate-mode") is not None:
            try:
                self.rtph264pay.set_property("aggregate-mode", "zero-latency")
            except Exception as error:
                print("[RTP] 设置 aggregate-mode 失败，跳过:", error)

        caps_rtp.set_property(
            "caps",
            Gst.Caps.from_string(
                "application/x-rtp,"
                "media=(string)video,"
                "encoding-name=(string)H264,"
                "clock-rate=(int)90000,"
                "packetization-mode=(string)1"
            ),
        )

        # webrtcbin
        self.webrtc.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self.webrtc.set_property("stun-server", "stun://stun.l.google.com:19302")

        # 加入管道
        elements = [
            self.appsrc, queue_input, nvvidconv, caps_nvmm, encoder,
            h264parse, caps_h264, queue_rtp, self.rtph264pay, caps_rtp, self.webrtc,
        ]
        for element in elements:
            self.pipeline.add(element)

        # 连接
        normal_links = [
            (self.appsrc, queue_input),
            (queue_input, nvvidconv),
            (nvvidconv, caps_nvmm),
            (caps_nvmm, encoder),
            (encoder, h264parse),
            (h264parse, caps_h264),
            (caps_h264, queue_rtp),
            (queue_rtp, self.rtph264pay),
            (self.rtph264pay, caps_rtp),
        ]
        for source, destination in normal_links:
            if not source.link(destination):
                raise RuntimeError(f"元素连接失败: {source.get_name()} -> {destination.get_name()}")

        # RTP capsfilter -> webrtcbin
        rtp_src_pad = caps_rtp.get_static_pad("src")
        try:
            self.webrtc_sink_pad = self.webrtc.request_pad_simple("sink_%u")
        except AttributeError:
            self.webrtc_sink_pad = self.webrtc.get_request_pad("sink_%u")
        if self.webrtc_sink_pad is None:
            raise RuntimeError("无法获取 webrtcbin sink_%u pad")
        if rtp_src_pad.link(self.webrtc_sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("RTP -> webrtcbin pad 连接失败")
        print(f"[Pipeline] RTP connected to {self.webrtc_sink_pad.get_name()}")

        # 信号
        self.webrtc.connect("on-ice-candidate", self.on_local_ice_candidate)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

        # 进入 PLAYING
        state_result = self.pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline 无法进入 PLAYING 状态")
        print("[Pipeline] Pipeline is PLAYING, waiting for browser offer")

    # ========================================================
    # Bus 回调
    # ========================================================
    def on_bus_message(self, bus, message):
        message_type = message.type
        if message_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print("\n========== GStreamer ERROR ==========")
            print("Source:", message.src.get_name())
            print("Error :", error)
            print("Debug :", debug)
            print("=====================================\n")
        elif message_type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print("\n========== GStreamer WARNING ==========")
            print("Source :", message.src.get_name())
            print("Warning:", warning)
            print("Debug  :", debug)
            print("=======================================\n")
        elif message_type == Gst.MessageType.EOS:
            print("[Pipeline] EOS")
        elif message_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, _ = message.parse_state_changed()
                print(f"[Pipeline] State: {old_state.value_nick} -> {new_state.value_nick}")

    # ========================================================
    # SDP 解析和协商（保持原逻辑）
    # ========================================================
    @staticmethod
    def find_h264_payload_type(sdp_text):
        rtpmap_matches = re.findall(
            r"^a=rtpmap:(\d+)\s+H264/90000\s*$",
            sdp_text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if not rtpmap_matches:
            raise RuntimeError("浏览器 offer 中没有 H264 编码器。")
        candidates = []
        for payload_string in rtpmap_matches:
            payload_type = int(payload_string)
            fmtp_match = re.search(rf"^a=fmtp:{payload_type}\s+(.+)$", sdp_text, flags=re.MULTILINE | re.IGNORECASE)
            fmtp = fmtp_match.group(1) if fmtp_match else ""
            packetization_match = re.search(r"packetization-mode=(\d+)", fmtp, flags=re.IGNORECASE)
            packetization_mode = packetization_match.group(1) if packetization_match else "0"
            profile_match = re.search(r"profile-level-id=([0-9a-fA-F]+)", fmtp, flags=re.IGNORECASE)
            profile_level_id = profile_match.group(1).lower() if profile_match else ""
            print(f"[SDP] H264 candidate: PT={payload_type}, packetization-mode={packetization_mode}, profile-level-id={profile_level_id or 'none'}")
            if packetization_mode != "1":
                continue
            if profile_level_id.startswith("4200"):
                priority = 0
            elif profile_level_id.startswith("42e0") or profile_level_id.startswith("42c0"):
                priority = 1
            else:
                priority = 2
            candidates.append((priority, payload_type, profile_level_id))
        if not candidates:
            raise RuntimeError("浏览器 offer 中没有 packetization-mode=1 的 H264。")
        candidates.sort(key=lambda item: item[0])
        _, payload_type, profile_level_id = candidates[0]
        return payload_type, profile_level_id

    def parse_sdp(self, sdp_text):
        result, sdp_message = GstSdp.SDPMessage.new()
        if result != GstSdp.SDPResult.OK:
            raise RuntimeError(f"创建 SDPMessage 失败: {result}")
        result = GstSdp.sdp_message_parse_buffer(sdp_text.encode("utf-8"), sdp_message)
        if result != GstSdp.SDPResult.OK:
            raise RuntimeError(f"解析 SDP 失败: {result}")
        return sdp_message

    async def handle_offer(self, offer_sdp):
        try:
            payload_type, profile_level_id = self.find_h264_payload_type(offer_sdp)
            self.selected_payload_type = payload_type
            self.selected_profile_level_id = profile_level_id
            print(f"[SDP] Selected H264: PT={payload_type}, profile-level-id={profile_level_id}")
            self.rtph264pay.set_property("pt", payload_type)
            GLib.idle_add(self._set_remote_offer, offer_sdp)
        except Exception as error:
            print(f"[SDP] handle_offer error: {type(error).__name__}: {error}")
            traceback.print_exc()

    def _set_remote_offer(self, offer_sdp):
        try:
            print("[SDP] Setting remote offer...")
            sdp_message = self.parse_sdp(offer_sdp)
            offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdp_message)
            promise = Gst.Promise.new_with_change_func(self._on_remote_description_set, None, None)
            self.webrtc.emit("set-remote-description", offer, promise)
        except Exception as error:
            print(f"[SDP] _set_remote_offer error: {type(error).__name__}: {error}")
            traceback.print_exc()
        return False

    def _on_remote_description_set(self, promise, *unused):
        try:
            result = promise.wait()
            print(f"[SDP] set-remote-description: {result}")
            if result != Gst.PromiseResult.REPLIED:
                print("[SDP] set-remote-description failed")
                return
            self.remote_description_set = True
            print("[SDP] Remote offer set successfully")
            pending = self.pending_remote_candidates
            self.pending_remote_candidates = []
            for mlineindex, candidate in pending:
                self.webrtc.emit("add-ice-candidate", int(mlineindex), candidate)
            answer_promise = Gst.Promise.new_with_change_func(self._on_answer_created, None, None)
            self.webrtc.emit("create-answer", None, answer_promise)
        except Exception as error:
            print(f"[SDP] remote description callback error: {type(error).__name__}: {error}")
            traceback.print_exc()

    def _on_answer_created(self, promise, *unused):
        try:
            result = promise.wait()
            print(f"[SDP] create-answer: {result}")
            if result != Gst.PromiseResult.REPLIED:
                print("[SDP] create-answer failed")
                return
            reply = promise.get_reply()
            if reply is None:
                print("[SDP] create-answer reply is None")
                return
            answer = reply.get_value("answer")
            if answer is None:
                print("[SDP] create-answer did not return answer")
                print("[SDP] Reply:", reply.to_string())
                return
            answer_sdp = answer.sdp.as_text()
            self.pending_answer = answer
            self.pending_answer_sdp = answer_sdp
            local_promise = Gst.Promise.new_with_change_func(self._on_local_description_set, None, None)
            self.webrtc.emit("set-local-description", answer, local_promise)
        except Exception as error:
            print(f"[SDP] answer callback error: {type(error).__name__}: {error}")
            traceback.print_exc()

    def _on_local_description_set(self, promise, *unused):
        try:
            result = promise.wait()
            print(f"[SDP] set-local-description: {result}")
            if result != Gst.PromiseResult.REPLIED:
                print("[SDP] set-local-description failed")
                return
            self.local_description_set = True
            self.ready_to_push = True
            with self.lock:
                self.frame_count = 0
            print("[SDP] Local answer set; video pushing enabled")
            asyncio.run_coroutine_threadsafe(
                self.send_sdp(self.pending_answer_sdp, "answer"),
                self.asyncio_loop,
            )
        except Exception as error:
            print(f"[SDP] local description callback error: {type(error).__name__}: {error}")
            traceback.print_exc()

    async def send_sdp(self, sdp_text, sdp_type):
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps({"type": sdp_type, "sdp": sdp_text}))
            print(f"[WS] Sent SDP {sdp_type}")
        except Exception as error:
            print(f"[WS] Failed to send SDP: {error}")

    # ========================================================
    # ICE
    # ========================================================
    def on_local_ice_candidate(self, webrtc, mlineindex, candidate):
        if not candidate:
            return
        future = asyncio.run_coroutine_threadsafe(
            self.send_local_ice_candidate(mlineindex, candidate),
            self.asyncio_loop,
        )
        def completed(result_future):
            try:
                result_future.result()
            except Exception as error:
                print(f"[ICE] Failed to send local candidate: {error}")
        future.add_done_callback(completed)

    async def send_local_ice_candidate(self, mlineindex, candidate):
        if self.ws is None:
            return
        message = {
            "type": "candidate",
            "candidate": candidate,
            "sdpMLineIndex": int(mlineindex),
            "sdpMid": "0",
        }
        try:
            await self.ws.send(json.dumps(message))
            print(f"[ICE] Sent local candidate, mline={mlineindex}")
        except Exception as error:
            print(f"[ICE] Candidate send failed: {error}")

    def add_remote_ice_candidate(self, mlineindex, candidate):
        if not candidate:
            return False
        try:
            if not self.remote_description_set:
                print("[ICE] Remote description not ready; queueing browser candidate")
                self.pending_remote_candidates.append((int(mlineindex), candidate))
                return False
            self.webrtc.emit("add-ice-candidate", int(mlineindex), candidate)
            print(f"[ICE] Added browser candidate, mline={mlineindex}")
        except Exception as error:
            print(f"[ICE] Failed to add browser candidate: {error}")
        return False

    # ========================================================
    # 推送图像
    # ========================================================
    def push_frame(self, bgr_frame):
        if self.appsrc is None or not self.ready_to_push:
            return
        if bgr_frame is None or bgr_frame.size == 0:
            return

        start_time = time.perf_counter()
        try:
            # 缩放
            if bgr_frame.shape[1] != WIDTH or bgr_frame.shape[0] != HEIGHT:
                bgr_frame = cv2.resize(bgr_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            bgrx_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
            bgrx_frame = np.ascontiguousarray(bgrx_frame, dtype=np.uint8)
            expected_size = WIDTH * HEIGHT * 4
            if bgrx_frame.nbytes != expected_size:
                print(f"[AppSrc] Invalid buffer size: {bgrx_frame.nbytes}, expected={expected_size}")
                return
            data = bgrx_frame.tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)

            with self.lock:
                frame_number = self.frame_count
                self.frame_count += 1

            pts = frame_number * self.frame_duration
            buffer.pts = pts
            buffer.dts = pts
            buffer.duration = self.frame_duration
            buffer.offset = frame_number
            buffer.offset_end = frame_number + 1

            result = self.appsrc.emit("push-buffer", buffer)
            if result != Gst.FlowReturn.OK:
                self.push_error_count += 1
                if DEBUG_PUSH_RESULT:
                    print(f"[AppSrc] push-buffer result: {result} (frame {frame_number})")
            if frame_number == 0:
                print(f"[AppSrc] First frame pushed: {WIDTH}x{HEIGHT}, bytes={len(data)}")

            elapsed = time.perf_counter() - start_time
            self.push_time_total += elapsed
            self.push_time_count += 1
            if DEBUG_TIMING and frame_number % 30 == 0:
                avg = self.push_time_total / max(1, self.push_time_count)
                print(f"[Timing] Frame {frame_number}: last={elapsed*1000:.2f} ms, avg={avg*1000:.2f} ms, errors={self.push_error_count}")
        except Exception as error:
            print(f"[AppSrc] push_frame error: {type(error).__name__}: {error}")
            traceback.print_exc()

    def stop(self):
        self.ready_to_push = False
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            print("[Pipeline] Stopped")


# ============================================================
# ROS 2 Node：向所有管线广播图像
# ============================================================
class ROSBridge(Node):
    def __init__(self, pipeline_manager):
        super().__init__("webrtc_bridge")
        self.pipeline_manager = pipeline_manager
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            RosImage, ROS_TOPIC, self.image_callback, 10)
        self.received_frames = 0
        self.last_callback_time = time.perf_counter()

    def image_callback(self, message):
        try:
            now = time.perf_counter()
            interval = now - self.last_callback_time
            self.last_callback_time = now
            if DEBUG_ROS_FREQ and self.received_frames % 30 == 0:
                fps = 30 / interval if interval > 0 else 0
                self.get_logger().info(f"ROS callback FPS (30 frames): {fps:.2f}")

            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            if image is None or image.size == 0:
                self.get_logger().warning("Received an empty image")
                return
            if image.shape[1] != WIDTH or image.shape[0] != HEIGHT:
                image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            self.received_frames += 1
            if self.received_frames == 1:
                self.get_logger().info(f"First ROS image: {image.shape[1]}x{image.shape[0]}")

            # 向所有活跃管线推送
            for pipeline in self.pipeline_manager.get_all():
                pipeline.push_frame(image)

        except Exception as error:
            self.get_logger().warning(f"Image callback error: {type(error).__name__}: {error}")


# ============================================================
# WebSocket 信令：为每个连接创建独立管线
# ============================================================
async def signaling_handler(websocket, *args):
    global pipeline_manager
    print("[WS] Browser connected")

    # 为当前连接创建新管线
    asyncio_loop = asyncio.get_running_loop()
    pipeline = WebRTCPipeline(asyncio_loop)
    pipeline.build_pipeline()
    pipeline.ws = websocket

    pipeline_manager.add(websocket, pipeline)

    try:
        async for raw_message in websocket:
            try:
                data = json.loads(raw_message)
                message_type = data.get("type")
                print(f"[WS] Received: {message_type}")

                if message_type == "offer":
                    offer_sdp = data.get("sdp")
                    if not offer_sdp:
                        raise ValueError("Offer message has no SDP")
                    await pipeline.handle_offer(offer_sdp)

                elif message_type == "candidate":
                    candidate = data.get("candidate")
                    mlineindex = data.get("sdpMLineIndex", 0)
                    GLib.idle_add(
                        pipeline.add_remote_ice_candidate,
                        int(mlineindex),
                        candidate,
                    )

                else:
                    print(f"[WS] Ignoring unknown message: {message_type}")

            except Exception as error:
                print(f"[WS] Message processing error: {type(error).__name__}: {error}")
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosed as error:
        print(f"[WS] Browser disconnected: code={error.code}, reason={error.reason}")
    except Exception as error:
        print(f"[WS] Handler error: {type(error).__name__}: {error}")
    finally:
        pipeline_manager.remove(websocket)
        print("[WS] Pipeline removed for this connection")


# ============================================================
# Main
# ============================================================
async def async_main():
    global pipeline_manager
    asyncio_loop = asyncio.get_running_loop()

    # 启动 GLib 主循环
    glib_loop = GLib.MainLoop()
    glib_thread = threading.Thread(target=glib_loop.run, daemon=True, name="glib-main-loop")
    glib_thread.start()
    print("[Main] GLib MainLoop started")

    # 创建管线管理器
    pipeline_manager = PipelineManager()

    # 初始化 ROS
    rclpy.init()
    ros_node = ROSBridge(pipeline_manager)
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True, name="ros2-spin")
    ros_thread.start()
    print("[Main] ROS 2 node started")

    # 启动 WebSocket 服务器
    try:
        async with websockets.serve(
            signaling_handler,
            WS_HOST,
            WS_PORT,
            ping_interval=20,
            ping_timeout=20,
            max_size=2 * 1024 * 1024,
        ):
            print(f"[WS] Signaling server running on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()
    finally:
        print("[Main] Shutting down...")
        pipeline_manager.stop_all()
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        glib_loop.quit()

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[Main] Interrupted")
    except Exception as error:
        print(f"[Main] Fatal error: {type(error).__name__}: {error}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
