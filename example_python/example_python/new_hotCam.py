#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import json
import asyncio
import threading
import traceback
import time
import struct
import datetime
import queue

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")

from gi.repository import Gst, GstWebRTC, GstSdp, GLib

import cv2
import numpy as np
import usb.core
import usb.util
import rclpy
import websockets

from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage

# ==================== 热像仪设备参数 ====================
VID = 0x2bdf
PID = 0x0102
UNIT_ID = 0x0A
VC_INTERFACE = 0

CS_ID_SYSTEM        = 0x01
CS_ID_IMAGE         = 0x02
CS_ID_THERMAL       = 0x03
CS_ID_PROTOCOL_VER  = 0x04
CS_ID_COMMAND_SWITCH = 0x05
CS_ID_ERROR_CODE    = 0x06

SYSTEM_LOCALTIME     = 0x05
THERMAL_BASIC_PARAM  = 0x01
THERMAL_STREAM_PARAM = 0x05
IMAGE_ENHANCEMENT    = 0x05

STREAM_TYPE_TEMP_YUV = 8

THERM_WIDTH  = 256
THERM_HEIGHT = 192
TEMP_SIZE = THERM_WIDTH * THERM_HEIGHT * 2
FRAME_SIZE = TEMP_SIZE + THERM_WIDTH * THERM_HEIGHT * 2

# ==================== WebRTC 配置（已降低以减少带宽压力） ====================
VIDEO_WIDTH = 320    # 推流分辨率
VIDEO_HEIGHT = 180
FPS = 15
BITRATE = 2_000_000
WS_HOST = "0.0.0.0"
WS_PORT = 8081

DEBUG_TIMING = False
DEBUG_ROS_FREQ = False
DEBUG_PUSH_RESULT = False

# ==================== 优化参数 ====================
MIN_PROCESS_INTERVAL = 0.1          # 最小处理间隔（秒）
ROS_IMAGE_PUBLISH_INTERVAL = 5      # 每隔多少帧发布一次图像
QUEUE_MAX_SIZE = 10
USB_ERROR_THRESHOLD = 5             # 连续 USB 错误次数阈值，超过后重置设备

Gst.init(None)

# ==================== UVC XU 通信函数（原样保留） ====================
def uvc_xu_ctrl(dev, bmRequestType, bRequest, cs_id, data_or_len, timeout=2000):
    wValue = (cs_id << 8) | 0x00
    wIndex = (UNIT_ID << 8) | VC_INTERFACE
    return dev.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, data_or_len, timeout)

def get_len(dev, cs_id, retries=5):
    for _ in range(retries):
        try:
            ret = uvc_xu_ctrl(dev, 0xA1, 0x85, cs_id, 2)
            if len(ret) >= 2:
                length = ret[0] | (ret[1] << 8)
                if 0 < length < 4096:
                    return length
        except usb.core.USBError:
            pass
        time.sleep(0.1)
    return -1

def set_cur(dev, cs_id, data, retries=5):
    for _ in range(retries):
        try:
            transferred = uvc_xu_ctrl(dev, 0x21, 0x01, cs_id, data, timeout=2000)
            if transferred == len(data):
                return True
        except usb.core.USBError:
            pass
        time.sleep(0.1)
    return False

def get_cur(dev, cs_id, length, retries=5):
    if length <= 0:
        return None
    for _ in range(retries):
        try:
            data = uvc_xu_ctrl(dev, 0xA1, 0x81, cs_id, length)
            if len(data) == length:
                return data
        except usb.core.USBError:
            pass
        time.sleep(0.1)
    return None

def switch_function(dev, target_cs_id, sub_id=0):
    length = get_len(dev, CS_ID_COMMAND_SWITCH)
    if length != 2:
        return False
    data = bytes([target_cs_id, sub_id])
    success = set_cur(dev, CS_ID_COMMAND_SWITCH, data)
    if success:
        wait_cmd_done(dev)
    return success

def wait_cmd_done(dev, after_ms=50, repeat_ms=50, timeout=3.0):
    time.sleep(after_ms / 1000.0)
    start_time = time.time()
    while time.time() - start_time < timeout:
        length = get_len(dev, CS_ID_ERROR_CODE, retries=1)
        if length > 0:
            data = get_cur(dev, CS_ID_ERROR_CODE, length, retries=1)
            if data and data[0] != 1:
                return data[0]
        time.sleep(repeat_ms / 1000.0)
    return -1

def set_curr_data(dev, cs_id, payload):
    length = get_len(dev, cs_id)
    if length < 0:
        return False
    if len(payload) < length:
        payload = bytes(payload) + b'\x00' * (length - len(payload))
    else:
        payload = bytes(payload[:length])
    return set_cur(dev, cs_id, payload)

def get_curr_data(dev, cs_id):
    length = get_len(dev, cs_id)
    if length < 0:
        return None
    return get_cur(dev, cs_id, length)

def calibrate_time(dev):
    try:
        if not switch_function(dev, CS_ID_SYSTEM, SYSTEM_LOCALTIME):
            return False
        now = datetime.datetime.now()
        msec = now.microsecond // 1000
        data = struct.pack('<HBBBBBHB', msec, now.second, now.minute, now.hour,
                           now.day, now.month, now.year, 0)
        if set_curr_data(dev, CS_ID_SYSTEM, data):
            wait_cmd_done(dev, 100, 100)
            print("[OK] Time calibrated")
            return True
    except Exception as e:
        print(f"[FAIL] Time calibration: {e}")
    return False

def thermal_base_config(dev):
    try:
        if not switch_function(dev, CS_ID_THERMAL, THERMAL_BASIC_PARAM):
            return False
        data = get_curr_data(dev, CS_ID_THERMAL)
        if data is None:
            return False
        data = bytearray(data)
        if len(data) > 31 and data[6] != 2:
            data[0], data[1] = 1, 1
            data[6] = 2
            data[31] = 1
            if set_curr_data(dev, CS_ID_THERMAL, data):
                wait_cmd_done(dev, 1000, 1000)
                print("[OK] Thermal base config done")
                return True
        else:
            print("[OK] Thermal base config already correct")
            return True
    except Exception as e:
        print(f"[FAIL] Thermal base config: {e}")
    return False

def image_enhance_config(dev):
    try:
        if not switch_function(dev, CS_ID_IMAGE, IMAGE_ENHANCEMENT):
            return False
        data = get_curr_data(dev, CS_ID_IMAGE)
        if data is None:
            return False
        data = bytearray(data)
        if len(data) > 5 and data[5] != 13:
            data[0] = 1
            data[5] = 13
            if set_curr_data(dev, CS_ID_IMAGE, data):
                wait_cmd_done(dev, 100, 100)
                print("[OK] Image enhance config done")
                return True
        else:
            print("[OK] Image enhance already correct")
            return True
    except Exception as e:
        print(f"[FAIL] Image enhance config: {e}")
    return False

def stream_type_config(dev, stream_type):
    try:
        if not switch_function(dev, CS_ID_THERMAL, THERMAL_STREAM_PARAM):
            return False
        data = get_curr_data(dev, CS_ID_THERMAL)
        if data is None:
            return False
        data = bytearray(data)
        current = data[1] if len(data) > 1 else -1
        print(f"[INFO] Current stream type: {current}, target: {stream_type}")
        if current != stream_type:
            data[1] = stream_type
            if set_curr_data(dev, CS_ID_THERMAL, data):
                wait_cmd_done(dev, 100, 100)
                print(f"[OK] Stream type set to {stream_type}")
                return True
        else:
            print(f"[OK] Stream type already {stream_type}")
            return True
    except Exception as e:
        print(f"[FAIL] Stream type config: {e}")
    return False

def find_vs_endpoint(dev):
    cfg = dev.get_active_configuration()
    intf = usb.util.find_descriptor(cfg, bInterfaceClass=0x0E, bInterfaceSubClass=0x02)
    if intf is None:
        raise RuntimeError("VS interface not found")
    ep = usb.util.find_descriptor(intf, custom_match=lambda e: e.bEndpointAddress & 0x80)
    if ep is None:
        raise RuntimeError("IN endpoint not found")
    return ep, intf.bInterfaceNumber


# ==================== GStreamer WebRTC 管线（硬编码） ====================
class PipelineManager:
    def __init__(self):
        self.pipelines = {}
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

        self.push_error_count = 0
        self.push_time_total = 0.0
        self.push_time_count = 0

    @staticmethod
    def create_element(factory, name):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"无法创建 GStreamer 元素: {factory}")
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
        self.appsrc.set_property("max-bytes", VIDEO_WIDTH * VIDEO_HEIGHT * 4 * 2)
        self.appsrc.set_property("max-buffers", 2)

        appsrc_caps = Gst.Caps.from_string(
            "video/x-raw,"
            "format=(string)BGRx,"
            f"width=(int){VIDEO_WIDTH},"
            f"height=(int){VIDEO_HEIGHT},"
            f"framerate=(fraction){FPS}/1"
        )
        self.appsrc.set_property("caps", appsrc_caps)

        # 队列
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
                f"width=(int){VIDEO_WIDTH},"
                f"height=(int){VIDEO_HEIGHT},"
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
        self.set_property_if_supported(encoder, "low-latency", True)
        self.set_property_if_supported(encoder, "EnableTwopassCBR", False)

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
            except Exception:
                pass

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
        self.webrtc_sink_pad = self.webrtc.request_pad_simple("sink_%u")
        if self.webrtc_sink_pad is None:
            raise RuntimeError("无法获取 webrtcbin sink_%u pad")
        if rtp_src_pad.link(self.webrtc_sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("RTP -> webrtcbin pad 连接失败")

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

    def push_frame(self, bgr_frame):
        if self.appsrc is None or not self.ready_to_push:
            return
        if bgr_frame is None or bgr_frame.size == 0:
            return

        start_time = time.perf_counter()
        try:
            if bgr_frame.shape[1] != VIDEO_WIDTH or bgr_frame.shape[0] != VIDEO_HEIGHT:
                bgr_frame = cv2.resize(bgr_frame, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LINEAR)
            bgrx_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA)
            bgrx_frame = np.ascontiguousarray(bgrx_frame, dtype=np.uint8)
            expected_size = VIDEO_WIDTH * VIDEO_HEIGHT * 4
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
                print(f"[AppSrc] First frame pushed: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, bytes={len(data)}")

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


# ==================== 热像仪采集与 ROS 发布类 ====================
class ThermalCamera:
    def __init__(self, pipeline_manager):
        self.pipeline_manager = pipeline_manager
        self.smooth_min = None
        self.smooth_max = None
        self.SMOOTH_ALPHA_UP = 0.6
        self.SMOOTH_ALPHA_DOWN = 0.05
        self.MIN_SPAN = 4.0
        self._stop_event = threading.Event()
        self.frame_count = 0
        self.ros_pub_counter = 0
        self.frame_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.usb_error_count = 0

        rclpy.init()
        self.ros_node = Node("thermal_camera_node")
        self.string_pub = self.ros_node.create_publisher(String, "hotCam", 10)
        self.image_pub = self.ros_node.create_publisher(RosImage, "thermal_camera/image_raw", 10)
        self.temp_pub = self.ros_node.create_publisher(RosImage, "thermal_camera/temp_raw", 10)
        self.bridge = CvBridge()

        self.init_device()

        self.capture_thread = threading.Thread(target=self.usb_capture_loop, daemon=True)
        self.capture_thread.start()

        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def init_device(self):
        dev = usb.core.find(idVendor=VID, idProduct=PID)
        if dev is None:
            raise RuntimeError("Device not found")
        self.dev = dev

        if dev.is_kernel_driver_active(VC_INTERFACE):
            dev.detach_kernel_driver(VC_INTERFACE)

        ver_len = get_len(dev, CS_ID_PROTOCOL_VER)
        if ver_len > 0:
            ver_data = get_cur(dev, CS_ID_PROTOCOL_VER, ver_len)
            if ver_data:
                print(f"[INFO] Protocol: {ver_data.tobytes().decode('ascii').strip(chr(0))}")

        calibrate_time(dev)
        thermal_base_config(dev)
        image_enhance_config(dev)

        if not stream_type_config(dev, STREAM_TYPE_TEMP_YUV):
            raise RuntimeError("Stream type NOT set to TEMP_YUV.")

        time.sleep(0.5)
        print(f"[INFO] Frame size: {FRAME_SIZE} bytes, Res: {THERM_WIDTH}x{THERM_HEIGHT}")

        ep, intf_num = find_vs_endpoint(dev)
        if dev.is_kernel_driver_active(intf_num):
            dev.detach_kernel_driver(intf_num)

        # 不强制设置 altsetting，保持默认
        self.ep = ep
        self.usb_error_count = 0

    def reinit_device(self):
        """重新初始化 USB 设备"""
        print("[USB] Reinitializing device...")
        try:
            usb.util.dispose_resources(self.dev)
        except Exception:
            pass
        time.sleep(1.0)
        try:
            self.init_device()
            print("[USB] Device reinitialized successfully")
            return True
        except Exception as e:
            print(f"[USB] Reinit failed: {e}")
            return False

    def usb_capture_loop(self):
        frame_buffer = bytearray()
        current_fid = None
        last_put_time = time.monotonic()

        print("[INFO] FID-based frame sync started.")

        while not self._stop_event.is_set():
            try:
                raw_data = self.ep.read(16384, timeout=1000)
                chunk = raw_data.tobytes() if hasattr(raw_data, 'tobytes') else bytes(raw_data)
                if len(chunk) < 2:
                    continue

                # 成功读取后重置错误计数
                self.usb_error_count = 0

                idx = 0
                while idx < len(chunk) - 1:
                    hdr_len = chunk[idx]
                    if hdr_len not in (2, 12):
                        idx += 1
                        continue
                    if idx + hdr_len > len(chunk):
                        break
                    bm_info = chunk[idx + 1]
                    if not (bm_info & 0x80):
                        idx += 1
                        continue

                    fid = bm_info & 0x01
                    payload = chunk[idx + hdr_len:]

                    if current_fid is None:
                        current_fid = fid
                        frame_buffer.extend(payload)
                    elif fid != current_fid:
                        now = time.monotonic()
                        if len(frame_buffer) >= FRAME_SIZE and (now - last_put_time) >= MIN_PROCESS_INTERVAL:
                            try:
                                self.frame_queue.put_nowait(bytes(frame_buffer))
                                last_put_time = now
                            except queue.Full:
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put_nowait(bytes(frame_buffer))
                                    last_put_time = now
                                except queue.Empty:
                                    pass
                        frame_buffer = bytearray(payload)
                        current_fid = fid
                    else:
                        frame_buffer.extend(payload)

                    idx = len(chunk)
                    break

            except usb.core.USBTimeoutError:
                continue
            except usb.core.USBError as e:
                self.usb_error_count += 1
                print(f"[USB] Read error: {e} (count={self.usb_error_count})")
                try:
                    self.ep.clear_halt()
                    print("[USB] Endpoint halt cleared")
                except Exception as clear_err:
                    print(f"[USB] Failed to clear halt: {clear_err}")

                frame_buffer.clear()
                current_fid = None

                if self.usb_error_count >= USB_ERROR_THRESHOLD:
                    print("[USB] Too many errors, attempting device reset...")
                    if self.reinit_device():
                        self.usb_error_count = 0
                        print("[USB] Device reset successful, resuming capture.")
                    else:
                        print("[USB] Device reset failed, will retry after delay.")
                time.sleep(0.5)
                continue
            except KeyboardInterrupt:
                break
        print("[INFO] USB capture loop exiting")

    def processing_loop(self):
        print("[INFO] Processing loop started.")
        while not self._stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self.process_frame(frame_data)
            except Exception as e:
                print(f"[Processing] Error processing frame: {e}")
                traceback.print_exc()
            finally:
                self.frame_queue.task_done()
        print("[INFO] Processing loop exiting")

    def process_frame(self, frame_data):
        if len(frame_data) < TEMP_SIZE:
            print(f"[WARN] Frame too short ({len(frame_data)} bytes)")
            return

        temp_raw = np.frombuffer(frame_data[:TEMP_SIZE], dtype=np.uint16).reshape(THERM_HEIGHT, THERM_WIDTH)
        temp_c = (temp_raw.astype(np.float32) / 64.0) - 50.0

        temps = temp_c.flatten()
        valid = temps[(temps > -25) & (temps < 560)]
        if len(valid) == 0:
            return

        t_min = np.percentile(valid, 2)
        t_max_real = np.max(valid)

        raw_min = t_min
        raw_max = t_max_real
        if raw_max - raw_min < self.MIN_SPAN:
            mid = (raw_max + raw_min) / 2
            raw_min = mid - self.MIN_SPAN / 2
            raw_max = mid + self.MIN_SPAN / 2

        if self.smooth_min is None or self.smooth_max is None:
            self.smooth_min = raw_min
            self.smooth_max = raw_max
        else:
            self.smooth_min = 0.2 * raw_min + 0.8 * self.smooth_min
            if raw_max > self.smooth_max:
                self.smooth_max = self.SMOOTH_ALPHA_UP * raw_max + (1 - self.SMOOTH_ALPHA_UP) * self.smooth_max
            else:
                self.smooth_max = self.SMOOTH_ALPHA_DOWN * raw_max + (1 - self.SMOOTH_ALPHA_DOWN) * self.smooth_max

        pipelines = self.pipeline_manager.get_all()
        need_webrtc = len(pipelines) > 0
        need_ros_image = self.image_pub.get_subscription_count() > 0
        need_ros_temp = self.temp_pub.get_subscription_count() > 0

        self.frame_count += 1
        self.ros_pub_counter += 1

        norm = None
        if need_webrtc or need_ros_image:
            norm = ((temp_c - self.smooth_min) / (self.smooth_max - self.smooth_min) * 255.0)
            norm = np.clip(norm, 0, 255).astype(np.uint8)

        if need_webrtc:
            color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            color_big = cv2.resize(color, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_LINEAR)
            cv2.putText(color_big, f"Max: {t_max_real:.1f}C", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for pipeline in pipelines:
                pipeline.push_frame(color_big)

        msg = String()
        msg.data = f"{t_min:.1f} {t_max_real:.1f}"
        self.string_pub.publish(msg)

        if need_ros_image and (self.ros_pub_counter % ROS_IMAGE_PUBLISH_INTERVAL == 0):
            ros_gray = self.bridge.cv2_to_imgmsg(norm, encoding="mono8")
            ros_gray.header.stamp = self.ros_node.get_clock().now().to_msg()
            ros_gray.header.frame_id = "thermal"
            self.image_pub.publish(ros_gray)

        if need_ros_temp and (self.ros_pub_counter % ROS_IMAGE_PUBLISH_INTERVAL == 0):
            ros_temp = self.bridge.cv2_to_imgmsg(temp_raw, encoding="mono16")
            ros_temp.header.stamp = self.ros_node.get_clock().now().to_msg()
            ros_temp.header.frame_id = "thermal"
            self.temp_pub.publish(ros_temp)

    def stop(self):
        self._stop_event.set()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        usb.util.dispose_resources(self.dev)
        self.ros_node.destroy_node()


# ==================== WebSocket 信令处理 ====================
async def signaling_handler(websocket, *args):
    global pipeline_manager
    print("[WS] Browser connected")
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


# ==================== Main ====================
async def async_main():
    global pipeline_manager
    asyncio_loop = asyncio.get_running_loop()

    glib_loop = GLib.MainLoop()
    glib_thread = threading.Thread(target=glib_loop.run, daemon=True, name="glib-main-loop")
    glib_thread.start()
    print("[Main] GLib MainLoop started")

    pipeline_manager = PipelineManager()
    thermal_camera = ThermalCamera(pipeline_manager)
    print("[Main] Thermal camera started")

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
        thermal_camera.stop()
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
