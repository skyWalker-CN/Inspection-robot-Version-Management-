#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import usb.core
import usb.util
import numpy as np
import cv2
import struct
import time
import datetime
import threading
import asyncio
import json
from types import SimpleNamespace

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# WebRTC 相关库
import websockets
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate

# ==================== 设备参数 ====================
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

WIDTH  = 256
HEIGHT = 192
TEMP_SIZE = WIDTH * HEIGHT * 2   # 98304
FRAME_SIZE = TEMP_SIZE + WIDTH * HEIGHT * 2  # 196608 (TEMP + YUV)

# ==================== UVC XU 通信函数 ====================
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


# ==================== WebRTC 视频轨道 ====================
class ProcessedFrameTrack(VideoStreamTrack):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.frame_count = 0
        self.start_time = time.time()

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame_bgr = await asyncio.get_event_loop().run_in_executor(
            None, self.node.get_latest_frame
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed
            print(f"WebRTC 发送帧率: {fps:.2f} fps")

        return video_frame


# ==================== 热像仪 ROS2 节点 ====================
class ThermalCameraNode(Node):
    def __init__(self):
        super().__init__('thermal_camera_node')

        # 平滑参数
        self.smooth_min = None
        self.smooth_max = None
        self.SMOOTH_ALPHA_UP = 0.6
        self.SMOOTH_ALPHA_DOWN = 0.05
        self.MIN_SPAN = 4.0

        # 帧共享
        self.latest_frame = None
        self.frame_condition = threading.Condition()

        # 初始化 USB 设备
        self.init_device()

        # 创建 ROS2 发布者
        self.publisher = self.create_publisher(String, 'hotCam', 10)

        # 启动 USB 捕获线程
        self.capture_thread = threading.Thread(target=self.usb_capture_loop, daemon=True)
        self.capture_thread.start()

        # 启动 WebRTC 信令服务器
        #self.webrtc_thread = threading.Thread(target=self.run_webrtc_server, daemon=True)
        #self.webrtc_thread.start()

        self.get_logger().info("热像仪节点已启动，正在通过 FID 同步帧...")

    # ---------- USB 初始化 ----------
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
        print(f"[INFO] Frame size: {FRAME_SIZE} bytes (TEMP={TEMP_SIZE} + YUV={TEMP_SIZE})")
        print(f"[INFO] Resolution: {WIDTH}x{HEIGHT}")

        ep, intf_num = find_vs_endpoint(dev)
        if dev.is_kernel_driver_active(intf_num):
            dev.detach_kernel_driver(intf_num)
        try:
            dev.set_interface_altsetting(intf_num, 0)
        except Exception:
            pass
        self.ep = ep

    # ---------- USB 帧捕获循环 ----------
    def usb_capture_loop(self):
        frame_buffer = bytearray()
        current_fid = None
        frame_count = 0

        print("[INFO] FID-based frame sync started.")

        while rclpy.ok():
            try:
                raw_data = self.ep.read(16384, timeout=1000)
                chunk = raw_data.tobytes() if hasattr(raw_data, 'tobytes') else bytes(raw_data)
                if len(chunk) < 2:
                    continue

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
                    payload = chunk[idx + hdr_len:]   # 本包剩余数据

                    if current_fid is None:
                        current_fid = fid
                        frame_buffer.extend(payload)
                    elif fid != current_fid:
                        if len(frame_buffer) > 0:
                            self.process_frame(frame_buffer, frame_count)
                            frame_count += 1
                        frame_buffer = bytearray(payload)
                        current_fid = fid
                    else:
                        frame_buffer.extend(payload)

                    idx = len(chunk)   # 跳出内层循环
                    break

            except usb.core.USBTimeoutError:
                continue
            except KeyboardInterrupt:
                break

        cv2.destroyAllWindows()

    # ---------- 帧处理 ----------
    def process_frame(self, frame_data, frame_num):
        if len(frame_data) < TEMP_SIZE:
            print(f"[WARN] Frame {frame_num} too short ({len(frame_data)} bytes)")
            return

        temp_raw = np.frombuffer(frame_data[:TEMP_SIZE], dtype=np.uint16).reshape(HEIGHT, WIDTH)
        temp_c = (temp_raw.astype(np.float32) / 64.0) - 50.0

        temps = temp_c.flatten()
        valid = temps[(temps > -25) & (temps < 560)]
        if len(valid) == 0:
            print(f"[Frame {frame_num:04d}] No valid temperature data")
            return

        t_min = np.percentile(valid, 2)
        t_max_real = np.max(valid)
        t_avg = np.mean(valid)

        raw_min = t_min
        raw_max = t_max_real
        if raw_max - raw_min < self.MIN_SPAN:
            mid = (raw_max + raw_min) / 2
            raw_min = mid - self.MIN_SPAN / 2
            raw_max = mid + self.MIN_SPAN / 2

        # 非对称平滑
        if self.smooth_min is None or self.smooth_max is None:
            self.smooth_min = raw_min
            self.smooth_max = raw_max
        else:
            self.smooth_min = 0.2 * raw_min + 0.8 * self.smooth_min
            if raw_max > self.smooth_max:
                self.smooth_max = self.SMOOTH_ALPHA_UP * raw_max + (1 - self.SMOOTH_ALPHA_UP) * self.smooth_max
            else:
                self.smooth_max = self.SMOOTH_ALPHA_DOWN * raw_max + (1 - self.SMOOTH_ALPHA_DOWN) * self.smooth_max

        print(f"[Frame {frame_num:04d}] Min: {t_min:.1f}°C | Max: {t_max_real:.1f}°C | "
              f"Avg: {t_avg:.1f}°C | Display range: {self.smooth_min:.1f}~{self.smooth_max:.1f}°C")

        # 伪彩色映射
        norm = ((temp_c - self.smooth_min) / (self.smooth_max - self.smooth_min) * 255.0)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        scale = 2.0
        h_s, w_s = int(HEIGHT * scale), int(WIDTH * scale)
        color_big = cv2.resize(color, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        cv2.putText(color_big, f"Max: {t_max_real:.1f}C Range: {self.smooth_min:.1f}~{self.smooth_max:.1f}C",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Thermal Camera (WebRTC)", color_big)
        cv2.waitKey(1)

        # 保存帧供 WebRTC 使用
        with self.frame_condition:
            self.latest_frame = color_big
            self.frame_condition.notify_all()

        # 发布温度到 ROS2
        msg = String()
        msg.data = f"{t_min:.1f} {t_max_real:.1f}"
        self.publisher.publish(msg)

    def get_latest_frame(self):
        with self.frame_condition:
            while self.latest_frame is None:
                self.frame_condition.wait()
            return self.latest_frame.copy()

    # ---------- WebRTC 服务器 ----------
    def run_webrtc_server(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        self.webrtc_loop = loop

        async def start_server():
            self.websocket_server = await websockets.serve(
                self.browser_handler, "0.0.0.0", 8081
            )
            print("🌐 WebRTC 信令服务器运行在 ws://0.0.0.0:8081")

        loop.run_until_complete(start_server())
        loop.run_forever()

    async def browser_handler(self, websocket):
        client_addr = websocket.remote_address
        print(f"新浏览器连接: {client_addr}")

        pc = RTCPeerConnection()
        track = ProcessedFrameTrack(self)
        pc.addTrack(track)

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                await websocket.send(json.dumps({
                    "type": "candidate",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex
                    }
                }))

        @pc.on("iceconnectionstatechange")
        async def on_ice_state():
            print(f"ICE state: {pc.iceConnectionState}")
            if pc.iceConnectionState in ["failed", "closed"]:
                await pc.close()

        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type == "offer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type="offer")
                    )
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    await websocket.send(json.dumps({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    }))
                elif msg_type == "candidate":
                    candidate_obj = self.create_ice_candidate(data["candidate"])
                    if candidate_obj:
                        try:
                            await pc.addIceCandidate(candidate_obj)
                        except Exception as e:
                            print(f"添加候选失败: {e}")
                elif msg_type == "bye":
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"浏览器 {client_addr} 断开")
            await pc.close()

    def create_ice_candidate(self, cand):
        candidate_str = cand.get("candidate", "")
        sdpMid = cand.get("sdpMid")
        sdpMLineIndex = cand.get("sdpMLineIndex")
        if not candidate_str:
            return None
        try:
            candidate = RTCIceCandidate.from_sdp(candidate_str)
            candidate.sdpMid = sdpMid
            candidate.sdpMLineIndex = sdpMLineIndex
            return candidate
        except AttributeError:
            pass
        parts = candidate_str.split()
        if len(parts) < 8:
            return None
        foundation = parts[0].split(':', 1)[1] if ':' in parts[0] else parts[0]
        component = int(parts[1])
        protocol = parts[2]
        priority = int(parts[3])
        ip = parts[4]
        port = int(parts[5])
        typ = parts[7]
        tcp_type = None
        if protocol.upper() == "TCP" and len(parts) > 9 and parts[8] == "tcptype":
            tcp_type = parts[9]
        return SimpleNamespace(
            component=component,
            foundation=foundation,
            protocol=protocol,
            priority=priority,
            ip=ip,
            port=port,
            type=typ,
            tcpType=tcp_type,
            relatedAddress=None,
            relatedPort=None,
            candidate=candidate_str,
            sdpMid=sdpMid,
            sdpMLineIndex=sdpMLineIndex
        )

    def destroy_node(self):
        cv2.destroyAllWindows()
        if hasattr(self, 'webrtc_loop') and self.webrtc_loop.is_running():
            self.webrtc_loop.call_soon_threadsafe(self.webrtc_loop.stop)
        usb.util.dispose_resources(self.dev)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ThermalCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
