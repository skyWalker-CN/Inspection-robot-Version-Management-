#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2节点：海康SDK云台控制 + 实时预览 + WebRTC + 报警功能（无显示窗口，持续发布图像）
订阅话题：/controlCam (std_msgs/String)
发布话题：/hik_camera/image_raw (sensor_msgs/Image)
WebRTC 信令服务器：ws://0.0.0.0:8080
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import threading
import queue
import ctypes
from ctypes import *
import sys
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import websockets
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
import fractions
import logging
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)

# ========== SDK路径与设备参数 ==========
SDK_PATH = "/home/jetson/ros2_ws/src/HKWS/HKWS"
HCNetSDKCom_path = SDK_PATH + "/HCNetSDKCom"

DEVICE_IP = "192.168.1.64"
DEVICE_PORT = 8000
USERNAME = "admin"
PASSWORD = "suzhouleog1380"
CHANNEL = 1

SPEEDS = {"pan": 50.0, "tilt": 30.0, "zoom": 6.7}
KEY_MOVE_DURATION = 0.2
KEY_ZOOM_DURATION = 0.3
PTZ_SPEED_LEVEL = 5
LONG = c_int32

# ========== SDK加载 ==========
ctypes.CDLL("libc.so.6", mode=ctypes.RTLD_GLOBAL)
try:
    libhcnetsdk = ctypes.CDLL(SDK_PATH + "/libhcnetsdk.so")
    libHCCore = ctypes.CDLL(SDK_PATH + "/libHCCore.so")
    libplayctrl = ctypes.CDLL(SDK_PATH + "/libPlayCtrl.so")
    try:
        ctypes.CDLL(HCNetSDKCom_path + "/libHCPreview.so")
    except:
        pass
except OSError as e:
    print(f"加载SDK失败: {e}")
    sys.exit(1)

# ========== 常量定义 ==========
NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
STREAME_REALTIME = 0
PAN_LEFT, PAN_RIGHT = 23, 24
TILT_UP, TILT_DOWN = 21, 22
ZOOM_IN, ZOOM_OUT = 11, 12
WIPER_PWRON, LIGHT_PWRON = 3, 2

# 预置位命令
SET_PRESET = 8
GOTO_PRESET = 39
CLE_PRESET = 9

COMM_ALARM_V30 = 0x4000
COMM_ALARM_DEVICE = 0x4004
COMM_ALARM_DEVICE_V40 = 0x4009
COMM_ALARM_RULE = 0x1102
COMM_ALARM_FACE = 0x1106
COMM_ALARM_ACS = 0x5002
COMM_ALARM_NOTIFICATION_REPORT = 0x1117
COMM_VCA_ALARM = 0x4993
COMM_ISAPI_ALARM = 0x6009

ALARM_MINOR_NAME = {
    0x01: "报警输入", 0x02: "报警输出", 0x03: "移动侦测开始", 0x04: "移动侦测结束",
    0x05: "遮挡报警开始", 0x06: "遮挡报警结束", 0x07: "智能报警开始", 0x08: "智能报警结束",
    0x09: "交通事件报警开始", 0x0A: "交通事件报警结束", 0x0B: "网络报警开始", 0x0C: "网络报警结束",
    0x0D: "网络报警恢复", 0x0E: "无线报警开始", 0x0F: "无线报警结束", 0x10: "PIR报警开始",
    0x11: "PIR报警结束", 0x12: "呼救报警开始", 0x13: "呼救报警结束", 0x14: "数字通道报警输入开始",
    0x15: "数字通道报警输入结束", 0x16: "人脸侦测报警开始", 0x17: "人脸侦测报警结束",
    0x18: "VQD报警开始", 0x19: "VQD报警结束", 0x1A: "场景侦测报警", 0x1B: "离开区域侦测开始",
    0x1C: "离开区域侦测结束", 0x1D: "徘徊侦测开始", 0x1E: "徘徊侦测结束",
}

SMART_EVENT_NAME = {
    1: "穿越警戒面/越界检测", 2: "进入区域", 3: "离开区域", 4: "区域入侵/周界入侵",
    5: "徘徊检测", 6: "物品遗留/拿取", 7: "停车检测", 8: "快速移动", 9: "区域人员聚集",
    10: "剧烈运动", 20: "倒地检测", 25: "折线警戒面",
}

# ========== 结构体定义 ==========
class NET_DVR_TIME(Structure):
    _fields_ = [("dwYear", c_uint32), ("dwMonth", c_uint32), ("dwDay", c_uint32),
                ("dwHour", c_uint32), ("dwMinute", c_uint32), ("dwSecond", c_uint32)]

class NET_DVR_ALARMINFO_V30(Structure):
    _fields_ = [
        ("dwAlarmType", c_uint32),
        ("dwAlarmInputNumber", c_uint32),
        ("byAlarmOutputNumber", c_byte * 96),
        ("byAlarmRelateChannel", c_byte * 64),
        ("byChannel", c_byte * 64),
        ("byDiskNumber", c_byte * 33)
    ]

class NET_ALARM_CVR_SUBINFO_UNION(Union):
    _fields_ = [("byLen", c_byte * 492)]

class NET_DVR_ALARMINFO_DEV_V40(Structure):
    _fields_ = [
        ("dwAlarmType", c_uint32),
        ("struTime", NET_DVR_TIME),
        ("uSubAlarmInfo", NET_ALARM_CVR_SUBINFO_UNION),
        ("byRes", c_byte * 256),
        ("dwNumber", c_uint32),
        ("pNO", POINTER(c_ushort))
    ]

class NET_DVR_ALARMER(Structure):
    _fields_ = [
        ("byUserIDValid", c_byte), ("bySerialValid", c_byte), ("byVersionValid", c_byte),
        ("byDeviceNameValid", c_byte), ("byMacAddrValid", c_byte), ("byLinkPortValid", c_byte),
        ("byDeviceIPValid", c_byte), ("byIPv6Valid", c_byte), ("byRes", c_byte * 3),
        ("dwUserID", c_uint32), ("bySerial", c_byte * 48), ("dwDeviceVersion", c_uint32),
        ("byDeviceName", c_char * 32), ("byMacAddr", c_byte * 6), ("wLinkPort", c_ushort),
        ("byDeviceIP", c_char * 128), ("byIPv6", c_char * 128), ("byRes1", c_byte * 116)
    ]

class NET_DVR_DEVICEINFO_V30(Structure):
    _fields_ = [("sSerialNumber", c_byte * 48), ("byAlarmInPortNum", c_byte),
                ("byAlarmOutPortNum", c_byte), ("byDiskNum", c_byte), ("byDVRType", c_byte),
                ("byChanNum", c_byte), ("byIPChanNum", c_byte), ("byStartChan", c_byte),
                ("byAudioChanNum", c_byte), ("byIPChanNum2", c_byte), ("byHighStartChan", c_byte),
                ("byRes2", c_byte * 2)]

class NET_DVR_DEVICEINFO_V40(Structure):
    _fields_ = [("struDeviceV30", NET_DVR_DEVICEINFO_V30), ("bySupportLock", c_byte),
                ("byRetryLoginTime", c_byte), ("byPasswordLevel", c_byte), ("byRes2", c_byte * 117)]

class NET_DVR_USER_LOGIN_INFO(Structure):
    _fields_ = [("sDeviceAddress", c_char * 129), ("wPort", c_ushort), ("sUserName", c_char * 64),
                ("sPassword", c_char * 64), ("cbLoginResult", c_void_p), ("pUser", c_void_p),
                ("bUseAsynLogin", c_bool), ("byProxyType", c_byte), ("byUseUTCTime", c_byte),
                ("byLoginMode", c_byte), ("byRes2", c_byte * 2), ("byRes3", c_byte * 120)]

class NET_DVR_PREVIEWINFO(Structure):
    _fields_ = [("lChannel", LONG), ("dwStreamType", c_uint32), ("dwLinkMode", c_uint32),
                ("hPlayWnd", c_void_p), ("bBlocked", c_bool), ("bPassbackRecord", c_bool),
                ("byPreviewMode", c_byte), ("byStreamID", c_byte * 32), ("byProtoType", c_byte),
                ("byRes1", c_byte), ("byVideoLevel", c_byte), ("dwDisplayNum", c_uint32)]

class FRAME_INFO(Structure):
    _fields_ = [("nWidth", LONG), ("nHeight", LONG), ("nStamp", LONG),
                ("nType", LONG), ("nFrameRate", LONG), ("nFrameNum", LONG)]

class NET_DVR_SETUPALARM_PARAM(Structure):
    _fields_ = [
        ("dwSize", c_uint32),
        ("byLevel", c_byte),
        ("byAlarmInfoType", c_byte),
        ("byRetAlarmTypeV40", c_byte),
        ("byRetDevInfoVersion", c_byte),
        ("byRetVQDAlarmType", c_byte),
        ("byFaceAlarmDetection", c_byte),
        ("bySupport", c_byte),
        ("byBrokenNetHttp", c_byte),
        ("wTaskNo", c_ushort),
        ("byDeployType", c_byte),
        ("bySubScription", c_byte),
        ("byRes1", c_byte * 2),
        ("byAlarmTypeURL", c_byte),
        ("byCustomCtrl", c_byte),
    ]

# ========== 回调函数类型 ==========
REALDATACALLBACK = CFUNCTYPE(None, LONG, c_uint32, POINTER(c_byte), c_uint32, c_void_p)
EXCEPTIONCALLBACK = CFUNCTYPE(None, c_uint32, LONG, LONG, c_void_p)
DECCALLBACK = CFUNCTYPE(None, LONG, POINTER(c_byte), LONG, POINTER(FRAME_INFO), LONG, LONG)
MSGCallBack = CFUNCTYPE(None, LONG, POINTER(NET_DVR_ALARMER), c_char_p, c_uint32, c_void_p)

# ========== 辅助函数 ==========
def yv12_to_bgr(yuv_array, width, height):
    expected_size = width * height * 3 // 2
    if len(yuv_array) < expected_size:
        return None
    try:
        yuv_image = yuv_array[:expected_size].reshape((height * 3 // 2, width))
        return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_YV12)
    except Exception as e:
        print(f"YV12转换失败: {e}")
        return None

def alarm_bytes(p_alarm_info, length):
    if not p_alarm_info or length <= 0:
        return b""
    try:
        return ctypes.string_at(p_alarm_info, length)
    except Exception:
        return b""

def decode_c_string(raw):
    if not raw:
        return ""
    raw = raw.split(b"\x00", 1)[0]
    for enc in ("utf-8", "gbk", "latin1"):
        try:
            return raw.decode(enc, errors="ignore").strip()
        except Exception:
            pass
    return ""

# ========== 全局变量 ==========
lPort = LONG(-1)
frame_queue = queue.Queue(maxsize=5)
play_ready_event = threading.Event()
first_frame_event = threading.Event()
node_instance = None
alarm_queue = queue.Queue(maxsize=100)

# ========== 回调函数实现 ==========
def dec_callback(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    try:
        fi = pFrameInfo.contents
        w, h = fi.nWidth, fi.nHeight
        if nSize > 0 and w > 0 and h > 0:
            yuv = np.ctypeslib.as_array(pBuf, shape=(nSize,)).view(np.uint8).copy()
            bgr = yv12_to_bgr(yuv, w, h)
            if bgr is not None:
                first_frame_event.set()
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except: pass
                frame_queue.put(bgr)
                if node_instance is not None:
                    node_instance.update_latest_frame(bgr)
    except Exception as e:
        print(f"解码回调异常: {e}")

_dec_cb = DECCALLBACK(dec_callback)

def real_data_callback(lRealHandle, dwDataType, pBuffer, dwBufSize, pUser):
    global lPort
    if dwDataType == NET_DVR_SYSHEAD:
        port = LONG(-1)
        if not libplayctrl.PlayM4_GetPort(byref(port)):
            return
        lPort = port
        if dwBufSize > 0 and lPort.value >= 0:
            libplayctrl.PlayM4_SetStreamOpenMode(lPort.value, STREAME_REALTIME)
            libplayctrl.PlayM4_OpenStream(lPort.value, pBuffer, dwBufSize, 1024*1024)
            libplayctrl.PlayM4_SetDecCallBack(lPort.value, _dec_cb)
            libplayctrl.PlayM4_Play(lPort.value, None)
            play_ready_event.set()
    elif dwDataType in (NET_DVR_STREAMDATA, 3):
        if dwBufSize > 0 and lPort.value >= 0:
            libplayctrl.PlayM4_InputData(lPort.value, pBuffer, dwBufSize)

_real_data_cb = REALDATACALLBACK(real_data_callback)

def exception_callback(dwType, lUserID, lHandle, pUser):
    print(f"异常回调: 0x{dwType:04X}")

_exception_cb = EXCEPTIONCALLBACK(exception_callback)

def alarm_callback(lCommand, pAlarmer, pAlarmInfo, dwBufLen, pUser):
    try:
        alarmer_data = None
        if pAlarmer:
            try:
                alarmer = cast(pAlarmer, POINTER(NET_DVR_ALARMER)).contents
                alarmer_data = {
                    "device_name": decode_c_string(alarmer.byDeviceName),
                    "device_ip": decode_c_string(alarmer.byDeviceIP),
                    "serial": decode_c_string(alarmer.bySerial),
                }
            except:
                alarmer_data = {"error": "failed to read alarmer"}
        alarm_data = alarm_bytes(pAlarmInfo, dwBufLen)
        alarm_queue.put({
            "command": lCommand,
            "alarmer": alarmer_data,
            "data": alarm_data,
            "len": dwBufLen,
        }, block=False)
    except Exception as e:
        print(f"报警回调异常: {e}")

_alarm_cb = MSGCallBack(alarm_callback)

# ========== 报警解析线程 ==========
def alarm_processing_thread():
    while not shutdown_flag:
        try:
            alarm = alarm_queue.get(timeout=0.5)
            command = alarm["command"]
            data = alarm["data"]
            data_len = alarm["len"]
            if command == COMM_ALARM_RULE:
                parse_vca_rule_alarm(data, data_len)
            elif command == COMM_ALARM_V30:
                parse_alarm_v30(data, data_len)
            elif command == COMM_ALARM_DEVICE_V40:
                parse_alarm_device_v40(data, data_len)
            elif command != 0:
                print(f"[报警] 未处理命令: 0x{command:X}, 长度={data_len}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"报警处理线程异常: {e}")

def parse_vca_rule_alarm(data, length):
    if len(data) < 16:
        return
    for i in range(16, min(len(data)-3, 256)):
        val = int.from_bytes(data[i:i+4], 'little')
        if 1 <= val <= 25:
            event_name = SMART_EVENT_NAME.get(val, f"未知Smart事件({val})")
            print(f"[智能报警] {event_name}")
            return

def parse_alarm_v30(data, length):
    if len(data) < 4:
        return
    alarm_type = int.from_bytes(data[0:4], 'little')
    input_num = int.from_bytes(data[4:8], 'little') if len(data) >= 8 else 0
    type_name = ALARM_MINOR_NAME.get(alarm_type, f"未知事件(0x{alarm_type:X})")
    print(f"[报警-9000] {type_name}, 输入端口={input_num}")

def parse_alarm_device_v40(data, length):
    if len(data) < 4:
        return
    major_type = int.from_bytes(data[0:4], 'little')
    type_name_map = {
        0: "编码器通道报警", 1: "私有卷二损坏", 2: "NVR服务退出",
        3: "编码器状态异常", 4: "系统时钟异常", 5: "录像卷剩余容量过低",
        6: "移动侦测报警", 7: "遮挡报警", 8: "录像丢失报警"
    }
    type_name = type_name_map.get(major_type, f"未知主类型(0x{major_type:X})")
    print(f"[设备报警] {type_name}")

# ========== SDK基础函数 ==========
def init_sdk():
    if not libhcnetsdk.NET_DVR_Init(): return False
    libhcnetsdk.NET_DVR_SetConnectTime(2000, 1)
    libhcnetsdk.NET_DVR_SetReconnect(10000, True)
    return True

def login_device(ip, port, username, password):
    info = NET_DVR_USER_LOGIN_INFO()
    info.sDeviceAddress = ip.encode()
    info.wPort = port
    info.sUserName = username.encode()
    info.sPassword = password.encode()
    info.bUseAsynLogin = False
    dev = NET_DVR_DEVICEINFO_V40()
    libhcnetsdk.NET_DVR_Login_V40.argtypes = [POINTER(NET_DVR_USER_LOGIN_INFO), POINTER(NET_DVR_DEVICEINFO_V40)]
    libhcnetsdk.NET_DVR_Login_V40.restype = LONG
    uid = libhcnetsdk.NET_DVR_Login_V40(byref(info), byref(dev))
    return uid if uid >= 0 else None

def start_preview(uid, ch=1, stream=0):
    pinfo = NET_DVR_PREVIEWINFO()
    pinfo.lChannel = ch
    pinfo.dwStreamType = stream
    libhcnetsdk.NET_DVR_RealPlay_V40.argtypes = [LONG, POINTER(NET_DVR_PREVIEWINFO), REALDATACALLBACK, c_void_p]
    libhcnetsdk.NET_DVR_RealPlay_V40.restype = LONG
    return libhcnetsdk.NET_DVR_RealPlay_V40(uid, byref(pinfo), _real_data_cb, None)

def stop_preview(handle):
    if handle is not None and handle >= 0:
        libhcnetsdk.NET_DVR_StopRealPlay(handle)

def stop_playctrl():
    global lPort
    if lPort.value >= 0:
        libplayctrl.PlayM4_Stop(lPort.value)
        libplayctrl.PlayM4_CloseStream(lPort.value)
        libplayctrl.PlayM4_FreePort(lPort.value)
        lPort = LONG(-1)

def ptz_control(uid, ch, cmd, stop=0, speed=0):
    if speed == 0:
        libhcnetsdk.NET_DVR_PTZControl_Other.argtypes = [LONG, LONG, c_uint, c_uint]
        return libhcnetsdk.NET_DVR_PTZControl_Other(uid, ch, cmd, stop)
    else:
        libhcnetsdk.NET_DVR_PTZControlWithSpeed_Other.argtypes = [LONG, LONG, c_uint, c_uint, c_uint]
        return libhcnetsdk.NET_DVR_PTZControlWithSpeed_Other(uid, ch, cmd, stop, speed)

def switch_once(uid, ch, cmd, duration=0.3):
    ptz_control(uid, ch, cmd, 0)
    time.sleep(duration)
    ptz_control(uid, ch, cmd, 1)

def ptz_preset(uid, ch, preset_cmd, idx):
    libhcnetsdk.NET_DVR_PTZPreset_Other.argtypes = [LONG, LONG, c_uint, c_uint]
    return libhcnetsdk.NET_DVR_PTZPreset_Other(uid, ch, preset_cmd, idx)

def setup_alarm(uid):
    try:
        alarm_param = NET_DVR_SETUPALARM_PARAM()
        alarm_param.dwSize = sizeof(NET_DVR_SETUPALARM_PARAM)
        alarm_param.byLevel = 0
        alarm_param.byAlarmInfoType = 0
        alarm_param.byRetAlarmTypeV40 = 0
        alarm_param.byRetDevInfoVersion = 0
        alarm_param.byDeployType = 0
        alarm_param.bySubScription = 0
        alarm_param.byCustomCtrl = 0
        libhcnetsdk.NET_DVR_SetupAlarmChan_V41.argtypes = [LONG, POINTER(NET_DVR_SETUPALARM_PARAM)]
        libhcnetsdk.NET_DVR_SetupAlarmChan_V41.restype = LONG
        handle = libhcnetsdk.NET_DVR_SetupAlarmChan_V41(uid, byref(alarm_param))
        if handle >= 0:
            print(f"布防成功，句柄={handle}")
            return handle
        else:
            err = libhcnetsdk.NET_DVR_GetLastError()
            print(f"布防失败，错误码={err}")
            return None
    except Exception as e:
        print(f"布防异常: {e}")
        return None

# ========== 运动控制函数 ==========
def move_angle(uid, ch, direction, angle):
    if angle <= 0: return
    if direction in ("left","right"):
        speed = SPEEDS["pan"]
        cmd = PAN_LEFT if direction == "left" else PAN_RIGHT
    elif direction in ("up","down"):
        speed = SPEEDS["tilt"]
        cmd = TILT_UP if direction == "up" else TILT_DOWN
    else: return
    duration = angle / speed
    if duration < 0.05: duration = 0.05
    print(f"[移动] {direction} {angle}度，耗时 {duration:.2f}s")
    ptz_control(uid, ch, cmd, 0, PTZ_SPEED_LEVEL)
    time.sleep(duration)
    ptz_control(uid, ch, cmd, 1, PTZ_SPEED_LEVEL)

def move_zoom(uid, ch, zoom_step):
    if zoom_step == 0: return
    speed = SPEEDS["zoom"]
    cmd = ZOOM_IN if zoom_step>0 else ZOOM_OUT
    duration = abs(zoom_step) / speed
    if duration < 0.05: duration = 0.05
    print(f"[变焦] {'放大' if zoom_step>0 else '缩小'} {abs(zoom_step)}倍，耗时 {duration:.2f}s")
    ptz_control(uid, ch, cmd, 0, PTZ_SPEED_LEVEL)
    time.sleep(duration)
    ptz_control(uid, ch, cmd, 1, PTZ_SPEED_LEVEL)

def do_ptz_move(uid, ch, cmd, duration):
    ptz_control(uid, ch, cmd, 0, PTZ_SPEED_LEVEL)
    time.sleep(duration)
    ptz_control(uid, ch, cmd, 1, PTZ_SPEED_LEVEL)

# ========== WebRTC 视频轨道（仅用于推流，不发布ROS图像）==========
class ProcessedFrameTrack(VideoStreamTrack):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.frame_count = 0
        self.start_time = time.time()
        self._last_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        try:
            frame_bgr = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.node.get_latest_frame),
                timeout=0.1
            )
            self._last_frame = frame_bgr
        except asyncio.TimeoutError:
            if self._last_frame is not None:
                frame_bgr = self._last_frame
            else:
                frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        # WebRTC 需要 RGB 格式
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

# ========== ROS2节点 ==========
class HikCameraNode(Node):
    def __init__(self):
        super().__init__('hik_camera_node')
        self.subscription = self.create_subscription(
            String,
            'controlCam',
            self.control_callback,
            10)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.motion_lock = threading.Lock()

        # 图像发布器（全局唯一）
        self.image_pub = self.create_publisher(Image, '/hik_camera/image_raw', 10)
        self.bridge = CvBridge()

        self.latest_frame = None
        self.frame_condition = threading.Condition()

        if not init_sdk():
            self.get_logger().error("SDK初始化失败")
            rclpy.shutdown()
            return
        libhcnetsdk.NET_DVR_SetExceptionCallBack_V30.argtypes = [c_uint32, c_void_p, EXCEPTIONCALLBACK, c_void_p]
        libhcnetsdk.NET_DVR_SetExceptionCallBack_V30(0, None, _exception_cb, None)

        self.user_id = login_device(DEVICE_IP, DEVICE_PORT, USERNAME, PASSWORD)
        if self.user_id is None:
            self.get_logger().error("登录失败")
            rclpy.shutdown()
            return

        self.alarm_handle = setup_alarm(self.user_id)
        if self.alarm_handle is None:
            self.get_logger().warn("布防失败，报警功能不可用")
        else:
            libhcnetsdk.NET_DVR_SetDVRMessageCallBack_V30.argtypes = [MSGCallBack, c_void_p]
            libhcnetsdk.NET_DVR_SetDVRMessageCallBack_V30.restype = c_bool
            if not libhcnetsdk.NET_DVR_SetDVRMessageCallBack_V30(_alarm_cb, None):
                self.get_logger().error("设置报警回调失败")
            else:
                self.get_logger().info("报警回调已注册")

        self.preview_handle = start_preview(self.user_id, CHANNEL, 0)
        if self.preview_handle < 0:
            self.get_logger().error(f"启动预览失败，错误码 {libhcnetsdk.NET_DVR_GetLastError()}")
            self.cleanup()
            rclpy.shutdown()
            return

        if not play_ready_event.wait(10.0) or not first_frame_event.wait(15.0):
            self.get_logger().error("超时：未能获取画面")
            self.cleanup()
            rclpy.shutdown()
            return

        # 无本地显示窗口

        global shutdown_flag
        shutdown_flag = False
        self.alarm_thread = threading.Thread(target=alarm_processing_thread, daemon=True)
        self.alarm_thread.start()

        #self.webrtc_thread = threading.Thread(target=self.run_webrtc_server, daemon=True)
        #self.webrtc_thread.start()
        self.get_logger().info("WebRTC 信令服务器已启动 (ws://0.0.0.0:8080)")

        self.get_logger().info("节点已启动，持续发布图像到 /hik_camera/image_raw")
        self.get_logger().info("控制说明：订阅话题 /controlCam，格式如 '0 10' 或 '0 0'")
        self.get_logger().info("  动作:0-右,1-左,2-上,3-下,4-放大,5-缩小,6-雨刷,7-补光")
        self.get_logger().info("  动作:8-设置预置位(数值为编号), 9-调用预置位(数值为编号)")
        self.get_logger().info("  数值:0点动，非0为度数/倍数；补光:0开1关")

        global node_instance
        node_instance = self

    def update_latest_frame(self, frame):
        with self.frame_condition:
            self.latest_frame = frame
            self.frame_condition.notify_all()

    def get_latest_frame(self):
        with self.frame_condition:
            if self.latest_frame is None:
                self.frame_condition.wait(timeout=1.0)
            if self.latest_frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.latest_frame.copy()

    def run_webrtc_server(self):
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            self.webrtc_loop = asyncio.get_event_loop()
            async def start_server():
                self.websocket_server = await websockets.serve(
                    self.browser_handler, "0.0.0.0", 8080
                )
                print("✅ WebRTC 信令服务器运行在 ws://0.0.0.0:8080")
            self.webrtc_loop.run_until_complete(start_server())
            self.webrtc_loop.run_forever()
        except Exception as e:
            print(f"❌ WebRTC 服务器启动失败: {e}")

    async def browser_handler(self, websocket):
        client_addr = websocket.remote_address
        print(f"新浏览器连接: {client_addr}")
        pc = RTCPeerConnection()
        track = ProcessedFrameTrack(self)
        pc.addTransceiver(track, direction="sendonly")

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
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
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

    # ---------- ROS2 控制回调 ----------
    def control_callback(self, msg):
        data = msg.data.strip()
        if not data:
            return
        parts = data.split()
        if len(parts) != 2:
            self.get_logger().warn(f"无效消息格式: {data}")
            return
        try:
            action = int(parts[0])
            value = float(parts[1])
        except ValueError:
            self.get_logger().warn(f"解析失败: {data}")
            return
        self.thread_pool.submit(self._execute_motion, action, value)

    def _execute_motion(self, action, value):
        with self.motion_lock:
            # 雨刷
            if action == 6:
                if value == 0:
                    print("[操作] 雨刷刮一次...")
                    switch_once(self.user_id, CHANNEL, WIPER_PWRON, duration=1.0)
                else:
                    self.get_logger().warn(f"动作6只支持数值0，收到 {value}")
                return
            # 补光灯
            elif action == 7:
                if value == 0:
                    print("[操作] 开启补光灯")
                    ptz_control(self.user_id, CHANNEL, LIGHT_PWRON, 0)
                elif value == 1:
                    print("[操作] 关闭补光灯")
                    ptz_control(self.user_id, CHANNEL, LIGHT_PWRON, 1)
                else:
                    self.get_logger().warn(f"动作7支持数值0(开)或1(关)，收到 {value}")
                return

            # 设置预置位
            elif action == 8:
                idx = int(value)
                print(f"[操作] 设置预置位 {idx}")
                ptz_preset(self.user_id, CHANNEL, SET_PRESET, idx)
                return

            # 调用预置位
            elif action == 9:
                idx = int(value)
                print(f"[操作] 调用预置位 {idx}")
                ptz_preset(self.user_id, CHANNEL, GOTO_PRESET, idx)
                return

            # 原有云台方向控制
            if action == 0:
                direction = "right"
                cmd = PAN_RIGHT
            elif action == 1:
                direction = "left"
                cmd = PAN_LEFT
            elif action == 2:
                direction = "up"
                cmd = TILT_UP
            elif action == 3:
                direction = "down"
                cmd = TILT_DOWN
            elif action == 4:
                if value == 0:
                    do_ptz_move(self.user_id, CHANNEL, ZOOM_IN, KEY_ZOOM_DURATION)
                else:
                    move_zoom(self.user_id, CHANNEL, value)
                return
            elif action == 5:
                if value == 0:
                    do_ptz_move(self.user_id, CHANNEL, ZOOM_OUT, KEY_ZOOM_DURATION)
                else:
                    move_zoom(self.user_id, CHANNEL, -value)
                return
            else:
                self.get_logger().warn(f"未知动作: {action}")
                return

            if value == 0:
                do_ptz_move(self.user_id, CHANNEL, cmd, KEY_MOVE_DURATION)
            else:
                move_angle(self.user_id, CHANNEL, direction, value)

    # ---------- 主循环（发布图像，无显示）----------
    def run(self):
        import rclpy.executors
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)

        while rclpy.ok() and not shutdown_flag:
            try:
                frame = frame_queue.get(timeout=0.05)
                self.last_frame = frame
                # ----- 发布 ROS 图像（不依赖 WebRTC）-----
                try:
                    ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    ros_image.header.stamp = self.get_clock().now().to_msg()
                    ros_image.header.frame_id = "hik_camera"
                    self.image_pub.publish(ros_image)
                except Exception as e:
                    self.get_logger().error(f"图像发布失败: {e}")
            except queue.Empty:
                pass
            executor.spin_once(timeout_sec=0.01)

        self.cleanup()

    def cleanup(self):
        global shutdown_flag
        shutdown_flag = True

        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        if hasattr(self, 'alarm_handle') and self.alarm_handle is not None:
            try:
                libhcnetsdk.NET_DVR_CloseAlarmChan_V30(self.alarm_handle)
            except:
                pass
        if hasattr(self, 'preview_handle') and self.preview_handle is not None:
            stop_preview(self.preview_handle)
        stop_playctrl()
        if hasattr(self, 'user_id') and self.user_id is not None:
            libhcnetsdk.NET_DVR_Logout(self.user_id)
        libhcnetsdk.NET_DVR_Cleanup()
        self.get_logger().info("资源已释放")

shutdown_flag = False

def main(args=None):
    rclpy.init(args=args)
    node = HikCameraNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("键盘中断，关闭节点")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
