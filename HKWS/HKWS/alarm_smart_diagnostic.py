#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2节点：海康SDK云台控制 + 实时预览 + WebRTC 视频流发布
支持报警布防及多种报警类型解析（事件名称可读，不丢失报警）
修改：启用智能报警上报（区域入侵、越界侦测等 VCA 事件）
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import time
import threading
import queue
import ctypes
from ctypes import *
import sys
import os
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

# ========== SDK路径配置 ==========
SDK_PATH = "/home/jetson/ros2_ws/src/HKWS/HKWS"
HCNetSDKCom_path = SDK_PATH + "/HCNetSDKCom"

DEVICE_IP = "192.168.2.64"
DEVICE_PORT = 8000
USERNAME = "admin"
PASSWORD = "suzhouleog1380"
CHANNEL = 1
ALARM_DEPLOY_TYPE = 0  # 0-客户端布防；若完全无回调，可改成 1 再对比一次。

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

# 报警命令常量
COMM_ALARM_V30 = 0x4000          # 9000设备报警信息主动上传
COMM_ALARM_DEVICE = 0x4004       # 设备报警信息（旧版）
COMM_ALARM_DEVICE_V40 = 0x4009   # 设备报警信息扩展（V40）
COMM_ALARM_RULE = 0x1102              # 异常行为检测报警信息：越界/区域入侵等
COMM_ALARM_FACE = 0x1106
COMM_ALARM_ACS = 0x5002
COMM_ALARM_NOTIFICATION_REPORT = 0x1117 # 通知事件上报
COMM_VCA_ALARM = 0x4993               # 智能检测报警，常见为 JSON/ISAPI 透传
COMM_ISAPI_ALARM = 0x6009             # ISAPI 报警透传

# ========== 报警次类型名称映射（基于海康SDK头文件）==========
ALARM_MINOR_NAME = {
    0x01: "报警输入",
    0x02: "报警输出",
    0x03: "移动侦测开始",
    0x04: "移动侦测结束",
    0x05: "遮挡报警开始",
    0x06: "遮挡报警结束",
    0x07: "智能报警开始",
    0x08: "智能报警结束",
    0x09: "交通事件报警开始",
    0x0A: "交通事件报警结束",
    0x0B: "网络报警开始",
    0x0C: "网络报警结束",
    0x0D: "网络报警恢复",
    0x0E: "无线报警开始",
    0x0F: "无线报警结束",
    0x10: "PIR报警开始",
    0x11: "PIR报警结束",
    0x12: "呼救报警开始",
    0x13: "呼救报警结束",
    0x14: "数字通道报警输入开始",
    0x15: "数字通道报警输入结束",
    0x16: "人脸侦测报警开始",
    0x17: "人脸侦测报警结束",
    0x18: "VQD报警开始",
    0x19: "VQD报警结束",
    0x1A: "场景侦测报警",
    0x1B: "离开区域侦测开始",
    0x1C: "离开区域侦测结束",
    0x1D: "徘徊侦测开始",
    0x1E: "徘徊侦测结束",
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

class NET_DVR_ALARM_ISAPI_INFO(Structure):
    _fields_ = [
        ("pAlarmData", c_void_p),
        ("dwAlarmDataLen", c_uint32),
        ("byDataType", c_byte),        # 0-invalid, 1-xml, 2-json
        ("byPicturesNumber", c_byte),
        ("byRes", c_byte * 2),
        ("pPicPackData", c_void_p),
        ("byRes1", c_byte * 32),
    ]

# ========== 回调函数类型 ==========
REALDATACALLBACK = CFUNCTYPE(None, LONG, c_uint32, POINTER(c_byte), c_uint32, c_void_p)
EXCEPTIONCALLBACK = CFUNCTYPE(None, c_uint32, LONG, LONG, c_void_p)
DECCALLBACK = CFUNCTYPE(None, LONG, POINTER(c_byte), LONG, POINTER(FRAME_INFO), LONG, LONG)
MSGCallBack = CFUNCTYPE(None, LONG, POINTER(NET_DVR_ALARMER), c_void_p, c_uint32, c_void_p)

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

# ========== 全局变量 ==========
lPort = LONG(-1)
frame_queue = queue.Queue(maxsize=5)
play_ready_event = threading.Event()
first_frame_event = threading.Event()
node_instance = None
shutdown_flag = False

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

# ===== 报警回调函数（保留所有报警，只做名称映射）=====
def alarm_callback(lCommand, pAlarmer, pAlarmInfo, dwBufLen, pUser):
    raw = alarm_bytes(pAlarmInfo, min(dwBufLen, 64)) if pAlarmInfo and dwBufLen > 0 else b""
    dev_ip = "-"
    try:
        if pAlarmer:
            dev_ip = bytes(pAlarmer.contents.byDeviceIP).split(b"\x00", 1)[0].decode("ascii", errors="ignore") or "-"
    except Exception:
        pass
    print(f"[RAW报警回调] cmd=0x{lCommand:X}, len={dwBufLen}, devIP={dev_ip}, first64={raw.hex()}")

    if lCommand == COMM_ALARM_RULE:
        try:
            parse_vca_rule_alarm(pAlarmInfo, dwBufLen)
        except Exception as e:
            print(f"解析Smart规则报警异常: {e}")
        return

    if lCommand in (COMM_VCA_ALARM, COMM_ISAPI_ALARM, COMM_ALARM_NOTIFICATION_REPORT):
        try:
            parse_json_like_alarm(lCommand, pAlarmInfo, dwBufLen)
        except Exception as e:
            print(f"解析Smart透传报警异常: {e}")
        return

    # 处理 9000 设备报警 (COMM_ALARM_V30)
    if lCommand == COMM_ALARM_V30:
        try:
            alarm = cast(pAlarmInfo, POINTER(NET_DVR_ALARMINFO_V30)).contents
            atype = alarm.dwAlarmType
            ainput = alarm.dwAlarmInputNumber

            type_name = ALARM_MINOR_NAME.get(atype, f"未知事件(0x{atype:X})")
            print(f"[报警-9000] {type_name}, 输入端口={ainput}")
        except Exception as e:
            print(f"解析9000报警异常: {e}")
        return

    # 处理设备报警 (0x4004 / 0x4009)
    if lCommand in (COMM_ALARM_DEVICE, COMM_ALARM_DEVICE_V40):
        try:
            alarm_info = cast(pAlarmInfo, POINTER(NET_DVR_ALARMINFO_DEV_V40)).contents
            major_type = alarm_info.dwAlarmType
            point_count = alarm_info.dwNumber

            type_name_map = {
                0: "编码器通道报警", 1: "私有卷二损坏", 2: "NVR服务退出",
                3: "编码器状态异常", 4: "系统时钟异常", 5: "录像卷剩余容量过低",
                6: "移动侦测报警", 7: "遮挡报警", 8: "录像丢失报警"
            }
            type_name = type_name_map.get(major_type, f"未知主类型(0x{major_type:X})")
            if point_count > 0 or major_type != 0:
                print(f"[报警] {type_name}, 报警点数={point_count}")

            if point_count > 0 and alarm_info.pNO:
                ptr = alarm_info.pNO
                for i in range(point_count):
                    val = ptr[i]
                    if major_type in (0, 6, 7):
                        print(f"  报警点{i+1}: 通道号={val}")
                    elif major_type == 5:
                        print(f"  报警点{i+1}: 硬盘号={val}")
                    else:
                        print(f"  报警点{i+1}: 数值={val}")
        except Exception as e:
            print(f"报警解析错误: {e}")
        return

    if lCommand != 0:
        print(f"[报警] 未处理命令: 0x{lCommand:X}, 长度={dwBufLen}")

SMART_EVENT_NAME = {
    1: "穿越警戒面/越界检测",
    2: "进入区域",
    3: "离开区域",
    4: "区域入侵/周界入侵",
    5: "徘徊检测",
    6: "物品遗留/拿取",
    7: "停车检测",
    8: "快速移动",
    9: "区域人员聚集",
    10: "剧烈运动",
    20: "倒地检测",
    25: "折线警戒面",
}

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

def parse_vca_rule_alarm(p_alarm_info, length):
    data = alarm_bytes(p_alarm_info, length)
    if len(data) < 52:
        print(f"[Smart报警] COMM_ALARM_RULE 数据过短，长度={length}")
        return

    rule_base = 12
    rule_id = data[rule_base]
    scene_id = data[rule_base + 1]
    event_ex = int.from_bytes(data[rule_base + 2:rule_base + 4], "little", signed=False)
    rule_name = decode_c_string(data[rule_base + 4:rule_base + 36])
    event_old = int.from_bytes(data[rule_base + 36:rule_base + 40], "little", signed=False)
    event_type = event_ex or event_old
    event_name = SMART_EVENT_NAME.get(event_type, f"未知Smart事件({event_type})")
    print(f"[Smart报警] {event_name}, eventType={event_type}, ruleID={rule_id}, sceneID={scene_id}, ruleName={rule_name or '-'}")

def print_json_like_payload(command, data, data_type=None):
    if not data:
        print(f"[Smart报警] 0x{command:X} 空数据")
        return
    text = data.decode("utf-8", errors="ignore")
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        body = text[start:end + 1]
        event_type = re.search(r'"eventType"\s*:\s*"?([^",}]+)', body)
        channel = re.search(r'"channelID"\s*:\s*"?([^",}]+)', body)
        type_text = f", dataType={data_type}" if data_type is not None else ""
        print(f"[Smart报警-JSON] cmd=0x{command:X}{type_text}, eventType={(event_type.group(1) if event_type else '-')}, channel={(channel.group(1) if channel else '-')}")
    else:
        print(f"[Smart报警] cmd=0x{command:X}, 长度={len(data)}，未发现内联JSON，前32字节={data[:32].hex()}")

def parse_json_like_alarm(command, p_alarm_info, length):
    if command == COMM_ISAPI_ALARM:
        try:
            info = cast(p_alarm_info, POINTER(NET_DVR_ALARM_ISAPI_INFO)).contents
            data = alarm_bytes(info.pAlarmData, info.dwAlarmDataLen)
            print_json_like_payload(command, data, info.byDataType)
            return
        except Exception as e:
            print(f"[Smart报警] COMM_ISAPI_ALARM 结构解析失败: {e}")
            return

    data = alarm_bytes(p_alarm_info, length)
    print_json_like_payload(command, data)
_alarm_cb = MSGCallBack(alarm_callback)

# ========== SDK基础函数 ==========
def init_sdk():
    if not libhcnetsdk.NET_DVR_Init():
        return False
    try:
        log_dir = b"/tmp/hcnet_alarm_sdk_log"
        os.makedirs(log_dir.decode(), exist_ok=True)
        libhcnetsdk.NET_DVR_SetLogToFile.argtypes = [c_uint32, c_char_p, c_bool]
        libhcnetsdk.NET_DVR_SetLogToFile.restype = c_bool
        # 开启详细日志（级别3），便于调试智能报警
        if libhcnetsdk.NET_DVR_SetLogToFile(3, log_dir, False):
            print(f"SDK日志已开启: {log_dir.decode()}")
        else:
            print(f"SDK日志开启失败，错误码={libhcnetsdk.NET_DVR_GetLastError()}")
    except Exception as e:
        print(f"SDK日志配置异常: {e}")
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

# ========== WebRTC 视频轨道 ==========
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

        self.latest_frame = None
        self.frame_condition = threading.Condition()
        self.alarm_handle = None

        if not init_sdk():
            self.get_logger().error("SDK初始化失败")
            rclpy.shutdown()
            return
        libhcnetsdk.NET_DVR_SetExceptionCallBack_V30.argtypes = [c_uint32, c_void_p, EXCEPTIONCALLBACK, c_void_p]
        libhcnetsdk.NET_DVR_SetExceptionCallBack_V30(0, None, _exception_cb, None)

        # 注册报警回调
        libhcnetsdk.NET_DVR_SetDVRMessageCallBack_V30.argtypes = [MSGCallBack, c_void_p]
        libhcnetsdk.NET_DVR_SetDVRMessageCallBack_V30.restype = c_bool
        if not libhcnetsdk.NET_DVR_SetDVRMessageCallBack_V30(_alarm_cb, None):
            self.get_logger().error(f"设置报警回调失败，错误码={libhcnetsdk.NET_DVR_GetLastError()}")
        else:
            self.get_logger().info("报警回调已注册")

        self.user_id = login_device(DEVICE_IP, DEVICE_PORT, USERNAME, PASSWORD)
        if self.user_id is None:
            self.get_logger().error("登录失败")
            rclpy.shutdown()
            return

        # 布防（已修改参数）
        self.alarm_handle = self.setup_alarm()
        if self.alarm_handle is None:
            self.get_logger().error("布防失败，继续运行但无报警功能")
        else:
            self.get_logger().info(f"布防成功，句柄={self.alarm_handle}")

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

        cv2.namedWindow("Hikvision Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hikvision Control", 1280, 720)

        self.webrtc_thread = threading.Thread(target=self.run_webrtc_server, daemon=True)
        self.webrtc_thread.start()
        self.get_logger().info("WebRTC 信令服务器已启动 (ws://0.0.0.0:8080)")

        self.get_logger().info("节点已启动，预览画面已显示")
        self.get_logger().info("控制说明：订阅话题 /controlCam，格式如 '0 10' 或 '0 0'")
        self.get_logger().info("  动作:0-右,1-左,2-上,3-下,4-放大,5-缩小,6-雨刷,7-补光")
        self.get_logger().info("  数值:0点动，非0为度数/倍数；补光:0开1关")
        self.get_logger().info("报警监控已启动，事件名称已优化为可读格式")

        self.absolute_mode_active = [False]
        self.control_thread = None
        self.last_frame = None

        global node_instance
        node_instance = self

    # ========== 修改点：布防参数使能智能报警 ==========
    def setup_alarm(self):
        try:
            alarm_param = NET_DVR_SETUPALARM_PARAM()
            alarm_param.dwSize = sizeof(NET_DVR_SETUPALARM_PARAM)
            # 关键参数：启用扩展报警信息类型（支持 VCA 智能报警）
            alarm_param.byLevel = 1                    # 布防等级，高
            alarm_param.byAlarmInfoType = 1            # 扩展报警信息类型（必须为1才能收到智能事件）
            alarm_param.byRetAlarmTypeV40 = 1          # 报警类型使用 V40 结构
            alarm_param.byRetDevInfoVersion = 1        # CVR 设备报警返回 COMM_ALARM_DEVICE_V40
            alarm_param.byDeployType = 1               # 客户端布防（主动连接）
            alarm_param.bySubScription = 1             # 订阅报警，启用智能事件上报
            alarm_param.byCustomCtrl = 1               # 自定义控制使能
            # 其他字段保持默认（0）
            self.get_logger().info(
                f"布防参数: size={alarm_param.dwSize}, "
                f"byAlarmInfoType={alarm_param.byAlarmInfoType}, "
                f"byDeployType={alarm_param.byDeployType}, "
                f"bySubScription={alarm_param.bySubScription}"
            )
            libhcnetsdk.NET_DVR_SetupAlarmChan_V41.argtypes = [LONG, POINTER(NET_DVR_SETUPALARM_PARAM)]
            libhcnetsdk.NET_DVR_SetupAlarmChan_V41.restype = LONG
            handle = libhcnetsdk.NET_DVR_SetupAlarmChan_V41(self.user_id, byref(alarm_param))
            if handle >= 0:
                return handle
            else:
                err = libhcnetsdk.NET_DVR_GetLastError()
                self.get_logger().error(f"布防失败，错误码={err}")
                return None
        except Exception as e:
            self.get_logger().error(f"布防异常: {e}")
            return None

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

    # ---------- WebRTC 服务器 ----------
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
            print("WebRTC 服务器已启动")
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
            if action == 6:
                if value == 0:
                    print("[操作] 雨刷刮一次...")
                    switch_once(self.user_id, CHANNEL, WIPER_PWRON, duration=1.0)
                else:
                    self.get_logger().warn(f"动作6只支持数值0，收到 {value}")
                return
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

    # ---------- 主循环 ----------
    def run(self):
        import rclpy.executors
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)

        while rclpy.ok():
            try:
                frame = frame_queue.get(timeout=0.05)
                self.last_frame = frame
            except queue.Empty:
                frame = self.last_frame

            if frame is not None:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    display = cv2.resize(frame, (1280, int(h * scale)))
                else:
                    display = frame
                cv2.imshow("Hikvision Control", display)

            executor.spin_once(timeout_sec=0.01)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("用户按Q退出")
                break
            elif key == ord('m'):
                if not self.absolute_mode_active[0]:
                    self.absolute_mode_active[0] = True
                    self.get_logger().info("\n[提示] 进入绝对控制模式，输入命令控制云台（输入 exit 退出）")
                    self.control_thread = threading.Thread(target=self.absolute_control_thread,
                                                           args=(self.user_id, CHANNEL, self.absolute_mode_active),
                                                           daemon=True)
                    self.control_thread.start()
                else:
                    self.get_logger().info("[提示] 强制退出绝对控制模式")
                    self.absolute_mode_active[0] = False
                    if self.control_thread and self.control_thread.is_alive():
                        self.control_thread.join(timeout=0.5)
                continue

            if self.absolute_mode_active[0]:
                continue

            # 键盘控制
            if key in (ord('w'), 82):
                do_ptz_move(self.user_id, CHANNEL, TILT_UP, KEY_MOVE_DURATION)
            elif key in (ord('s'), 84):
                do_ptz_move(self.user_id, CHANNEL, TILT_DOWN, KEY_MOVE_DURATION)
            elif key in (ord('a'), 81):
                do_ptz_move(self.user_id, CHANNEL, PAN_LEFT, KEY_MOVE_DURATION)
            elif key in (ord('d'), 83):
                do_ptz_move(self.user_id, CHANNEL, PAN_RIGHT, KEY_MOVE_DURATION)
            elif key == ord('z'):
                do_ptz_move(self.user_id, CHANNEL, ZOOM_IN, KEY_ZOOM_DURATION)
            elif key == ord('x'):
                do_ptz_move(self.user_id, CHANNEL, ZOOM_OUT, KEY_ZOOM_DURATION)
            elif key == ord('1'):
                ptz_preset(self.user_id, CHANNEL, 8, 1)
                self.get_logger().info("已设置预置点1")
            elif key == ord('2'):
                ptz_preset(self.user_id, CHANNEL, 8, 2)
                self.get_logger().info("已设置预置点2")
            elif key == ord('3'):
                ptz_preset(self.user_id, CHANNEL, 8, 3)
                self.get_logger().info("已设置预置点3")
            elif key == ord('r'):
                self.get_logger().info("[操作] 雨刷刮一次...")
                switch_once(self.user_id, CHANNEL, WIPER_PWRON, duration=1.0)
            elif key == ord('f'):
                self.get_logger().info("[操作] 停止雨刷...")
                ptz_control(self.user_id, CHANNEL, WIPER_PWRON, 1)
            elif key == ord('e'):
                self.get_logger().info("[操作] 开启补光灯")
                ptz_control(self.user_id, CHANNEL, LIGHT_PWRON, 0)
            elif key == ord('c'):
                self.get_logger().info("[操作] 关闭补光灯")
                ptz_control(self.user_id, CHANNEL, LIGHT_PWRON, 1)

        self.cleanup()

    def absolute_control_thread(self, user_id, channel, active_flag):
        print("\n[绝对控制模式] 已启动，支持命令格式：")
        print("  - 变焦：'变大10倍'、'缩小5倍'")
        print("  - 角度：'往左45度'、'往右90度'、'往上15度'、'往下10度'")
        print("  - 查看速度：'speed'")
        print("  - 退出：'exit' （将自动退出绝对控制模式，恢复键盘控制）")

        while active_flag[0]:
            try:
                cmd = input("\n请输入控制命令: ").strip()
                if not cmd:
                    continue
                if cmd.lower() == 'exit':
                    print("退出绝对控制模式，恢复键盘控制")
                    active_flag[0] = False
                    break

                cmd_low = cmd.lower()
                if "变大" in cmd_low or "放大" in cmd_low:
                    match = re.search(r'(\d+\.?\d*)倍', cmd_low)
                    if match:
                        zoom = float(match.group(1))
                        move_zoom(user_id, channel, zoom)
                    else:
                        print("格式错误，示例：变大10倍")
                elif "缩小" in cmd_low:
                    match = re.search(r'(\d+\.?\d*)倍', cmd_low)
                    if match:
                        zoom = -float(match.group(1))
                        move_zoom(user_id, channel, zoom)
                    else:
                        print("格式错误，示例：缩小5倍")
                elif any(cmd_low.startswith(d) for d in ["往左","往右","往上","往下","向左","向右","向上","向下"]):
                    dir_match = re.search(r'(往左|往右|往上|往下|向左|向右|向上|向下)', cmd_low)
                    if not dir_match:
                        print("无效方向")
                        continue
                    dir_raw = dir_match.group(1)
                    dir_map = {"往左":"left","向左":"left","往右":"right","向右":"right",
                               "往上":"up","向上":"up","往下":"down","向下":"down"}
                    direction = dir_map[dir_raw]
                    num_match = re.search(r'(\d+\.?\d*)度', cmd_low)
                    if num_match:
                        angle = float(num_match.group(1))
                        move_angle(user_id, channel, direction, angle)
                    else:
                        print("请指定度数，例如：往左45度")
                elif cmd_low == "speed":
                    print(f"当前速度配置：水平 {SPEEDS['pan']}°/s, 垂直 {SPEEDS['tilt']}°/s, 变焦 {SPEEDS['zoom']}倍/s")
                else:
                    print("未知命令，支持：变大10倍, 缩小5倍, 往左45度, speed, exit")
            except Exception as e:
                print(f"命令处理异常: {e}")

    def cleanup(self):
        global shutdown_flag
        shutdown_flag = True
        if hasattr(self, 'alarm_handle') and self.alarm_handle is not None:
            libhcnetsdk.NET_DVR_CloseAlarmChan_V30(self.alarm_handle)
            self.get_logger().info("已撤防")
            time.sleep(0.3)
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        if hasattr(self, 'preview_handle') and self.preview_handle is not None:
            stop_preview(self.preview_handle)
        stop_playctrl()
        if hasattr(self, 'user_id') and self.user_id is not None:
            libhcnetsdk.NET_DVR_Logout(self.user_id)
        libhcnetsdk.NET_DVR_Cleanup()
        cv2.destroyAllWindows()
        self.get_logger().info("资源已释放")

def main(args=None):
    rclpy.init(args=args)
    node = HikCameraNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("键盘中断，关闭节点")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
