#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2节点：海康SDK云台控制 + 实时预览 + ROS图像话题发布
订阅话题：/controlCam (std_msgs/String) 可控制云台
发布话题：/hik_camera/image_raw (sensor_msgs/Image)
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

# ========== SDK路径配置 ==========
SDK_PATH = "/home/jetson/ros2_ws/src/HKWS/HKWS"
HCNetSDKCom_path = SDK_PATH + "/HCNetSDKCom"

DEVICE_IP = "192.168.2.64"
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

# ========== 结构体 ==========
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

# ========== 回调函数类型 ==========
REALDATACALLBACK = CFUNCTYPE(None, LONG, c_uint32, POINTER(c_byte), c_uint32, c_void_p)
EXCEPTIONCALLBACK = CFUNCTYPE(None, c_uint32, LONG, LONG, c_void_p)
DECCALLBACK = CFUNCTYPE(None, LONG, POINTER(c_byte), LONG, POINTER(FRAME_INFO), LONG, LONG)

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
                # 放入队列供主循环显示
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except: pass
                frame_queue.put(bgr)
                # 更新节点的最新帧，用于ROS发布
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

        # 最新帧（用于ROS发布）
        self.latest_frame = None
        self.frame_condition = threading.Condition()

        # 初始化SDK
        if not init_sdk():
            self.get_logger().error("SDK初始化失败")
            rclpy.shutdown()
            return
        libhcnetsdk.NET_DVR_SetExceptionCallBack_V30.argtypes = [c_uint32, c_void_p, EXCEPTIONCALLBACK, c_void_p]
        libhcnetsdk.NET_DVR_SetExceptionCallBack_V30(0, None, _exception_cb, None)

        # 登录设备
        self.user_id = login_device(DEVICE_IP, DEVICE_PORT, USERNAME, PASSWORD)
        if self.user_id is None:
            self.get_logger().error("登录失败")
            rclpy.shutdown()
            return

        # 启动预览
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

        # 创建本地显示窗口（可选）
        self.show_display = True  # 是否显示本地窗口
        if self.show_display:
            cv2.namedWindow("Hikvision Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Hikvision Camera", 1280, 720)

        # ROS图像发布器
        self.image_pub = self.create_publisher(Image, '/hik_camera/image_raw', 10)
        self.bridge = CvBridge()

        self.get_logger().info("节点已启动，预览画面正常，图像已发布到 /hik_camera/image_raw")
        self.get_logger().info("控制说明：订阅话题 /controlCam，格式如 '0 10' 或 '0 0'")
        self.get_logger().info("  动作:0-右,1-左,2-上,3-下,4-放大,5-缩小,6-雨刷,7-补光")
        self.get_logger().info("  数值:0点动，非0为度数/倍数；补光:0开1关")
        self.get_logger().info("键盘控制：W/A/S/D 移动，Z/X 变焦，1/2/3 预置点，R 雨刷，E/C 补光")

        self.absolute_mode_active = [False]
        self.control_thread = None
        self.last_frame = None

        global node_instance
        node_instance = self

    def update_latest_frame(self, frame):
        """更新最新帧并发布到ROS"""
        with self.frame_condition:
            self.latest_frame = frame
            self.frame_condition.notify_all()
        # 发布ROS图像
        if frame is not None:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "hik_camera"
                self.image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"发布图像失败: {e}")

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

            if frame is not None and self.show_display:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    display = cv2.resize(frame, (1280, int(h * scale)))
                else:
                    display = frame
                cv2.imshow("Hikvision Camera", display)

            executor.spin_once(timeout_sec=0.01)

            if self.show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("用户按Q退出")
                    break
                elif key == ord('m'):
                    if not self.absolute_mode_active[0]:
                        self.absolute_mode_active[0] = True
                        self.get_logger().info("\n[提示] 进入绝对控制模式，输入 exit 退出")
                        self.control_thread = threading.Thread(target=self.absolute_control_thread,
                                                               args=(self.user_id, CHANNEL, self.absolute_mode_active),
                                                               daemon=True)
                        self.control_thread.start()
                    else:
                        self.get_logger().info("[提示] 退出绝对控制模式")
                        self.absolute_mode_active[0] = False
                        if self.control_thread and self.control_thread.is_alive():
                            self.control_thread.join(timeout=0.5)
                    continue

                if self.absolute_mode_active[0]:
                    continue

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
        print("\n[绝对控制模式] 命令：变大10倍、缩小5倍、往左45度、speed、exit")
        while active_flag[0]:
            try:
                cmd = input("\n命令: ").strip()
                if not cmd:
                    continue
                if cmd.lower() == 'exit':
                    print("退出绝对控制模式")
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
                elif any(d in cmd_low for d in ["往左","往右","往上","往下","向左","向右","向上","向下"]):
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
                    print(f"速度：水平{SPEEDS['pan']}°/s, 垂直{SPEEDS['tilt']}°/s, 变焦{SPEEDS['zoom']}倍/s")
                else:
                    print("未知命令，支持：变大10倍, 缩小5倍, 往左45度, speed, exit")
            except Exception as e:
                print(f"命令处理异常: {e}")

    def cleanup(self):
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        if hasattr(self, 'preview_handle') and self.preview_handle is not None:
            stop_preview(self.preview_handle)
        stop_playctrl()
        if hasattr(self, 'user_id') and self.user_id is not None:
            libhcnetsdk.NET_DVR_Logout(self.user_id)
        libhcnetsdk.NET_DVR_Cleanup()
        if self.show_display:
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
