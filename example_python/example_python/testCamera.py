from onvif import ONVIFCamera
import time
import zeep
import cv2
import threading
import queue
from datetime import datetime


class ONVIFPTZController:
    def __init__(self, ip, port, username, password):
        """
        初始化ONVIF PTZ控制器
        """
        try:
            # 创建ONVIF相机对象
            self.camera = ONVIFCamera(ip, port, username, password)
            # 创建媒体和PTZ服务
            self.media = self.camera.create_media_service()
            self.ptz = self.camera.create_ptz_service()

            # 获取媒体配置文件，通常第一个配置文件用于PTZ控制
            self.media_profile = self.media.GetProfiles()[0]

            # 获取RTSP流地址
            self.rtsp_url = self.get_rtsp_url()

            # 视频流控制变量
            self.is_streaming = False
            self.video_thread = None
            self.cap = None
            # 添加命令队列用于线程间通信
            self.command_queue = queue.Queue()
            self.current_display_width = 800  # 默认显示宽度

            print("ONVIF连接成功！")
            print(f"配置文件: {self.media_profile.Name}")
            print(f"RTSP地址: {self.rtsp_url}")

        except Exception as e:
            print(f"连接失败: {e}")
            raise

    def get_rtsp_url(self):
        """
        获取RTSP流地址
        """
        try:
            # 获取流URI
            stream_uri = self.media.GetStreamUri({
                'StreamSetup': {
                    'Stream': 'RTP-Unicast',
                    'Transport': {'Protocol': 'RTSP'}
                },
                'ProfileToken': self.media_profile.token
            })
            return stream_uri.Uri
        except Exception as e:
            print(f"获取RTSP地址失败: {e}")
            # 如果获取失败，使用已知的RTSP地址
            return "rtsp://10.0.1.137:554/stream2"

    def _video_stream_worker(self, window_name, display_width):
        """
        视频流工作线程的主要逻辑
        """
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if not self.cap.isOpened():
                print("无法打开RTSP视频流")
                return False

            # 创建可调整大小的窗口
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # 初始窗口大小
            cv2.resizeWindow(window_name, display_width, int(display_width * 9/16))
            
            self.is_streaming = True
            self.current_display_width = display_width
            print("视频流显示已启动，按Q键退出视频窗口")
            
            while self.is_streaming:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("无法读取视频帧")
                    # 尝试重新连接
                    time.sleep(1)
                    continue
                
                # 处理待执行的PTZ命令（非阻塞检查）
                try:
                    if not self.command_queue.empty():
                        command = self.command_queue.get_nowait()
                        if command == "stop":
                            self.stop()
                        elif command.startswith("move"):
                            # 这里可以解析具体的移动命令
                            pass
                except queue.Empty:
                    pass
                
                # 自动调整图像尺寸，保持宽高比
                height, width = frame.shape[:2]
                scale_factor = display_width / width
                new_width = display_width
                new_height = int(height * scale_factor)
                
                # 缩放图像
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # 在调整后的画面上添加信息文本
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(resized_frame, f"时间: {current_time}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(resized_frame, f"分辨率: {width}x{height} -> {new_width}x{new_height}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(resized_frame, "PTZ控制演示 - 按Q退出视频，控制台输入命令", (10, new_height-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 显示调整后的帧
                cv2.imshow(window_name, resized_frame)
                
                # 非阻塞等待，允许其他操作
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("收到Q键信号，关闭视频窗口")
                    break
                    
        except Exception as e:
            print(f"视频流显示错误: {e}")
        finally:
            self.stop_video_stream()

    def start_video_stream(self, window_name='摄像头画面 - 按Q退出', display_width=800):
        """
        启动视频流显示 - 修复版
        """
        if self.is_streaming:
            print("视频流已在运行中")
            return

        # 在新线程中启动视频流
        self.video_thread = threading.Thread(
            target=self._video_stream_worker,
            args=(window_name, display_width)
        )
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # 等待视频流初始化
        time.sleep(2)
        return True

    def stop_video_stream(self):
        """停止视频流显示"""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("视频流已停止")

    def continuous_move(self, pan=0.0, tilt=0.0, zoom=0.0, timeout=0):
        """
        连续移动PTZ
        pan: 水平速度 (-1.0~1.0, 负左正右)
        tilt: 垂直速度 (-1.0~1.0, 负下正上)
        zoom: 变焦速度 (-1.0~1.0, 负缩小正放大)
        timeout: 移动持续时间(秒)，0表示持续移动直到调用stop
        """
        try:
            # 创建连续移动请求
            move_request = self.ptz.create_type('ContinuousMove')
            move_request.ProfileToken = self.media_profile.token

            # 设置速度参数
            move_request.Velocity = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }

            # 执行移动
            self.ptz.ContinuousMove(move_request)

            # 如果设置了超时，在指定时间后停止
            if timeout > 0:
                time.sleep(timeout)
                self.stop()

        except Exception as e:
            print(f"连续移动失败: {e}")

    def absolute_move(self, pan=0.0, tilt=0.0, zoom=0.0):
        """
        绝对移动：将云台移动到指定的绝对位置
        参数范围通常在0~1之间
        """
        try:
            move_request = self.ptz.create_type('AbsoluteMove')
            move_request.ProfileToken = self.media_profile.token
            move_request.Position = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }

            self.ptz.AbsoluteMove(move_request)

        except Exception as e:
            print(f"绝对移动失败: {e}")

    def stop(self):
        """停止所有PTZ运动"""
        try:
            stop_request = self.ptz.create_type('Stop')
            stop_request.ProfileToken = self.media_profile.token
            stop_request.PanTilt = True
            stop_request.Zoom = True

            self.ptz.Stop(stop_request)
            print("PTZ运动已停止")

        except Exception as e:
            print(f"停止命令失败: {e}")

    def goto_home_position(self):
        """
        恢复到初始位置（Home位置）
        注意：此功能需要摄像头支持Home位置功能[8](@ref)
        """
        try:
            # 使用GotoHomePosition方法返回初始位置
            home_request = self.ptz.create_type('GotoHomePosition')
            home_request.ProfileToken = self.media_profile.token
            
            self.ptz.GotoHomePosition(home_request)
            print("已恢复到初始位置")
            
        except Exception as e:
            print(f"恢复初始位置失败: {e}")
            # 如果Home位置功能不支持，尝试使用绝对移动到中心位置
            print("尝试使用绝对移动恢复到中心位置...")
            try:
                self.absolute_move(pan=0.0, tilt=0.0, zoom=0.0)
                print("已通过绝对移动恢复到中心位置")
            except Exception as e2:
                print(f"绝对移动恢复位置也失败: {e2}")


def interactive_control_demo(ptz_controller):
    """
    交互式控制演示：同时显示视频流并进行PTZ控制 - 修改版
    """
    print("\n" + "="*50)
    print("交互式PTZ控制演示")
    print("="*50)
    
    # 启动视频流（会在新线程中运行）
    print("启动视频流...")
    success = ptz_controller.start_video_stream('实时监控 - PTZ控制演示', display_width=1024)
    
    if not success:
        print("视频流启动失败，但仍可进行PTZ控制")
    
    # 等待视频流初始化
    time.sleep(3)
    
    try:
        while ptz_controller.is_streaming:
            print("\n" + "-"*30)
            print("PTZ控制菜单")
            print("-"*30)
            print("1 - 向右平移3秒")
            print("2 - 向上移动3秒") 
            print("3 - 向左平移3秒")
            print("4 - 向下平移3秒")
            print("5 - 恢复原位")
            print("0 - 停止所有运动")
            print("q - 退出程序")
            print("-"*30)
            
            choice = input("请选择操作 (1-5, 0, q): ").strip().lower()
            
            if choice == 'q':
                print("退出程序...")
                break
            elif choice == '1':
                print("执行：向右平移...")
                ptz_controller.continuous_move(pan=0.5, timeout=3)
            elif choice == '2':
                print("执行：向上移动...")
                ptz_controller.continuous_move(tilt=0.5, timeout=3)
            elif choice == '3':
                print("执行：向左平移...")
                ptz_controller.continuous_move(pan=-0.5, timeout=3)
            elif choice == '4':
                print("执行：向下平移...")
                ptz_controller.continuous_move(tilt=-0.5, timeout=3)
            elif choice == '5':
                print("执行：恢复原位...")
                ptz_controller.goto_home_position()
            elif choice == '0':
                print("停止所有运动...")
                ptz_controller.stop()
            else:
                print("无效选择，请重新输入")
                
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"控制演示错误: {e}")
    finally:
        ptz_controller.stop_video_stream()
        print("交互式控制演示结束")


# 使用示例
if __name__ == "__main__":
    # 摄像头配置信息（请替换为您的实际信息）
    CAMERA_IP = "10.0.1.137"  # 替换为您的摄像头IP
    CAMERA_PORT = 80  # 通常为80
    USERNAME = "admin"  # 您的摄像头用户名
    PASSWORD = "88888888"  # 您的摄像头密码

    try:
        # 创建PTZ控制器实例
        ptz_controller = ONVIFPTZController(CAMERA_IP, CAMERA_PORT, USERNAME, PASSWORD)

        print("选择运行模式:")
        print("1 - 自动PTZ控制演示（无视频显示）")
        print("2 - 交互式PTZ控制（带视频显示）")
        print("3 - 仅显示视频流")
        
        mode = input("请选择模式 (1-3): ").strip()
        
        if mode == '1':
            # 自动演示模式
            print("开始PTZ控制演示...")
            print("向右平移...")
            ptz_controller.continuous_move(pan=0.5, timeout=3)
            time.sleep(1)
            print("向上移动...")
            ptz_controller.continuous_move(tilt=0.5, timeout=3)
            time.sleep(1)
            print("向左平移...")
            ptz_controller.continuous_move(pan=-0.5, timeout=3)
            time.sleep(1)
            print("向下平移...")
            ptz_controller.continuous_move(tilt=-0.5, timeout=3)
            time.sleep(1)
            print("恢复原位...")
            ptz_controller.goto_home_position()

        elif mode == '2':
            # 交互式控制模式（带视频显示）
            interactive_control_demo(ptz_controller)
            
        elif mode == '3':
            # 仅显示视频流模式
            print("启动视频流显示...")
            ptz_controller.start_video_stream(display_width=1024)
            # 等待视频流线程结束
            if ptz_controller.video_thread:
                ptz_controller.video_thread.join()
        else:
            print("无效选择，使用默认的自动演示模式")
            # 默认执行自动演示
            ptz_controller.continuous_move(pan=0.5, timeout=3)
            time.sleep(1)
            ptz_controller.continuous_move(tilt=0.5, timeout=3)

        print("程序执行完成！")

    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        # 确保资源被正确释放
        if 'ptz_controller' in locals():
            ptz_controller.stop_video_stream()
