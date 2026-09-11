import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import shlex
import os
import signal
import threading
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.subscription = self.create_subscription(
            String,
            'control_robot',
            self.control_callback,
            10)
        self.saveMapResultPub = self.create_publisher(
            String,
            'mapSavedResult',
            10
            )
        self.carto_process = None  # Cartographer进程
        self.map_save_process = None  # 地图保存进程
        self.nav_process = None
        self.get_logger().info("Robot controller node started. Waiting for commands...")
        
        self.mapName = '' # 保存地图的名字

    def control_callback(self, msg):
        if msg.data == "1":
            self.start_cartographer()
        elif msg.data.startswith('2'):
            self.mapName = msg.data.split(',')[1]
            # 在新线程中执行地图保存和停止操作
            threading.Thread(target=self.save_map).start()
        elif msg.data == "3":
            self.stop_cartographer()
        elif msg.data.startswith('4'):
            self.mapName = msg.data.split(',')[1]
            self.start_navigation()
        elif msg.data == "5":
            self.stop_navigation()
        else:
            self.get_logger().warn(f"Received unknown command: {msg.data}")

    def stop_navigation(self):
        """停止Cartographer进程"""
        if self.nav_process is None:
            self.get_logger().warn("No navigation process to stop")
            return

        if self.nav_process.poll() is not None:
            self.get_logger().warn("Navigation process already terminated")
            self.nav_process = None
            return

        try:
            # 向整个进程组发送终止信号
            os.killpg(os.getpgid(self.nav_process.pid), signal.SIGTERM)
            self.get_logger().info(f"Sent SIGTERM to navigation process group (PID: {self.nav_process.pid})")
            # 等待进程终止
            self.nav_process.wait(timeout=5)
            self.get_logger().info("Navigation stopped")
        except ProcessLookupError:
            self.get_logger().warn("Navigation process already terminated")
        except Exception as e:
            self.get_logger().error(f"Error stopping navigation: {str(e)}")
        finally:
            self.nav_process = None

    def start_cartographer(self):
        if self.carto_process and self.carto_process.poll() is None:
            self.get_logger().info("Cartographer is already running")
            return
        
        try:
            cmd = "ros2 launch cartographer_ros backpack_2d.launch.py"
            self.get_logger().info(f"Launching Cartographer: {cmd}")
            
            self.carto_process = subprocess.Popen(
                shlex.split(cmd),
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.get_logger().info(f"Cartographer started with PID: {self.carto_process.pid}")
        except Exception as e:
            self.get_logger().error(f"Failed to start cartographer: {str(e)}")

    def start_navigation(self):
        if self.carto_process and self.carto_process.poll() is None:
            self.get_logger().info("Navigation is already running")
            return

        try:
            cmd = "ros2 launch nav2_bringup navigation2.launch.py map:=/home/jetson/ros2_ws/src/navigation2/nav2_bringup/maps/" + self.mapName
            self.get_logger().info(f"Launching Navigation: {cmd}")

            self.nav_process = subprocess.Popen(
                shlex.split(cmd),
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.get_logger().info(f"Navigation started with PID: {self.nav_process.pid}")
        except Exception as e:
            self.get_logger().error(f"Failed to start navigation: {str(e)}")

    def save_map(self):
        self.get_logger().info("Saving map...")
        self.save_map()

    def save_map(self):
        """执行地图保存命令"""
        try:
            cmd = "ros2 run nav2_map_server map_saver_cli -f " + self.mapName
            self.get_logger().info(f"Running map saver: {cmd}")
            
            self.map_save_process = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.get_logger().info(f"Map saver started with PID: {self.map_save_process.pid}")
            
            # 等待命令完成或超时
            self.get_logger().info("Waiting for map saver to complete...")
            try:
                # 设置较长时间限以完成地图保存
                _, _ = self.map_save_process.communicate(timeout=45)
                if self.map_save_process.returncode == 0:
                    self.get_logger().info("Map saved successfully!")
                    
                    # 等待地图保存完成
                    time.sleep(1)  # 增加短暂延迟确保地图完全保存
        
                    mapPath = '/home/jetson/ros2_ws/' + self.mapName + '.yaml'
                    if os.path.exists(mapPath):
                        result = String()
                        result.data = 'map saved Successfully'
                        self.saveMapResultPub.publish(result)
                        print("Map Saved Successfully!")
                else:
                    error = self.map_save_process.stderr.read().decode().strip()
                    self.get_logger().error(f"Map saver failed with error: {error}")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Map saver timed out, force terminating...")
                self.terminate_map_saver()
        except Exception as e:
            self.get_logger().error(f"Failed to run map saver: {str(e)}")
        finally:
            self.terminate_map_saver()

    def terminate_map_saver(self):
        """终止地图保存进程"""
        if self.map_save_process and self.map_save_process.poll() is None:
            try:
                # 尝试正常终止
                self.map_save_process.terminate()
                self.get_logger().info(f"Terminating map saver (PID: {self.map_save_process.pid})")
                
                # 等待进程终止
                self.map_save_process.wait(timeout=2)
                self.get_logger().info("Map saver terminated")
            except (subprocess.TimeoutExpired, Exception):
                self.get_logger().warn("Force killing map saver process")
                self.map_save_process.kill()
        
        self.map_save_process = None

    def stop_cartographer(self):
        """停止Cartographer进程"""
        if self.carto_process is None:
            self.get_logger().warn("No cartographer process to stop")
            return
            
        if self.carto_process.poll() is not None:
            self.get_logger().warn("Cartographer process already terminated")
            self.carto_process = None
            return
            
        try:
            # 向整个进程组发送终止信号
            os.killpg(os.getpgid(self.carto_process.pid), signal.SIGTERM)
            self.get_logger().info(f"Sent SIGTERM to cartographer process group (PID: {self.carto_process.pid})")
            # 等待进程终止
            self.carto_process.wait(timeout=5)
            self.get_logger().info("Cartographer stopped")
        except ProcessLookupError:
            self.get_logger().warn("Cartographer process already terminated")
        except Exception as e:
            self.get_logger().error(f"Error stopping cartographer: {str(e)}")
        finally:
            self.carto_process = None

    def __del__(self):
        # 清理所有进程
        if self.carto_process and self.carto_process.poll() is None:
            self.stop_cartographer()
        if self.map_save_process and self.map_save_process.poll() is None:
            self.terminate_map_saver()

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
