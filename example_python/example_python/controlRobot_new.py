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
        self.carto_process = None
        self.map_save_process = None
        self.nav_process = None
        self.map_save_lock = threading.Lock()
        self.map_name_lock = threading.Lock()
        self.mapName = ''
        self.get_logger().info("Robot controller node started. Waiting for commands...")

    # ---------- 辅助函数 ----------
    @staticmethod
    def strip_yaml_suffix(name):
        return name[:-5] if name.endswith('.yaml') else name

    @staticmethod
    def ensure_yaml_suffix(name):
        return name if name.endswith('.yaml') else name + '.yaml'

    def is_navigation_running(self):
        try:
            result = subprocess.run(
                ["ros2", "node", "list"],
                capture_output=True, text=True, timeout=2
            )
            nodes = result.stdout.splitlines()
            nav_nodes = {"bt_navigator", "controller_server", "planner_server", "recoveries_server"}
            return any(any(node_name in n for n in nodes) for node_name in nav_nodes)
        except Exception as e:
            self.get_logger().warn(f"Failed to check navigation nodes: {e}")
            return False

    def is_cartographer_running(self):
        try:
            result = subprocess.run(
                ["ros2", "node", "list"],
                capture_output=True, text=True, timeout=2
            )
            nodes = result.stdout.splitlines()
            carto_nodes = {"cartographer_node", "cartographer_occupancy_grid_node"}
            return any(any(cn in n for n in nodes) for cn in carto_nodes)
        except Exception as e:
            self.get_logger().warn(f"Failed to check cartographer nodes: {e}")
            return False

    # ---------- 进程启动和停止 ----------
    def start_cartographer(self):
        if self.carto_process and self.carto_process.poll() is None:
            self.get_logger().info("Cartographer is already running (managed)")
            return
        if self.is_cartographer_running():
            self.get_logger().warn("Cartographer is running externally, cannot start")
            return
        if self.nav_process and self.nav_process.poll() is None:
            self.get_logger().warn("Navigation is running, cannot start cartographer")
            return
        if self.is_navigation_running():
            self.get_logger().warn("Navigation is running externally, cannot start cartographer")
            return

        try:
            cmd = "ros2 launch cartographer_ros backpack_2d.launch.py"
            self.get_logger().info(f"Launching Cartographer: {cmd}")
            self.carto_process = subprocess.Popen(
                shlex.split(cmd),
                start_new_session=True,
                stdout=None,
                stderr=None
            )
            self.get_logger().info(f"Cartographer started with PID: {self.carto_process.pid}")
        except Exception as e:
            self.get_logger().error(f"Failed to start cartographer: {e}")

    def start_navigation(self):
        with self.map_name_lock:
            map_name_raw = self.mapName

        if self.nav_process and self.nav_process.poll() is None:
            self.get_logger().info("Navigation is already running (managed)")
            return
        if self.is_navigation_running():
            self.get_logger().warn("Navigation is running externally, cannot start")
            return
        if self.carto_process and self.carto_process.poll() is None:
            self.get_logger().warn("Cartographer is running, cannot start navigation")
            return
        if self.is_cartographer_running():
            self.get_logger().warn("Cartographer is running externally, cannot start navigation")
            return

        try:
            map_name_with_suffix = self.ensure_yaml_suffix(map_name_raw)
            # 加载地图的绝对路径（您的习惯：maps目录）
            map_abs_path = f"/home/jetson/ros2_ws/src/navigation2/nav2_bringup/maps/{map_name_with_suffix}"
            cmd = f"ros2 launch nav2_bringup navigation2.launch.py map:={map_abs_path}"
            self.get_logger().info(f"Launching Navigation: {cmd}")

            self.nav_process = subprocess.Popen(
                shlex.split(cmd),
                start_new_session=True,
                stdout=None,
                stderr=None
            )
            self.get_logger().info(f"Navigation started with PID: {self.nav_process.pid}")
        except Exception as e:
            self.get_logger().error(f"Failed to start navigation: {e}")

    def save_map(self):
        with self.map_save_lock:
            with self.map_name_lock:
                base_name = self.mapName
            clean_name = self.strip_yaml_suffix(base_name)
            self._save_map_impl(clean_name)

    def _save_map_impl(self, clean_name):
        try:
            # 保存地图的绝对路径（您的习惯：ros2_ws根目录）
            save_dir = "/home/jetson/ros2_ws"
            save_base = os.path.join(save_dir, clean_name)
            cmd = f"ros2 run nav2_map_server map_saver_cli -f {save_base}"
            self.get_logger().info(f"Running map saver: {cmd}")

            self.map_save_process = subprocess.Popen(
                shlex.split(cmd),
                stdout=None,
                stderr=None
            )
            self.get_logger().info(f"Map saver started with PID: {self.map_save_process.pid}")

            try:
                self.map_save_process.wait(timeout=45)
                if self.map_save_process.returncode == 0:
                    self.get_logger().info("Map saved successfully!")
                    time.sleep(1)
                    map_yaml_path = f"{save_base}.yaml"
                    if os.path.exists(map_yaml_path):
                        result = String()
                        result.data = 'map saved Successfully'
                        self.saveMapResultPub.publish(result)
                        self.get_logger().info("Map saved result published")
                        self.get_logger().info(f"Map files saved to: {save_dir}/{clean_name}.yaml and .pgm")
                        self.get_logger().info("Note: To use this map for navigation, copy both files to /home/jetson/ros2_ws/src/navigation2/nav2_bringup/maps/")
                    else:
                        self.get_logger().warn(f"Map file not found at {map_yaml_path}")
                else:
                    self.get_logger().error("Map saver returned non-zero exit code")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Map saver timed out, terminating...")
                self._terminate_map_saver()
        except Exception as e:
            self.get_logger().error(f"Failed to run map saver: {e}")
        finally:
            self._terminate_map_saver()

    def _terminate_map_saver(self):
        if self.map_save_process and self.map_save_process.poll() is None:
            try:
                self.map_save_process.terminate()
                self.map_save_process.wait(timeout=2)
                self.get_logger().info("Map saver terminated")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Force killing map saver")
                self.map_save_process.kill()
                self.map_save_process.wait()
            except Exception as e:
                self.get_logger().error(f"Error terminating map saver: {e}")
        self.map_save_process = None

    def stop_cartographer(self):
        if self.carto_process is None:
            self.get_logger().warn("No cartographer process to stop")
            return
        if self.carto_process.poll() is not None:
            self.get_logger().warn("Cartographer already terminated")
            self.carto_process = None
            return

        pgid = None
        try:
            pgid = os.getpgid(self.carto_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.get_logger().info(f"Sent SIGTERM to cartographer group (PGID: {pgid})")
            self.carto_process.wait(timeout=5)
            self.get_logger().info("Cartographer stopped")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Cartographer did not terminate, force killing...")
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self.carto_process.kill()
                self.carto_process.wait(timeout=2)
                self.get_logger().info("Cartographer force killed")
            except Exception as e:
                self.get_logger().error(f"Force kill failed: {e}")
        except ProcessLookupError:
            self.get_logger().warn("Cartographer process already gone")
        except Exception as e:
            self.get_logger().error(f"Error stopping cartographer: {e}")
        finally:
            self.carto_process = None

    def stop_navigation(self):
        if self.nav_process is None:
            self.get_logger().warn("No navigation process to stop")
            return
        if self.nav_process.poll() is not None:
            self.get_logger().warn("Navigation already terminated")
            self.nav_process = None
            return

        pgid = None
        try:
            pgid = os.getpgid(self.nav_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.get_logger().info(f"Sent SIGTERM to navigation group (PGID: {pgid})")
            self.nav_process.wait(timeout=5)
            self.get_logger().info("Navigation stopped")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Navigation did not terminate, force killing...")
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self.nav_process.kill()
                self.nav_process.wait(timeout=2)
                self.get_logger().info("Navigation force killed")
            except Exception as e:
                self.get_logger().error(f"Force kill failed: {e}")
        except ProcessLookupError:
            self.get_logger().warn("Navigation process already gone")
        except Exception as e:
            self.get_logger().error(f"Error stopping navigation: {e}")
        finally:
            self.nav_process = None

    def cleanup_all_processes(self):
        if self.carto_process and self.carto_process.poll() is None:
            self.stop_cartographer()
        if self.nav_process and self.nav_process.poll() is None:
            self.stop_navigation()
        if self.map_save_process and self.map_save_process.poll() is None:
            self._terminate_map_saver()

    # ---------- 回调 ----------
    def control_callback(self, msg):
        if msg.data == "1":
            self.start_cartographer()
        elif msg.data.startswith('2'):
            raw_name = msg.data.split(',')[1]
            with self.map_name_lock:
                self.mapName = self.strip_yaml_suffix(raw_name)
            threading.Thread(target=self.save_map).start()
        elif msg.data == "3":
            self.stop_cartographer()
        elif msg.data.startswith('4'):
            raw_name = msg.data.split(',')[1]
            with self.map_name_lock:
                self.mapName = raw_name
            self.start_navigation()
        elif msg.data == "5":
            self.stop_navigation()
        else:
            self.get_logger().warn(f"Unknown command: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down by user request")
    finally:
        controller.cleanup_all_processes()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
