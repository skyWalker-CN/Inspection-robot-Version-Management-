import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class RobotStatus(Node):
    def __init__(self):
        super().__init__('serial_reader_node')
        
        # 创建发布者
        self.status_pub = self.create_publisher(String, 'robot_status', 10)
        
        # 创建发布者
        self.bad_pub = self.create_publisher(String, '/environment/alerts', 10)
        
        # 串口配置参数
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('timeout', 1.0)
        
        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        timeout = self.get_parameter('timeout').value
        
        self.get_logger().info(f"尝试连接串口: {port}, 波特率: {baudrate}")
        
        try:
            # 打开串口设备
            self.ser = serial.Serial(
                port = port,
                baudrate = baudrate,
                timeout = timeout
            )
            
            if self.ser.is_open:
                self.get_logger().info(f"成功连接串口 {port}")
            else:
                self.get_logger().error("串口连接失败")
                return
                
        except serial.SerialException as e:
            self.get_logger().error(f"串口连接错误: {e}")
            return
        
        # 创建定时器来定期读取串口数据
        self.timer = self.create_timer(0.01, self.read_serial_callback)  # 100Hz
        self.get_logger().info("串口读取节点已启动，开始发布 robot_status 话题")
    
    def read_serial_callback(self):
        """定时器回调函数，读取串口数据并发布到ROS2话题"""
        try:
            if self.ser.in_waiting > 0:
                # 读取一行数据
                data_line = self.ser.readline()
                
                try:
                    # 尝试解码为UTF-8
                    decoded_data = data_line.decode('utf-8').strip()
                    
                    if decoded_data:
                        # 创建ROS2消息
                        msg = String()
                        msg.data = decoded_data
                        
                        # 发布消息
                        self.status_pub.publish(msg)
                        
                        # 可选：在控制台显示接收到的数据
                        self.get_logger().info(f"发布数据: {decoded_data}")
                        
                        # 按 | 分割数据（根据您的需求）
                        if " | " in decoded_data:
                            bad_msg = String()
                            data_split = decoded_data.split(" | ")
                            status_02 = data_split[1]
                            status_135 = data_split[3]
                            self.get_logger().info(f"分割数据: {data_split}")
                            if status_02 == "1" and status_135  == "0":
                                bad_msg.data = "警告！警告！检测到异常浓度的可燃气体或烟雾！"
                                self.bad_pub.publish(bad_msg)
                            elif status_02 == "0" and status_135 == "1":
                                bad_msg.data = "警告！警告！检测到异常浓度的氨气或硫化物！"
                                self.bad_pub.publish(bad_msg)
                            elif status_02 == "1" and status_135 == "1":
                                bad_msg.data = "警告！警告！检测到异常浓度的可燃气体或烟雾以及异常浓度的氨气或硫化物！"
                                self.bad_pub.publish(bad_msg)
                            else:
                                pass
                            
                            
                        
                except UnicodeDecodeError:
                    # 如果UTF-8解码失败，发布十六进制数据
                    hex_data = data_line.hex()
                    msg = String()
                    msg.data = f"HEX: {hex_data}"
                    self.publisher_.publish(msg)
                    self.get_logger().warn(f"UTF-8解码失败，发布十六进制数据: {hex_data}")
                    
        except Exception as e:
            self.get_logger().error(f"读取串口数据错误: {e}")
    
    def destroy_node(self):
        """重写销毁方法，确保串口正确关闭"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("串口已关闭")
        super().destroy_node()

def main(args=None):
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 创建节点
    serial_node = RobotStatus()
    
    try:
        # 运行节点
        rclpy.spin(serial_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        serial_node.get_logger().error(f"节点运行错误: {e}")
    finally:
        # 清理资源
        serial_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
