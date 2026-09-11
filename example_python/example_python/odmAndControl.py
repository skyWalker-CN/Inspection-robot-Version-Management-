#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import serial
import math
from tf_transformations import quaternion_from_euler
import threading
from serial.serialutil import SerialException
from tf2_ros.transform_broadcaster import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        # ROS参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('serial_port', '/dev/ttyCH341USB2'),
                ('baudrate', 230400),
                ('wheel_separation', 0.6),
                ('wheel_radius', 0.12),
                ('cmd_vel_topic', '/cmd_vel'),
                ('odom_topic', '/odom'),
                ('amcl_topic','/amcl_pose'),
                ('robotLocation_topic','/robot_location'),
                ('angle_range_deg', 30.0)
            ]
        )
        
        # 获取参数
        serial_port = self.get_parameter('serial_port').value
        baudrate = self.get_parameter('baudrate').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        
        # 初始化串口连接
        try:
            self.ser = serial.Serial(
                port=serial_port,
                baudrate=baudrate,
                timeout=0
            )
            self.get_logger().info(f"Connected to motor controller at {serial_port}")
        except SerialException as e:
            self.get_logger().error(f"Serial connection error: {str(e)}")
            raise SystemExit
        
        # 里程计变量
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.last_time = self.get_clock().now()
        self.lock = threading.Lock()  # 线程锁

        # 控制机器人指令
        self.hex_str_start = "ABCD"
        self.hex_str_LRPM = ""
        self.hex_str_RRPM = ""
        self.LSpeed = 0
        self.RSpeed = 0
        self.hex_str_check = ""
        self.hex_str = ""
        self.COMMAND = ""
        
        # 创建订阅和发布
        self.subscription = self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            self.cmd_vel_callback,
            10)
        self.amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('amcl_topic').value,
            self.amcl_callback,
            10)
        self.robotLocation_pub = self.create_publisher(
            String,
            self.get_parameter('robotLocation_topic').value,
            10)
        self.odom_pub = self.create_publisher(
            Odometry,
            self.get_parameter('odom_topic').value,
            10)
        
        # 坐标变换
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 启动串口读取线程
        self.serial_thread = threading.Thread(target=self.read_serial)
        self.serial_thread.daemon = True
        self.serial_thread.start()

        self.angle_range_deg = self.get_parameter('angle_range_deg').value

        # 创建订阅者
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10  # QoS profile depth
        )

        self.subscription_1 = self.create_subscription(
            Twist,
            'cmd_vel_1',
            self.cmd_vel_1_callback,
            10)

        self.front_dis = ''
        self.back_dis = ''

    def scan_callback(self, msg):
        """
        处理激光雷达数据的回调函数
        """
        # 1. 计算需要检测的扇形区域的索引范围
        # 将角度从度转换为弧度
        angle_range_rad = math.radians(self.angle_range_deg)
        
        # 计算正前方（0度）方向的索引
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        num_readings = len(msg.ranges)
        
        # 正前方（0弧度）的索引
        front_center_idx = int((0.0 - angle_min) / angle_inc)
        # 正后方（π弧度）的索引
        rear_center_idx = int((math.pi - angle_min) / angle_inc) % num_readings

        # 计算扇形区域左右的索引偏移量（基于角度范围）
        idx_offset = int(angle_range_rad / angle_inc)
        
        # 正前方区域索引范围 [front_start_idx, front_end_idx]
        front_start_idx = front_center_idx - idx_offset
        front_end_idx = front_center_idx + idx_offset
        # 正后方区域索引范围 [rear_start_idx, rear_end_idx]
        rear_start_idx = rear_center_idx - idx_offset
        rear_end_idx = rear_center_idx + idx_offset

        # 2. 提取指定区域内的距离数据，并过滤无效值
        front_ranges = []
        rear_ranges = []
        
        # 处理正前方区域
        for i in range(front_start_idx, front_end_idx + 1):
            idx = i % num_readings  # 处理循环索引
            val = msg.ranges[idx]
            if not math.isinf(val) and not math.isnan(val) and (msg.range_min <= val <= msg.range_max):
                front_ranges.append(val)
                
        # 处理正后方区域
        for i in range(rear_start_idx, rear_end_idx + 1):
            idx = i % num_readings  # 处理循环索引
            val = msg.ranges[idx]
            if not math.isinf(val) and not math.isnan(val) and (msg.range_min <= val <= msg.range_max):
                rear_ranges.append(val)

        # 3. 计算区域内的最小距离（如果没有有效值，则视为无限远）
        min_front = min(front_ranges) if front_ranges else float('inf')
        min_rear = min(rear_ranges) if rear_ranges else float('inf')

        self.front_dis = min_rear
        self.back_dis = min_front

    def hex_xor(self, a, b, c):
        """对三个十六进制数进行异或操作"""
        a_int = int(a, 16)
        b_int = int(b, 16)
        c_int = int(c, 16)
        
        result = a_int ^ b_int ^ c_int
        hex_result = hex(result)[2:].upper().zfill(4)
        
        return hex_result

    def convert_hex_pattern(self, hex_str):
        """转换命令格式"""
        cleaned = hex_str.replace(" ", "").replace("0x", "").upper()
        
        if len(cleaned) != 16:
            raise ValueError("输入字符串必须为16个十六进制字符（8字节）")
            
        groups = [cleaned[i:i+4] for i in range(0, len(cleaned), 4)]
        converted_groups = []
        
        for group in groups:
            part1 = group[:2]
            part2 = group[2:]
            converted = part2 + part1
            converted_groups.append(converted)
            
        return "".join(converted_groups)
    
    def swap_hex_groups(self,hex_str):
        # 将字符串每4个字符分成一组
        groups = [hex_str[i:i+4] for i in range(0, len(hex_str), 4)]
        
        # 对每组中的两个2字符子组进行位置交换
        swapped_groups = []
        for group in groups:
            if len(group) == 4:
                # 将4字符组拆成两个2字符的子组并交换位置
                swapped = group[2:4] + group[0:2]
                swapped_groups.append(swapped)
            else:
                # 如果最后一组不足4个字符，则保持原样
                swapped_groups.append(group)
        
        # 合并所有组并返回结果
        return ''.join(swapped_groups)

    def amcl_callback(self,msg):
        print(msg.pose.pose.position.x)
        print(msg.pose.pose.position.y)
        robotLocation_msg = String()
        robotLocation = str(msg.pose.pose.position.x) + "," + str(msg.pose.pose.position.y)
        robotLocation_msg.data = robotLocation
        self.robotLocation_pub.publish(robotLocation_msg)
    
    def cmd_vel_callback(self, msg):
        """接收速度指令并发送给电机驱动板"""
        if msg.linear.x == 0.0 and msg.angular.z == 1.0:
            msg.angular.z = 2.0
        elif msg.linear.x == 0.0 and msg.angular.z == -1.0:
            msg.angular.z = -2.0
        # 计算轮速 (差速驱动模型)
        print("cmd:",msg.linear.x,msg.angular.z)
        left_speed = (msg.linear.x - msg.angular.z * self.wheel_separation / 2) 
        right_speed = (msg.linear.x + msg.angular.z * self.wheel_separation / 2)

        #print(left_speed,right_speed)

        left_RPM = int((30 * left_speed) / (math.pi * self.wheel_radius))
        right_RPM = int((30 * right_speed) / (math.pi * self.wheel_radius))

        if ((left_RPM > 0 and right_RPM > 0) and (left_RPM != 53 and right_RPM != 99)):
            left_RPM += 4
        #print("RPM:",left_RPM,right_RPM)

        #left_RPM = 50
        #right_RPM = 50


        #print("give:",left_RPM,right_RPM)

        # 计算左右轮速度的16进制表示
        if left_RPM < 0:
            #if left_RPM < -60:
                #left_RPM = -60
            self.hex_str_LRPM = f"{(left_RPM & 0xFFFF):04X}"
        else:
            #if left_RPM > 60:
                #left_RPM = 60
            self.hex_str_LRPM = f"{left_RPM:04X}"
            
        if right_RPM < 0:
            #if right_RPM < -60:
                #right_RPM = -60
            self.hex_str_RRPM = f"{(right_RPM & 0xFFFF):04X}"
        else:
            #if right_RPM > 60:
                #right_RPM = 60
            self.hex_str_RRPM = f"{right_RPM:04X}"

        self.hex_str_check = self.hex_xor(self.hex_str_start,self.hex_str_LRPM,self.hex_str_RRPM)

        hex_str = self.hex_str_start + self.hex_str_LRPM + self.hex_str_RRPM + self.hex_str_check

        self.hex_str = self.convert_hex_pattern(hex_str)

        self.COMMAND = bytes.fromhex(self.hex_str)
        #print(self.hex_str)
         
        # 转换为电机控制命令 (根据实际协议修改)
        #command = f"SPD {left_speed:.2f},{right_speed:.2f}\n".encode()
        
        try:
            with self.lock:
                self.ser.write(self.COMMAND)
        except Exception as e:
            self.get_logger().error(f"Command send error: {str(e)}")

    def cmd_vel_1_callback(self, msg):
        """接收速度指令并发送给电机驱动板"""
        # 计算轮速 (差速驱动模型)
        if msg.linear.x > 0.0:
            if math.isinf(float(self.front_dis)):
                msg.linear.x = 0.0
            elif float(self.front_dis) <= 1.0:
                msg.linear.x = 0.0

        if msg.linear.x < 0.0:
            if math.isinf(float(self.back_dis)):
                msg.linear.x = 0.0
            elif float(self.back_dis) <= 1.0:
                msg.linear.x = 0.0
            
        #print("cmd:",msg.linear.x,msg.angular.z)
        left_speed = (msg.linear.x - msg.angular.z * self.wheel_separation / 2)
        right_speed = (msg.linear.x + msg.angular.z * self.wheel_separation / 2)

        #print(left_speed,right_speed)

        left_RPM = int((30 * left_speed) / (math.pi * self.wheel_radius))
        right_RPM = int((30 * right_speed) / (math.pi * self.wheel_radius))

       # if ((left_RPM > 0 and right_RPM > 0) and (left_RPM != 53 and right_RPM != 99)):
            #left_RPM += 3

        print("RPM:",left_RPM,right_RPM)

        #left_RPM = -30
        #right_RPM = -30


        #print("give:",left_RPM,right_RPM)

        # 计算左右轮速度的16进制表示
        if left_RPM < 0:
            #if left_RPM < -60:
                #left_RPM = -60
            self.hex_str_LRPM = f"{(left_RPM & 0xFFFF):04X}"
        else:
            #if left_RPM > 60:
                #left_RPM = 60
            self.hex_str_LRPM = f"{left_RPM:04X}"

        if right_RPM < 0:
            #if right_RPM < -60:
                #right_RPM = -60
            self.hex_str_RRPM = f"{(right_RPM & 0xFFFF):04X}"
        else:
            #if right_RPM > 60:
                #right_RPM = 60
            self.hex_str_RRPM = f"{right_RPM:04X}"

        self.hex_str_check = self.hex_xor(self.hex_str_start,self.hex_str_LRPM,self.hex_str_RRPM)

        hex_str = self.hex_str_start + self.hex_str_LRPM + self.hex_str_RRPM + self.hex_str_check

        self.hex_str = self.convert_hex_pattern(hex_str)

        self.COMMAND = bytes.fromhex(self.hex_str)
        #print(self.hex_str)

        # 转换为电机控制命令 (根据实际协议修改)
        #command = f"SPD {left_speed:.2f},{right_speed:.2f}\n".encode()

        try:
            with self.lock:
                self.ser.write(self.COMMAND)
        except Exception as e:
            self.get_logger().error(f"Command send error: {str(e)}")

    # def parse_serial_data(self, data):
    #     try:
    #         hex_str_left = data[4:8]
    #         hex_str_right = data[8:12]
    #         #print(hex_str_left,hex_str_right)
    #         LRPM = int(hex_str_left,16)
    #         RRPM = int(hex_str_right,16)
    #         self.LSpeed = (LRPM * math.pi * self.wheel_radius) / 30
    #         self.RSpeed = (RRPM * math.pi * self.wheel_radius) / 30
    #         return self.LSpeed,self.RSpeed
    #     except Exception as e:
    #         self.get_logger().warn(f"Parse error: {str(e)}, Data: {data.decode().strip()}")
    #     return None
    
    def parse_serial_data(self, data):
        try:
            hex_str_left = data[4:8]
            hex_str_right = data[8:12]
        
        # 新增：16位补码转有符号整数
            def hex_to_signed(hex_str):
                num = int(hex_str, 16)
                return num - 0x10000 if num > 0x7FFF else num
            
            LRPM = hex_to_signed(hex_str_left)   # 处理负数
            RRPM = hex_to_signed(hex_str_right)  # 处理负数
        
            self.LSpeed = (LRPM * math.pi * self.wheel_radius) / 30
            self.RSpeed = (RRPM * math.pi * self.wheel_radius) / 30
            return self.LSpeed,self.RSpeed
        except Exception as e:
            self.get_logger().warn(f"Parse error: {str(e)}, Data: {data.decode().strip()}")
        return None    
   
    def update_odometry(self, left_speed, right_speed):
        """更新里程计信息"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # 计算线速度和角速度
        v = (left_speed + right_speed) / 2.0
        w = (right_speed - left_speed) / self.wheel_separation

        # 更新位置
        delta_x = v * math.cos(self.th) * dt
        delta_y = v * math.sin(self.th) * dt
        delta_th = w * dt

        with self.lock:  # 保护共享变量
            self.x += delta_x
            self.y += delta_y
            self.th += delta_th

        # 发布里程计信息
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        # 设置位置
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        # 设置方向 (转换为四元数)
        q = quaternion_from_euler(0, 0, self.th)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        # 设置速度
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w

        self.odom_pub.publish(odom)

        # 发布TF变换
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def read_serial(self):
        """串口数据读取线程"""
        self.get_logger().info("Serial reader thread started")
        while rclpy.ok():
            try:
                # 读取一行数据 (根据实际协议修改)
                with self.lock:
                    if self.ser.in_waiting > 0:
                        data = self.ser.read(self.ser.in_waiting).hex()
                        swap_data = self.swap_hex_groups(data)
                        #print(data)
                        print(swap_data)
                        print("rec:",int(swap_data[4:8],16)
,int(swap_data[8:12],16))
                
                # 解析速度数据
                speeds = self.parse_serial_data(swap_data)
                if speeds:
                    left_speed, right_speed = speeds
                    #left_speed = right_speed
                    #print("receive:",left_speed,right_speed)
                    self.update_odometry(left_speed, right_speed)
                    
            except Exception as e:
                self.get_logger().error(f"Serial read error: {str(e)}")
                break

def main(args=None):
    rclpy.init(args=args)
    controller = MotorController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
        if hasattr(controller, 'ser'):
            controller.ser.close()

if __name__ == '__main__':
    main()
