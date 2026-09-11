#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import struct
import math
import threading
from std_msgs.msg import String


class UltraSensor(Node):
    def __init__(self):
        super().__init__('ultra_sensor')
        

        # 初始化串口连接
        try:
            self.ser = serial.Serial(
            port = '/dev/ttyCH341USB0',
            baudrate = 9600,
            bytesize = serial.EIGHTBITS,
            parity = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE,
            timeout = 1 # 读超时时间，单位为秒
        )
            self.get_logger().info(f"Connected to ultra_sensor at /dev/ttyUSB1")
        except serial.SerialException as e:
            self.get_logger().error(f"Serial connection error: {str(e)}")
            raise SystemExit
        
        # 创建订阅和发布
        self.sensor_pub = self.create_publisher(
            String,
            'ultarSensor',
            10)
                
        # 启动串口读取线程
        self.serial_thread = threading.Thread(target=self.read_serial)
        self.serial_thread.daemon = True
        self.serial_thread.start()

    def read_serial(self):
        """串口数据读取线程"""
        while True:
            # 读取 exactly 10 字节
            raw_data = self.ser.read(10)
            print(raw_data)
                
            if len(raw_data) != 10:
                    print(f"数据长度不足: 期望10字节, 实际收到 {len(raw_data)} 字节。继续等待...")
                    continue

            # 检查帧头 (第一个字节)
            if raw_data[0] != 0xFF:
                # 如果帧头不对，可以尝试在数据流中重新同步，这里简单跳过
                print(f"帧头错误: 期望 0xFF, 实际收到 {hex(raw_data[0])}。跳过此帧。")
                continue

            # 解析中间的四组数据（每组2字节）
            # 使用 struct.unpack 来解析二进制数据。格式字符串 '<4H' 表示4个无符号短整数（16位），小端字节序 (低位字节在前)
            # 如果你设备的数据是 大端字节序 (高位字节在前)，请将格式字符串改为 '>4H'
            # raw_data[1:9] 获取从索引1到索引8的字节（共8字节），即中间的四组数据
            try:
                group1, group2, group3, group4 = struct.unpack('>4H', raw_data[1:9])
            except struct.error as e:
                print(f"解析数据时发生错误: {e}")
                continue

            # 打印解析结果
            print(f"解析成功 -> 组1: {group1}, 组2: {group2}, 组3: {group3}, 组4: {group4}")

            ultra_sensor = str(group1 / 1000) + 'm ' + str(group2 / 1000) + 'm ' + str(group3 / 1000) + 'm ' + str(group4 / 1000) + 'm'
            ultra_msg = String()
            ultra_msg.data = ultra_sensor

            self.sensor_pub.publish(ultra_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = UltraSensor()
    
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
