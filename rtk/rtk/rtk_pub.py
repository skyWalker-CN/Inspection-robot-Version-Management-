#!/usr/bin/env python3

import sys
import rclpy
import serial
import time
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus

class GPSPublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(NavSatFix, '/fix', 10)
        
        # 串口配置
        self.ser = None
        try:
            self.ser = serial.Serial(
                port='/dev/ttyCH341USB1',
                baudrate=115200,
                timeout=1
            )
            self.get_logger().info('Open port successfully')
        except Exception as e:
            self.get_logger().error(f'Open port failed: {e}')
            return
        
        # 初始化变量
        self.longitude = 0.0
        self.latitude = 0.0
        self.altitude = 0.0
        self.status = 0
        self.service = NavSatStatus.SERVICE_GPS
        self.frame = 'gps_frame'
        self.hdop = 0.0
        self.type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        
        # 统计信息
        self.message_count = 0
        self.last_log_time = time.time()

    def getData(self):
        """从串口读取并解析GPS数据"""
        if not self.ser or not self.ser.is_open:
            self.get_logger().error('Serial port not available')
            return False
            
        try:
            line = self.ser.readline()
            if not line:
                return False
                
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            
            if decoded_line and decoded_line.startswith("$GNGGA"):
                # 解析GNGGA数据
                parts = decoded_line.split(',')
                if len(parts) < 10:
                    return False
                
                # 解析纬度（格式：DDMM.MMMMM）
                lat_str = parts[2]
                if lat_str and len(lat_str) >= 4:
                    lat_deg = float(lat_str[:2])
                    lat_min = float(lat_str[2:])
                    self.latitude = round(lat_deg + lat_min / 60.0, 8)
                
                # 解析经度（格式：DDDMM.MMMMM）
                lon_str = parts[4]
                if lon_str and len(lon_str) >= 5:
                    lon_deg = float(lon_str[:3])
                    lon_min = float(lon_str[3:])
                    self.longitude = round(lon_deg + lon_min / 60.0, 8)
                
                # 解析海拔高度
                alt_str = parts[9]
                if alt_str:
                    self.altitude = float(alt_str) if alt_str else 0.0
                
                # 解析定位质量
                qual_str = parts[6]
                if qual_str:
                    qual = int(qual_str) if qual_str else 0
                    if qual == 0:
                        self.status = NavSatStatus.STATUS_NO_FIX
                    else:
                        self.status = NavSatStatus.STATUS_FIX
                
                # 解析HDOP
                hdop_str = parts[8]
                if hdop_str:
                    self.hdop = float(hdop_str) if hdop_str else 0.0
                
                return True
                
        except UnicodeDecodeError:
            self.get_logger().warning('Unicode decode error in GPS data')
        except ValueError as e:
            self.get_logger().warning(f'Value error parsing GPS data: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in getData: {e}')
            
        return False

    def publishData(self):
        """发布GPS数据到ROS话题"""
        # 创建状态消息
        status = NavSatStatus()
        status.status = self.status
        status.service = self.service

        # 创建NavSatFix消息
        msg = NavSatFix()
        
        # 使用当前时间作为时间戳
        current_time = self.get_clock().now()
        msg.header.stamp = current_time.to_msg()
        msg.header.frame_id = self.frame

        msg.status = status
        msg.latitude = self.latitude
        msg.longitude = self.longitude
        msg.altitude = self.altitude

        # 设置协方差矩阵（基于HDOP）
        hdop_sq = self.hdop ** 2
        msg.position_covariance = [
            hdop_sq, 0.0, 0.0,
            0.0, hdop_sq, 0.0,
            0.0, 0.0, hdop_sq * 4  # 高度通常有更大不确定性
        ]
        msg.position_covariance_type = self.type

        # 发布消息
        self.pub.publish(msg)
        self.message_count += 1
        
        # 每分钟记录一次统计信息
        current_time = time.time()
        if current_time - self.last_log_time > 60:
            self.get_logger().info(f'Published {self.message_count} GPS messages')
            self.last_log_time = current_time

    def run(self):
        """主运行循环"""
        if not self.ser or not self.ser.is_open:
            self.get_logger().error('Cannot start: serial port not available')
            return
            
        self.get_logger().info('GPS publisher started')
        
        # 创建ROS定时器而不是使用while循环
        self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        # 保持节点运行
        rclpy.spin(self)

    def timer_callback(self):
        """定时器回调函数"""
        if self.getData():
            self.publishData()
            
            # 调试信息（可注释掉以减少日志输出）
            self.get_logger().debug(
                f'Lat: {self.latitude:.6f}, Lon: {self.longitude:.6f}, '
                f'Alt: {self.altitude:.1f}, Status: {self.status}'
            )

    def destroy_node(self):
        """清理资源"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GPSPublisher("gps_publisher")
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('GPS publisher stopped by user')
    except Exception as e:
        node.get_logger().error(f'GPS publisher error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
