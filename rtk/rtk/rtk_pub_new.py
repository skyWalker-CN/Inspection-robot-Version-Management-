#!/usr/bin/env python3

import sys
import rclpy
import serial
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus

class KalmanFilter:
    """简单的卡尔曼滤波器实现"""
    
    def __init__(self, process_variance=1e-3, measurement_variance=0.1**2):
        # 过程方差（描述模型的不确定性）
        self.process_variance = process_variance
        # 测量方差（描述传感器噪声）
        self.measurement_variance = measurement_variance
        # 后验估计误差方差
        self.posteri_estimate_error = 1.0
        # 先验估计误差方差
        self.priori_estimate_error = 1.0
        # 卡尔曼增益
        self.kalman_gain = 0.0
        # 后验估计值（滤波后的值）
        self.posteri_estimate = 0.0
        # 先验估计值
        self.priori_estimate = 0.0
    
    def update(self, measurement):
        """更新卡尔曼滤波器"""
        # 预测步骤
        self.priori_estimate = self.posteri_estimate
        self.priori_estimate_error = self.posteri_estimate_error + self.process_variance
        
        # 更新步骤
        self.kalman_gain = self.priori_estimate_error / (self.priori_estimate_error + self.measurement_variance)
        self.posteri_estimate = self.posteri_estimate + self.kalman_gain * (measurement - self.priori_estimate)
        self.posteri_estimate_error = (1 - self.kalman_gain) * self.priori_estimate_error
        
        return self.posteri_estimate
    
    def reset(self, initial_value):
        """重置滤波器状态"""
        self.posteri_estimate = initial_value
        self.priori_estimate = initial_value
        self.posteri_estimate_error = 1.0
        self.priori_estimate_error = 1.0
        self.kalman_gain = 0.0

class GPSPublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(NavSatFix, '/fix', 10)
        self.pub_raw = self.create_publisher(NavSatFix, '/fix_raw', 10)  # 新增：发布原始数据
        
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
        
        # 初始化卡尔曼滤波器
        # 为经度、纬度、海拔分别创建滤波器
        self.kf_longitude = KalmanFilter(process_variance=1e-5, measurement_variance=1e-4)
        self.kf_latitude = KalmanFilter(process_variance=1e-5, measurement_variance=1e-4)
        self.kf_altitude = KalmanFilter(process_variance=1e-2, measurement_variance=1e-1)
        
        # 初始化变量
        self.raw_longitude = 0.0  # 原始经度
        self.raw_latitude = 0.0   # 原始纬度
        self.raw_altitude = 0.0  # 原始海拔
        
        self.filtered_longitude = 0.0  # 滤波后经度
        self.filtered_latitude = 0.0   # 滤波后纬度
        self.filtered_altitude = 0.0   # 滤波后海拔
        
        self.status = 0
        self.service = NavSatStatus.SERVICE_GPS
        self.frame = 'gps_frame'
        self.hdop = 0.0
        self.type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        
        # 滤波器状态标志
        self.filter_initialized = False
        self.first_fix_received = False
        
        # 统计信息
        self.message_count = 0
        self.last_log_time = time.time()
        self.filter_improvement_stats = []  # 记录滤波改善效果

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
                
                # 检查定位质量
                qual_str = parts[6]
                if not qual_str or int(qual_str) == 0:
                    self.get_logger().debug('No GPS fix available')
                    return False
                
                # 解析原始纬度（格式：DDMM.MMMMM），保留8位小数
                lat_str = parts[2]
                lat_dir = parts[3]  # N or S
                if lat_str and len(lat_str) >= 4:
                    lat_deg = float(lat_str[:2])
                    lat_min = float(lat_str[2:])
                    self.raw_latitude = round(lat_deg + lat_min / 60.0, 8)  # 保留8位小数
                    if lat_dir == 'S':
                        self.raw_latitude = -self.raw_latitude
                
                # 解析原始经度（格式：DDDMM.MMMMM），保留8位小数
                lon_str = parts[4]
                lon_dir = parts[5]  # E or W
                if lon_str and len(lon_str) >= 5:
                    lon_deg = float(lon_str[:3])
                    lon_min = float(lon_str[3:])
                    self.raw_longitude = round(lon_deg + lon_min / 60.0, 8)  # 保留8位小数
                    if lon_dir == 'W':
                        self.raw_longitude = -self.raw_longitude
                
                # 解析原始海拔高度，改为保留4位小数[2,6](@ref)
                alt_str = parts[9]
                if alt_str:
                    self.raw_altitude = round(float(alt_str) if alt_str else 0.0, 4)  # 改为保留4位小数
                
                # 解析定位质量
                if qual_str:
                    qual = int(qual_str) if qual_str else 0
                    if qual == 0:
                        self.status = NavSatStatus.STATUS_NO_FIX
                    else:
                        self.status = NavSatStatus.STATUS_FIX
                
                # 解析HDOP，保留8位小数
                hdop_str = parts[8]
                if hdop_str:
                    self.hdop = round(float(hdop_str) if hdop_str else 0.0, 8)  # 保留8位小数
                
                # 应用卡尔曼滤波
                self.apply_kalman_filter()
                
                return True
                
        except UnicodeDecodeError:
            self.get_logger().warning('Unicode decode error in GPS data')
        except ValueError as e:
            self.get_logger().warning(f'Value error parsing GPS data: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in getData: {e}')
            
        return False

    def apply_kalman_filter(self):
        """应用卡尔曼滤波处理GPS数据"""
        if not self.first_fix_received:
            # 第一次收到有效定位数据，初始化滤波器
            self.kf_longitude.reset(self.raw_longitude)
            self.kf_latitude.reset(self.raw_latitude)
            self.kf_altitude.reset(self.raw_altitude)
            self.filtered_longitude = round(self.raw_longitude, 8)  # 保留8位小数
            self.filtered_latitude = round(self.raw_latitude, 8)     # 保留8位小数
            self.filtered_altitude = round(self.raw_altitude, 4)     # 改为保留4位小数[6,7](@ref)
            self.first_fix_received = True
            self.get_logger().info('Kalman filter initialized with first GPS fix')
        else:
            # 应用卡尔曼滤波
            # 经度纬度保留8位小数，海拔保留4位小数[2,8](@ref)
            self.filtered_longitude = round(self.kf_longitude.update(self.raw_longitude), 8)
            self.filtered_latitude = round(self.kf_latitude.update(self.raw_latitude), 8)
            self.filtered_altitude = round(self.kf_altitude.update(self.raw_altitude), 4)  # 改为保留4位小数
            
            # 记录滤波效果（用于调试）
            if self.message_count % 100 == 0:
                lon_diff = abs(self.filtered_longitude - self.raw_longitude) * 111000  # 转换为米
                lat_diff = abs(self.filtered_latitude - self.raw_latitude) * 111000   # 转换为米
                alt_diff = abs(self.filtered_altitude - self.raw_altitude)
                self.get_logger().info(
                    f'Filter correction - Lon: {lon_diff:.8f}m, '  # 显示8位小数
                    f'Lat: {lat_diff:.8f}m, Alt: {alt_diff:.4f}m'  # 海拔改为显示4位小数[6](@ref)
                )

    def publishData(self):
        """发布GPS数据到ROS话题"""
        if not self.first_fix_received:
            return
            
        # 创建状态消息
        status = NavSatStatus()
        status.status = self.status
        status.service = self.service

        # 发布原始数据（用于对比）
        raw_msg = NavSatFix()
        current_time = self.get_clock().now()
        raw_msg.header.stamp = current_time.to_msg()
        raw_msg.header.frame_id = self.frame + "_raw"
        raw_msg.status = status
        raw_msg.latitude = round(self.raw_latitude, 8)  # 保留8位小数
        raw_msg.longitude = round(self.raw_longitude, 8)  # 保留8位小数
        raw_msg.altitude = round(self.raw_altitude, 4)  # 改为保留4位小数[2,7](@ref)
        self.pub_raw.publish(raw_msg)

        # 发布滤波后的数据
        filtered_msg = NavSatFix()
        filtered_msg.header.stamp = current_time.to_msg()
        filtered_msg.header.frame_id = self.frame
        filtered_msg.status = status
        filtered_msg.latitude = round(self.filtered_latitude, 8)  # 保留8位小数
        filtered_msg.longitude = round(self.filtered_longitude, 8)  # 保留8位小数
        filtered_msg.altitude = round(self.filtered_altitude, 4)  # 改为保留4位小数[6,8](@ref)

        # 设置协方差矩阵（基于HDOP，滤波后不确定性降低），保留8位小数
        hdop_sq = round((self.hdop ** 2) * 0.5, 8)  # 滤波后误差减小，保留8位小数
        filtered_msg.position_covariance = [
            hdop_sq, 0.0, 0.0,
            0.0, hdop_sq, 0.0,
            0.0, 0.0, round(hdop_sq * 2, 8)  # 高度不确定性也降低，保留8位小数
        ]
        filtered_msg.position_covariance_type = self.type

        # 发布滤波后的消息
        self.pub.publish(filtered_msg)
        self.message_count += 1
        
        # 每分钟记录一次统计信息
        current_time = time.time()
        if current_time - self.last_log_time > 60:
            self.get_logger().info(
                f'Published {self.message_count} GPS messages '
                f'(Raw: {self.raw_latitude:.8f}, {self.raw_longitude:.8f}, '  # 显示8位小数
                f'Filtered: {self.filtered_latitude:.8f}, {self.filtered_longitude:.8f})'  # 显示8位小数
            )
            self.last_log_time = current_time

    def run(self):
        """主运行循环"""
        if not self.ser or not self.ser.is_open:
            self.get_logger().error('Cannot start: serial port not available')
            return
            
        self.get_logger().info('GPS publisher with Kalman filter started')
        
        # 创建ROS定时器
        self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        # 保持节点运行
        rclpy.spin(self)

    def timer_callback(self):
        """定时器回调函数"""
        if self.getData():
            self.publishData()
            
            # 调试信息（可注释掉以减少日志输出）
            if self.message_count % 50 == 0:  # 每50条消息输出一次调试信息
                self.get_logger().debug(
                    f'Raw -> Lat: {self.raw_latitude:.8f}, Lon: {self.raw_longitude:.8f}, '  # 显示8位小数
                    f'Alt: {self.raw_altitude:.4f}\n'  # 海拔改为显示4位小数[7](@ref)
                    f'Filtered -> Lat: {self.filtered_latitude:.8f}, Lon: {self.filtered_longitude:.8f}, '  # 显示8位小数
                    f'Alt: {self.filtered_altitude:.4f}, Status: {self.status}'  # 海拔改为显示4位小数[6](@ref)
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
