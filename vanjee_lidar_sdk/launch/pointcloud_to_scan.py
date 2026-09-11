#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
import numpy as np
import time
import struct
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class PointCloudToLaserScan(Node):
    def __init__(self):
        super().__init__('pointcloud_to_laserscan')
        
        # ²ÎÊý
        self.min_height = 0.2
        self.max_height = 2.0
        self.angle_min = -3.14
        self.angle_max = 3.14
        self.angle_increment = 0.0349066
        self.range_min = 0.3
        self.range_max = 70.0
        
        # Ô¤¼ÆËã
        self.num_readings = int((self.angle_max - self.angle_min) / self.angle_increment)
        self.angle_inv_inc = 1.0 / self.angle_increment
        
        # Îª¶©ÔÄÕßÉèÖÃQoS - Óë´«¸ÐÆ÷Êý¾Ý¼æÈÝ
        qos_profile_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # Îª·¢²¼ÕßÉèÖÃQoS - ÓëRViz2¼æÈÝ
        qos_profile_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # RViz2ÐèÒªRELIABLE
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # ¶©ÔÄµãÔÆ - Ê¹ÓÃBEST_EFFORT
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.pointcloud_callback,
            qos_profile_sub
        )
        
        # ·¢²¼LaserScan - Ê¹ÓÃRELIABLE
        self.publisher = self.create_publisher(
            LaserScan,
            '/scan',
            qos_profile_pub
        )
        
        # ÐÔÄÜÍ³¼Æ
        self.frame_count = 0
        self.start_time = time.time()
        
        self.get_logger().info('µãÔÆ×ª¼¤¹âÉ¨ÃèÆô¶¯£¬QoS: ¶©ÔÄ(BEST_EFFORT), ·¢²¼(RELIABLE)')
    
    def extract_points_fast(self, msg):
        """¿ìËÙÌáÈ¡µãÔÆÊý¾Ý"""
        data = msg.data
        point_step = msg.point_step
        
        # ²éÕÒ×Ö¶ÎÆ«ÒÆ
        x_offset = 0
        y_offset = 4
        z_offset = 8
        
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        # ¼ÆËãµãÊý
        num_points = len(data) // point_step
        
        # Èç¹ûµãÌ«¶à£¬½øÐÐ½µ²ÉÑù
        if num_points > 50000:
            step = 2
        else:
            step = 1
        
        # Ô¤·ÖÅäÊý×é
        valid_points = num_points // step
        x = np.empty(valid_points, dtype=np.float32)
        y = np.empty(valid_points, dtype=np.float32)
        z = np.empty(valid_points, dtype=np.float32)
        
        # ÅúÁ¿ÌáÈ¡
        for i in range(0, num_points, step):
            idx = i // step
            if idx >= valid_points:
                break
            start = i * point_step
            x[idx] = struct.unpack('f', data[start+x_offset:start+x_offset+4])[0]
            y[idx] = struct.unpack('f', data[start+y_offset:start+y_offset+4])[0]
            z[idx] = struct.unpack('f', data[start+z_offset:start+z_offset+4])[0]
        
        return x, y, z
    
    def pointcloud_callback(self, msg):
        """´¦ÀíµãÔÆ»Øµ÷"""
        start_time = time.time()
        
        try:
            # ½âÎöµãÔÆ
            x, y, z = self.extract_points_fast(msg)
            
            # ¸ß¶È¹ýÂË
            height_mask = (z >= self.min_height) & (z <= self.max_height)
            x = x[height_mask]
            y = y[height_mask]
            z = z[height_mask]
            
            if len(x) == 0:
                # ´´½¨²¢·¢²¼¿ÕÊý¾Ý
                scan = self.create_scan_msg(msg)
                self.publisher.publish(scan)
                return
            
            # ¼ÆËã¾àÀëÆ½·½
            dist_sq = x*x + y*y
            
            # ¾àÀë¹ýÂË
            dist_mask = (dist_sq >= self.range_min*self.range_min) & \
                       (dist_sq <= self.range_max*self.range_max)
            x = x[dist_mask]
            y = y[dist_mask]
            dist_sq = dist_sq[dist_mask]
            
            if len(x) == 0:
                scan = self.create_scan_msg(msg)
                self.publisher.publish(scan)
                return
            
            # ¼ÆËã½Ç¶È
            angles = np.arctan2(y, x)
            
            # ¼ÆËãË÷Òý
            indices = ((angles - self.angle_min) * self.angle_inv_inc).astype(int)
            
            # È·±£Ë÷ÒýÔÚ·¶Î§ÄÚ
            indices = np.clip(indices, 0, self.num_readings-1)
            
            # ¼ÆËã¾àÀë
            distances = np.sqrt(dist_sq)
            
            # ÎªÃ¿¸ö½Ç¶ÈË÷ÒýÕÒµ½×îÐ¡¾àÀë
            ranges = np.full(self.num_readings, np.inf, dtype=np.float32)
            
            # Ê¹ÓÃ¼òµ¥µ«¸ßÐ§µÄ·½·¨
            for i in range(self.num_readings):
                mask = indices == i
                if np.any(mask):
                    ranges[i] = np.min(distances[mask])
            
            # ´´½¨²¢·¢²¼ÏûÏ¢
            scan = self.create_scan_msg(msg)
            scan.ranges = ranges.tolist()
            self.publisher.publish(scan)
            
        except Exception as e:
            self.get_logger().error(f'´¦Àí´íÎó: {e}', throttle_duration_sec=1.0)
            # ³ö´íÊ±·¢²¼¿ÕÊý¾Ý
            scan = self.create_scan_msg(msg)
            self.publisher.publish(scan)
        
        # ÐÔÄÜÍ³¼Æ
        proc_time = time.time() - start_time
        self.frame_count += 1
        
        if self.frame_count % 20 == 0:
            current_time = time.time()
            time_interval = current_time - self.start_time
            if time_interval > 0:
                fps = 20.0 / time_interval
                self.start_time = current_time
                self.get_logger().info(f'´¦ÀíÊ±¼ä: {proc_time*1000:.1f}ms, ÆµÂÊ: {fps:.1f}Hz')
    
    def create_scan_msg(self, msg):
        """´´½¨LaserScanÏûÏ¢"""
        scan = LaserScan()
        scan.header = msg.header
        scan.header.frame_id = 'laser_frame'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = [float('inf')] * self.num_readings
        return scan

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToLaserScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
