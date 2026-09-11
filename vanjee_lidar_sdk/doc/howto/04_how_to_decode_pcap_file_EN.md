# 4 How to decode PCAP file

## 4.1 Introduction

This document demonstrates how to decode PCAP files and send point cloud data to ROS.

Before reading this document, please ensure that you have read the LiDAR user manual and the [Parameter Introduction](../intro/02_parameter_intro_CN.md) .

## 4.2 Steps

### 4.2.1 Obtain data port number

Please refer to the LiDAR product manual or use third-party tools (such as Wireshark) to capture packets and determine the target port for the LiDAR. The default port value is `3001`.

### 4.2.2 Set up parameter file

Set up parameter file```config.yaml```.

#### 4.2.2.1 Common part

```yaml
  msg_source: 2                       # 1: The message originates from an online LiDAR
                                      # 2: The message originates from PCAP
                                      # 3: The message originates from ROS
  send_point_cloud_ros: true          # true: Send point clouds to ROS for visualization
  send_imu_packet_ros: true           # true: Send IMU to ROS for visualization
  send_device_ctrl_state_ros: true    # true: Send interactive data of device status settings to ROS for viewing
  send_packet_ros: false              # true: Send lidar raw packet topic to ROS                 
```

Set```msg_source```                   = 1, Messages originate from the online LiDAR.

Set ```send_point_cloud_ros```        = true, publish the official ROS defined point cloud type sensor-msgs/PointCloud2.
Set ```send_imu_packet_ros```         = true, publish the official ROS defined IMU type sensor_msgs/Imu.
Set ```send_device_ctrl_state_ros```  = true, customize topics and publish device status through ROS topics,
                                              please refer to (../intro/03_device_ctrl_intro_CN.md).
Set ```send_packet_ros```             = true, customize topics and publish LIDAR raw data packages through ROS topics.

#### 4.2.2.2 LiDAR-driver part

```yaml
lidar:
  - driver:
      lidar_type: vanjee_720_16                     # LiDAR Model
      connect_type: 1                               # Connection type: 1-udp  2-tcp 3-serial port
      host_msop_port: 3001                          # Host port number for receiving point cloud data
      lidar_msop_port: 3333                         # LiDAR port number
      wait_for_difop: false                         # Wait for the angle calibration parameters to be imported
      use_lidar_clock: true                         # true: Use LiDAR time as the message timestamp
                                                    # false: Use host computer time as the message timestamp
      pcap_path: "<FILE_PATH>/720_16.pcap"          # Absolute path to pcap file
      pcap_repeat: true                             # Whether to loop the pcap file playback
      pcap_rate: 1                                  # Playback speed of pcap file
      config_from_file: true                        # Whether to get parameters from the configuration file
      angle_path_ver: "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/Vanjee_720_16_VA.csv"      # Path to the vertical angle configuration file
      angle_path_hor: "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/Vanjee_720_16_hA.csv"      # Path to the horizontal angle configuration file
      imu_param_path: "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/vanjee_720_imu_param.csv"  # Path to the IMU parameters configuration file
      dense_points: false                           # true-the coordinate of the invalid point is 0, false-the coordinate of the invalid point is NAN
      ts_first_point: false                         # Whether the timestamp of the point cloud is the time of the first point
                                                    # true - time of the first point, false - time of the last point
      use_offset_timestamp: true                    # Use relative timestamp 
                                                    # true - each point in the point cloud uses the time difference relative to the topic
                                                    # false - each point uses UTC time
      publish_mode: 0                               # Echo mode: 0 - publish first return, 1 - publish second return, 2 - publish both returns
```

Set ```lidar_type```           to the type of LiDAR.
Set ```connect_type```         to the LiDAR connection type.
Set ```host_msop_port```       to the port number on the computer for receiving LiDAR data.
Set ```lidar_msop_port```      to the port number on the LiDAR for sending data.
Set ```wait_for_difop```       = false, the driver directly publishes the point cloud.
Set ```use_lidar_clock```      = true, using LiDAR time as message timestamp.
设置 ```pcap_path```            = "<FILE_PATH>/720_16.pcap"(absolute path), read the pcap file from this path.
设置 ```pcap_repeat```          = true, loop read pcap file.
设置 ```pcap_rate```            = 1, play speed adjustment.
Set ```config_from_file```     = false, not using configuration file data to participate in LiDAR data calculation.
Set ```angle_path_ver```       = "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/Vanjee_720_16_VA.csv"(absolute path),
                                  save the vertical angle table obtained by LIDAR online query to this file.
Set ```angle_path_hor```       = "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/Vanjee_720_16_HA.csv"(absolute path), 
                                  save the horizontal angle table obtained by LIDAR online query to this file.
Set ```imu_param_path```       = "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/vanjee_720_imu_param.csv"(absolute path), 
                                  save the IMU calibration parameters queried online by LIDAR to this file.
Set ```dense_points```         = false, annotate abnormal points in the point cloud as NAN.
Set ```ts_first_point```       = false, the timestamp of the header in the point cloud topic is the last point time of the current circle.
Set ```use_offset_timestamp``` = true, the current circle uses relative time to the header timestamp for each point in time.
Set ```publish_mode```         = 0, when the lidar is set to multiple echoes, only the first echo data is publish.
                                  This parameter only has a filtering effect and does not configure the echo mode of the lidar.

#### 4.2.2.3 lidar-ros part

```yaml
    ros:
      ros_frame_id: vanjee_lidar                       # Coordinate System
      ros_send_point_cloud_topic: /vanjee_points721    # Point cloud topic name
```

Set `ros_send_point_cloud_topic` to the topic for sending point clouds, which is `/vanjee_points733`

```yaml
    ros:
      ros_frame_id: vanjee_lidar                                          # Frame id of packet message and point cloud message
      ros_send_point_cloud_topic: /vanjee_points720_16                    # Topic used to send point cloud through ROS
      ros_send_imu_packet_topic: /vanjee_lidar_imu_packets                # Topic used to send imu through ROS 
      ros_send_device_ctrl_state_topic: /vanjee_lidar_device_ctrl_state   # Topic used to send device ctrl state through ROS 
      ros_packet_topic: /vanjee_lidar_packet                              # Topic used to send packet through ROS 
```

```ros_frame_id```                     Name of the coordinate system for sending ROS topics.
```ros_send_point_cloud_topic```       Name of the point cloud for sending ROS topics.
```ros_send_imu_packet_topic```        Name of the IMU for sending ROS topics.
```ros_send_device_ctrl_state_topic``` Name of the lidar state for sending customize ROS topics.
```ros_packet_topic```                 Name of the lidar raw packages for sending customize ROS topics.

### 4.2.3 Run

Run program
