# 5 How to use coordinate transformation

## 5.1 Introduction

`vanjee_lidar_sdk` supports coordinate transformation of point clouds. This document demonstrates how to perform this transformation.

Before reading this document, please ensure that you have read the LiDAR product manual.

## 5.2 Dependent Libraries

The coordinate transformation in `vanjee_lidar_sdk` is based on the `libeigen` library, so it needs to be installed first.

```bash
sudo apt-get install libeigen3-dev
```

## 5.3 Compile

To enable coordinate transformation, when compiling `vanjee_lidar_sdk`, the `ENABLE_TRANSFORM` option needs to be set to `ON`.

- Compile directly

  ```bash
  cmake -DENABLE_TRANSFORM=ON ..
  ```

- ROS

  ```bash
  catkin_make -DENABLE_TRANSFORM=ON
  ```

- ROS2

  ```bash
  colcon build --cmake-args -DENABLE_TRANSFORM=ON
  ```

## 5.4 Set up LiDAR parameter

## 5.4.1 Configure point cloud translation and rotation parameters

In the `config.yaml` file, set the parameters `x`, `y`, `z`, `roll`, `pitch`, and `yaw` under the `lidar-lidar` section, translate and rotate individual point cloud data.

```yaml
common:
  msg_source: 1 
  send_point_cloud_ros: true
lidar:
  - driver:
      lidar_type: vanjee_720_16                     # LiDAR Model
      connect_type: 1                               # Connection type: 1-udp  2-tcp 3-serial port
      host_msop_port: 3001                          # Host port number for receiving point cloud data
      lidar_msop_port: 3333                         # LiDAR port number
      start_angle: 0                                # Starting angle of the point cloud
      end_angle: 360                                # Ending angle of the point cloud
      wait_for_difop: true                          # Wait for the angle calibration parameters to be imported
      min_distance: 0.5                             # Minimum distance of the point cloud
      max_distance: 120                             # Maximum distance of the point cloud
      hide_points_range:                            # Hide point range
      use_lidar_clock: true                         # true: Use LiDAR time as the message timestamp
                                                    # false: Use host computer time as the message timestamp
      pcap_path:                                    # Absolute path to the PCAP file
      pcap_repeat: true                             # Whether to loop playback of the PCAP file
      pcap_rate: 10                                 # Playback speed of the PCAP file
      config_from_file: false                       # Whether to get parameters from the configuration file
      angle_path_ver:                               # Path to the vertical angle configuration file
      angle_path_hor:                               # Path to the horizontal angle configuration file
      imu_param_path:                               # Path to the IMU parameters configuration file
      dense_points: false                           # true-the coordinate of the invalid point is 0, false-the coordinate of the invalid point is NAN
      ts_first_point: false                         # Whether the timestamp of the point cloud is the time of the first point
                                                    # true - time of the first point, false - time of the last point
      use_offset_timestamp: true                    # Use relative timestamp 
                                                    # true - each point in the point cloud uses the time difference relative to the topic
                                                    # false - each point uses UTC time
      publish_mode: 0                               # Echo mode: 0 - publish first return, 1 - publish second return, 2 - publish both returns
      group_address: 0.0.0.0                        # Multicast address
      host_address: 192.168.2.88                    # Host IP address for receiving point cloud data
      lidar_address: 192.168.2.86                   # LiDAR IP address
      x: 1                                          # pointcloud X-axis offset (m)
      y: 2                                          # pointcloud Y-axis offset (m)
      z: 3                                          # pointcloud Z-axis offset (m)
      roll: 0                                       # pointcloud Roll angle offset (°)
      pitch: 0                                      # pointcloud Pitch angle offset (°)
      yaw: 90                                       # pointcloud Yaw angle offset (°)
      x_imu: 0                                      # imu X-axis offset (m)
      y_imu: 0                                      # imu Y-axis offset (m)
      z_imu: 0                                      # imu Z-axis offset (m)
      roll_imu: 0                                   # imu Roll angle offset (°)
      pitch_imu: 0                                  # imu Pitch angle offset °)
      yaw_imu: 0                                    # imu Yaw angle offset (°)
      use_vlan: false                               # Whether packets in the PCAP file contain VLAN layers

    ros:
      ros_frame_id: vanjee_lidar                            # Frame id of packet message and point cloud message
      ros_send_point_cloud_topic: /vanjee_points720_16      # Topic used to send point cloud through ROS
```

## 5.4.1 Configure IMU translation and rotation parameters

In the `config.yaml` file, set the parameters `x_imu`, `y_imu`, `z_imu`, `roll_imu`, `pitch_imu`, and `yaw_imu` under the `lidar-lidar` section, trotate individual IMU data.

**Note: Currently `x_imu`、 `y_imu` and `z_imu` not in effect**

```yaml
common:
  msg_source: 1 
  send_point_cloud_ros: true
lidar:
  - driver:
      lidar_type: vanjee_720_16                     # LiDAR Model
      connect_type: 1                               # Connection type: 1-udp  2-tcp 3-serial port
      host_msop_port: 3001                          # Host port number for receiving point cloud data
      lidar_msop_port: 3333                         # LiDAR port number
      start_angle: 0                                # Starting angle of the point cloud
      end_angle: 360                                # Ending angle of the point cloud
      wait_for_difop: true                          # Wait for the angle calibration parameters to be imported
      min_distance: 0.5                             # Minimum distance of the point cloud
      max_distance: 120                             # Maximum distance of the point cloud
      hide_points_range:                            # Hide point range
      use_lidar_clock: true                         # true: Use LiDAR time as the message timestamp
                                                    # false: Use host computer time as the message timestamp
      pcap_path:                                    # Absolute path to the PCAP file
      pcap_repeat: true                             # Whether to loop playback of the PCAP file
      pcap_rate: 10                                 # Playback speed of the PCAP file
      config_from_file: false                       # Whether to get parameters from the configuration file
      angle_path_ver:                               # Path to the vertical angle configuration file
      angle_path_hor:                               # Path to the horizontal angle configuration file
      imu_param_path:                               # Path to the IMU parameters configuration file
      dense_points: false                           # true-the coordinate of the invalid point is 0, false-the coordinate of the invalid point is NAN
      ts_first_point: false                         # Whether the timestamp of the point cloud is the time of the first point
                                                    # true - time of the first point, false - time of the last point
      use_offset_timestamp: true                    # Use relative timestamp 
                                                    # true - each point in the point cloud uses the time difference relative to the topic
                                                    # false - each point uses UTC time
      publish_mode: 0                               # Echo mode: 0 - publish first return, 1 - publish second return, 2 - publish both returns
      group_address: 0.0.0.0                        # Multicast address
      host_address: 192.168.2.88                    # Host IP address for receiving point cloud data
      lidar_address: 192.168.2.86                   # LiDAR IP address
      x: 0                                          # pointcloud X-axis offset (m)
      y: 0                                          # pointcloud Y-axis offset (m)
      z: 0                                          # pointcloud Z-axis offset (m)
      roll: 0                                       # pointcloud Roll angle offset (°)
      pitch: 0                                      # pointcloud Pitch angle offset (°)
      yaw: 0                                        # pointcloud Yaw angle offset (°)
      x_imu: 0                                      # imu X-axis offset (m)
      y_imu: 0                                      # imu Y-axis offset (m)
      z_imu: 0                                      # imu Z-axis offset (m)
      roll_imu: 0                                   # imu Roll angle offset (°)
      pitch_imu: 0                                  # imu Pitch angle offset °)
      yaw_imu: 90                                   # imu Yaw angle offset (°)
      use_vlan: false                               # Whether packets in the PCAP file contain VLAN layers

    ros:
      ros_frame_id: vanjee_lidar                            # Frame id of packet message and point cloud message
      ros_send_point_cloud_topic: /vanjee_points720_16      # Topic used to send point cloud through ROS
```

## 10.5 Run

Run program.
