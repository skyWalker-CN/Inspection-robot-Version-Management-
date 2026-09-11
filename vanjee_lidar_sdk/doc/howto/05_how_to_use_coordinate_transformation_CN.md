# 5 如何使用坐标变换功能

## 5.1 简介

 ```vanjee_lidar_sdk ```支持对点云进行坐标变换，本文档展示如何作这种变换。 

在阅读本文档之前，请确保已阅读雷达用户手册。

## 5.2 依赖库

 ```vanjee_lidar_sdk ```的坐标变换基于 ```libeigen ```库，所以要先安装它。

```bash
sudo apt-get install libeigen3-dev
```

## 5.3 编译

要启用坐标变换，编译 ```vanjee_lidar_sdk ```时，需要将```ENABLE_TRANSFORM```选项设置为```ON```.

- 直接编译

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

## 5.4 设置雷达参数

## 5.4.1 配置点云平移旋转参数

在`config.yaml`中，设置`lidar-lidar`部分的参数`x`、 `y`、 `z`、 `roll`、 `pitch` 、`yaw`, 单独点云数据进行平移旋转。

```yaml
common:
  msg_source: 1 
  send_point_cloud_ros: true
lidar:
  - driver:
      lidar_type: vanjee_720_16                     # LiDAR类型
      connect_type: 1                               # 连接方式 1-udp  2-tcp 3-serial port
      host_msop_port: 3001                          # 接收点云数据的主机端口号
      lidar_msop_port: 3333                         # 雷达端口号
      start_angle: 0                                # 点云起始角度
      end_angle: 360                                # 点云结束角度
      wait_for_difop: true                          # 是否等角度标定表参数导入
      min_distance: 0.5                             # 点云最小距离
      max_distance: 120                             # 点云最大距离
      hide_points_range:                            # 屏蔽点范围
      use_lidar_clock: true                         # true: 使用雷达时间作为消息时间戳
                                                    # false: 使用电脑主机时间作为消息时间戳
      pcap_path:                                    # pcap文件绝对路径
      pcap_repeat: true                             # 是否循环播放pcap
      pcap_rate: 10                                 # pcap播放速度
      config_from_file: false                       # 从配置文件内获取参数
      angle_path_ver:                               # 垂直角度配置文件地址
      angle_path_hor:                               # 水平角度配置文件地址
      imu_param_path:                               # imu参数配置文件地址
      dense_points: false                           # true-无效点坐标为0, false-无效点坐标为NAN
      ts_first_point: false                         # 点云的时间戳是否第一个点的时间 true-第一个点的时间，false-最后一个点的时间
      use_offset_timestamp: true                    # 使用相对时间戳 true-点云中每个点使用相对于话题的时间差，false-每个点使用utc时间
      publish_mode: 0                               # 回波模式 0-发布第一重，1-发布第二重；2-发布两重；
      group_address: 0.0.0.0                        # 组播地址
      host_address: 192.168.2.88                    # 接收点云数据的主机IP地址
      lidar_address: 192.168.2.86                   # 雷达IP地址
      x: 1                                          # 点云 x方向偏移量 m
      y: 2                                          # 点云 y方向偏移量 m
      z: 3                                          # 点云 z方向偏移量 m
      roll: 0                                       # 点云 横滚角偏移量 °
      pitch: 0                                      # 点云 俯仰角偏移量 °
      yaw: 90                                       # 点云 航向角偏移量 °
      x_imu: 0                                      # imu x方向偏移量 m
      y_imu: 0                                      # imu y方向偏移量 m
      z_imu: 0                                      # imu z方向偏移量 m
      roll_imu: 0                                   # imu 横滚角偏移量 °
      pitch_imu: 0                                  # imu 俯仰角偏移量 °
      yaw_imu: 0                                    # imu 航向角偏移量 °
    use_vlan: false                                 # PCAP文件中的Packet是否包含VLAN层
    ros:
      ros_frame_id: vanjee_lidar                            # Frame id of packet message and point cloud message
      ros_send_point_cloud_topic: /vanjee_points720_16      # Topic used to send point cloud through ROS
```

## 5.4.2 配置IMU平移旋转参数

在`config.yaml`中，设置`lidar-lidar`部分的参数`x_imu`、 `y_imu`、 `z_imu`、 `roll_imu`、 `pitch_imu` 、`yaw_imu`, 单独IMU数据进行旋转。

**备注: 目前IMU`x_imu`、 `y_imu`、 `z_imu` 未生效**

```yaml
common:
  msg_source: 1 
  send_point_cloud_ros: true
lidar:
  - driver:
      lidar_type: vanjee_720_16                     # LiDAR类型
      connect_type: 1                               # 连接方式 1-udp  2-tcp 3-serial port
      host_msop_port: 3001                          # 接收点云数据的主机端口号
      lidar_msop_port: 3333                         # 雷达端口号
      start_angle: 0                                # 点云起始角度
      end_angle: 360                                # 点云结束角度
      wait_for_difop: true                          # 是否等角度标定表参数导入
      min_distance: 0.5                             # 点云最小距离
      max_distance: 120                             # 点云最大距离
      hide_points_range:                            # 屏蔽点范围
      use_lidar_clock: true                         # true: 使用雷达时间作为消息时间戳
                                                    # false: 使用电脑主机时间作为消息时间戳
      pcap_path:                                    # pcap文件绝对路径
      pcap_repeat: true                             # 是否循环播放pcap
      pcap_rate: 10                                 # pcap播放速度
      config_from_file: false                       # 从配置文件内获取参数
      angle_path_ver:                               # 垂直角度配置文件地址
      angle_path_hor:                               # 水平角度配置文件地址
      imu_param_path:                               # imu参数配置文件地址
      dense_points: false                           # true-无效点坐标为0, false-无效点坐标为NAN
      ts_first_point: false                         # 点云的时间戳是否第一个点的时间 true-第一个点的时间，false-最后一个点的时间
      use_offset_timestamp: true                    # 使用相对时间戳 true-点云中每个点使用相对于话题的时间差，false-每个点使用utc时间
      publish_mode: 0                               # 回波模式 0-发布第一重，1-发布第二重；2-发布两重；
      group_address: 0.0.0.0                        # 组播地址
      host_address: 192.168.2.88                    # 接收点云数据的主机IP地址
      lidar_address: 192.168.2.86                   # 雷达IP地址
      x: 0                                          # 点云 x方向偏移量 m
      y: 0                                          # 点云 y方向偏移量 m
      z: 0                                          # 点云 z方向偏移量 m
      roll: 0                                       # 点云 横滚角偏移量 °
      pitch: 0                                      # 点云 俯仰角偏移量 °
      yaw: 0                                        # 点云 航向角偏移量 °
      x_imu: 0                                      # imu x方向偏移量 m
      y_imu: 0                                      # imu y方向偏移量 m
      z_imu: 0                                      # imu z方向偏移量 m
      roll_imu: 0                                   # imu 横滚角偏移量 °
      pitch_imu: 0                                  # imu 俯仰角偏移量 °
      yaw_imu: 90                                   # imu 航向角偏移量 °
    use_vlan: false                                 # PCAP文件中的Packet是否包含VLAN层
    ros:
      ros_frame_id: vanjee_lidar                            # Frame id of packet message and point cloud message
      ros_send_point_cloud_topic: /vanjee_points720_16      # Topic used to send point cloud through ROS
```

## 5.5 运行

运行程序。
