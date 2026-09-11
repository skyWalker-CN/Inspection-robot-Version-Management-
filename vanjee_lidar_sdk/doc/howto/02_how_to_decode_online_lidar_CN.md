# 2 如何连接在线雷达

## 2.1 简介

本文档描述如何连接在线雷达，并发送点云数据到ROS。

在阅读本文档之前， 请确保已经阅读过雷达用户手册和[参数简介](../intro/02_parameter_intro_CN.md) 。

## 2.2 步骤

### 2.2.1 获取数据端口号

根据雷达用户手册连接雷达, 并设置好您的电脑的IP地址。

请参考雷达用户手册，或使用第三方工具（如WireShark等）得到雷达数据目标端口号。

### 2.2.2 设置参数文件

设置参数文件```config.yaml```。

#### 2.2.2.1 common部分

```yaml
msg_source: 1                                     # 1: 消息来源于在线雷达
                                                  # 2: 消息来源于 pcap
                                                  # 3: 消息来源于 packet (rostopic)
  send_point_cloud_ros: true                      # true: 将点云发送到ROS以便查看
  send_imu_packet_ros: true                       # true: 将IMU发送到ROS以便查看
  send_device_ctrl_state_ros: true                # true: 将设置设备状态的交互数据发送到ROS以便查看
  send_packet_ros: true                           # true: 将雷达协议数据包发送到ROS以便查看                    
```

设置```msg_source```                   为1, 消息来源于在线雷达。

设置 ```send_point_cloud_ros```        为true, 点云消息的类型为ROS官方定义的点云类型sensor_msgs/PointCloud2发布。
设置 ```send_imu_packet_ros```         为true, IMU消息的类型为ROS官方定义的IMU类型sensor_msgs/Imu发布。
设置 ```send_device_ctrl_state_ros```  为true, 自定义话题, 将设备状态通过ROS话题发布。详情见(../intro/03_device_ctrl_intro_CN.md) 
设置 ```send_packet_ros```             为true, 自定义话题, 将LIDAR原始数据包通过ROS话题发布。

#### 6.2.2.2 lidar-driver部分

```yaml
lidar:
  - driver:
      lidar_type: vanjee_720_16                     # LiDAR类型
      connect_type: 1                               # 连接方式 1-udp  2-tcp 3-serial port
      host_msop_port: 3001                          # 接收点云数据的主机端口号
      lidar_msop_port: 3333                         # 雷达端口号
      wait_for_difop: true                          # 是否等角度标定表参数导入
      use_lidar_clock: true                         # true: 使用雷达时间作为消息时间戳
                                                    # false: 使用电脑主机时间作为消息时间戳
      config_from_file: false                       # 从配置文件内获取参数
      angle_path_ver: "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/Vanjee_720_16_VA.csv"          # 垂直角度配置文件地址
      angle_path_hor: "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/Vanjee_720_16_hA.csv"          # 水平角度配置文件地址
      imu_param_path: "<PROJECT_PATH>/src/vanjee_lidar_sdk/param/vanjee_720_imu_param.csv"      # imu参数配置文件地址
      dense_points: false                           # true-无效点坐标为0, false-无效点坐标为NAN
      ts_first_point: false                         # 点云的时间戳是否第一个点的时间 true-第一个点的时间，false-最后一个点的时间
      use_offset_timestamp: true                    # 使用相对时间戳 true-点云中每个点使用相对于话题的时间差，false-每个点使用utc时间
      publish_mode: 0                               # 回波模式 0-发布第一重，1-发布第二重；2-发布两重；
      group_address: 0.0.0.0                        # 组播地址
      host_address: 192.168.2.88                    # 接收点云数据的主机IP地址
      lidar_address: 192.168.2.86                   # 雷达IP地址
      port_name: /dev/ttyUSB0                       # 串口名
      baud_rate: 3125000                            # 波特率
```

将 ```lidar_type```             设置为LiDAR类型 。
将 ```connect_type```           设置为LiDAR网络连接类型 。
设置 ```host_msop_port```       电脑端接收LiDAR数据的端口号。
设置 ```lidar_msop_port```      LiDAR端发送数据的端口号。
设置 ```wait_for_difop```       为true, 驱动获取到LiDAR角度表后发布点云。
设置 ```use_lidar_clock```      为true, 使用LiDAR时间作为消息时间戳。
设置 ```config_from_file```     为false, 不使用配置文件数据参与LiDAR各数据计算。
设置 ```angle_path_ver```       为绝对路径"/src/vanjee_lidar_sdk/param/Vanjee_720_16_VA.csv", 将LIDAR在线查询到的垂直角度表保存到该文件。
设置 ```angle_path_hor```       为绝对路径"/src/vanjee_lidar_sdk/param/Vanjee_720_16_HA.csv", 将LIDAR在线查询到的水平角度表保存到该文件。
设置 ```imu_param_path```       为绝对路径"/src/vanjee_lidar_sdk/param/vanjee_720_imu_param.csv", 将LIDAR在线查询到的IMU标定参数保存到该文件。
设置 ```dense_points```         为false, 发布点云中异常点标记为NAN。
设置 ```ts_first_point```       为false, 点云话题中header时间戳为当前圈最后一个点时间。
设置 ```use_offset_timestamp``` 为true, 当前圈每个点时间使用相对于header时间戳相对时间。
设置 ```publish_mode```         为0, 当雷达设置为多重回波时也只发布第一重回波数据, 该参数只有过滤作用, 不对雷达做回波模式配置。
设置 ```group_address```        为0.0.0.0，不使用组播功能。
设置 ```host_address```         电脑端接收LiDAR数据的IP地址。
设置 ```lidar_address```        LiDAR端发送数据的IP地址。
设置 ```port_name```            电脑端接收/发送LiDAR数据的串口号, 该配置项只针对有串口通信雷达生效。
设置 ```baud_rate```            电脑端接收/发送LiDAR数据的串口波特率, 该配置项只针对有串口通信雷达生效。

#### 2.2.2.3 lidar-ros部分

```yaml
    ros:
      ros_frame_id: vanjee_lidar                                          # ROS话题坐标系
      ros_send_point_cloud_topic: /vanjee_points720_16                    # 点云ROS话题名
      ros_send_imu_packet_topic: /vanjee_lidar_imu_packets                # IMU ROS话题名
      ros_send_device_ctrl_state_topic: /vanjee_lidar_device_ctrl_state   # 设备状态自定义ROS话题名
      ros_packet_topic: /vanjee_lidar_packet                              # 原始数据包自定义ROS话题名
```

将 ```ros_frame_id```                     设置发送ROS话题坐标系。
将 ```ros_send_point_cloud_topic```       设置为发送点云的ROS话题。
将 ```ros_send_imu_packet_topic```        设置为发送IMU的ROS话题。
将 ```ros_send_device_ctrl_state_topic``` 设置为发送LIDAR状态的自定义ROS话题。
将 ```ros_packet_topic```                 设置为发送LIDAR 原始数据包的自定义ROS话题。

### 2.2.3 运行

运行程序。
