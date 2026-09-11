# ROS 2 巡检机器人系统

本项目是面向移动巡检机器人的 ROS 2 综合工作空间，集成底盘、导航、定位、激光雷达、深度相机、热成像、云台摄像机、异常检测、语音交互与自动充电等能力。

## 功能概览

- 移动底盘：AgileX Ranger 驱动、UGV SDK 与串口通信。
- 自主导航：Nav2、地图定位、路径规划、航点巡检和路径偏差记录。
- 环境感知：雷神/思岚/万集激光雷达、Orbbec 深度相机和毫米波雷达。
- 视觉检测：烟火、安全行为、车牌、人脸和综合异常检测。
- 热成像与网络相机：海康相机接入、告警及云台控制。
- 人机交互：离线语音识别、语音播放及 Ollama ROS 2 对话服务。
- 任务控制：巡检流程、机器人状态、跟点、转向和自动回充。

## 目录结构

```text
巡检机器人相关代码/
├── requirements.txt
├── README.md
└── ros2_ws/
    └── src/
        ├── example_python/       # 业务控制与巡检任务节点
        ├── myRobot_bringup/      # 整机启动文件
        ├── anomaly_detector/     # 综合异常检测
        ├── navigation2/          # Nav2 源码
        ├── ranger_ros2-humble/   # Ranger 底盘驱动
        ├── OrbbecSDK_ROS2/       # Orbbec 相机驱动
        ├── LSLIDAR_X_ROS2-*/     # 雷神雷达驱动
        ├── rplidar_ros-dev-ros2/ # 思岚雷达驱动
        ├── vanjee_lidar_sdk/     # 万集雷达驱动
        ├── wheeltec_radar/       # 毫米波雷达
        ├── thermal_camera_pkg/   # 热成像相机
        ├── HKWS/                 # 海康网络相机
        ├── ollama_ros_chat/      # 本地大模型对话
        └── rtk/                  # RTK 数据节点
```

## 推荐环境

- Ubuntu 22.04、ROS 2 Humble、Python 3.10
- NVIDIA Jetson Orin NX / JetPack 6.x（视觉推理场景）
- Ranger 移动底盘及对应 CAN 适配器
- 项目所需的相机、雷达、串口/USB 设备

## 安装依赖

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions python3-rosdep \
  ros-humble-desktop ros-humble-cv-bridge ros-humble-tf-transformations

cd "巡检机器人相关代码"
python3 -m pip install -r requirements.txt
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

仓库已经包含 Nav2 和若干厂商驱动源码。Orbbec、海康、雷达、CAN 和热成像设备可能还需要厂商 SDK、udev 规则或内核权限。Jetson 上的 CUDA、TensorRT、PyTorch 和 OpenCV 应优先使用 JetPack/NVIDIA 提供的版本。

## 编译

```bash
cd "巡检机器人相关代码/ros2_ws"
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

工作空间较大，可按需选择包：

```bash
colcon build --symlink-install --packages-select \
  myRobot_bringup example_python ranger_base ranger_bringup
```

## 启动与运行

整机启动入口：

```bash
ros2 launch myRobot_bringup myRobot.launch.py
```

常用独立节点示例：

```bash
# 键盘控制
ros2 run example_python tele_key

# 导航到目标点
ros2 run example_python my_navToPose

# 航点巡检
ros2 run example_python followPointsV8

# 自动充电
ros2 run example_python chargeV2

# 综合异常检测
ros2 run anomaly_detector anomaly_detector

# Ollama 对话服务
ros2 launch ollama_ros_chat ollama_ros_chat.launch.py
```

不同硬件组合对应不同雷达或相机启动文件。运行前请检查 `myRobot_bringup/launch/`、各驱动的 `config/`/`params/` 目录以及业务脚本中的串口号、IP、模型路径、地图和航点配置。

## 调试建议

```bash
ros2 node list
ros2 topic list
ros2 topic hz /scan
ros2 topic echo /odom
ros2 doctor --report
```

若设备无法访问，请检查 USB/CAN/串口权限、udev 规则和网络地址；若视觉节点失败，请优先检查模型路径、CUDA/PyTorch 架构和相机话题编码。

## 上传 GitHub 前

本项目包含模型、地图、第三方驱动和完整上游源码，体积可能较大。建议使用 Git LFS 管理模型及媒体文件，并检查各第三方目录的许可证和再分发条款。`build/`、`install/`、`log/`、缓存与运行输出不应提交。

