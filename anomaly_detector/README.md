# Anomaly Detector - 综合异常检测功能包

## 功能概述

本功能包整合了四种检测能力，在单个 ROS2 节点中运行：

| 检测模块 | 技术方案 | 检测目标 |
|----------|----------|----------|
| 火焰烟雾检测 | ONNX Runtime | fire, smoke, fire extinguisher |
| 安全检测 | YOLO (ultralytics) | 安全帽佩戴、吸烟行为 |
| 车牌检测 | HyperLPR3 | 车牌号码识别 |
| 人脸检测与识别 | InsightFace | 人脸检测 + 身份识别 |

## 功能包目录结构

部署前，请确保功能包目录结构如下：

```
anomaly_detector/
├── anomaly_detector/                  # Python 源码
│   ├── __init__.py
│   ├── anomaly_detector_node.py       # 主 ROS2 节点（入口）
│   ├── fire_smoke_detector.py         # 火焰烟雾检测模块
│   ├── safety_detector.py             # 安全帽/吸烟检测模块
│   ├── plate_detector.py              # 车牌检测模块
│   └── face_detector.py               # 人脸检测与识别模块
├── models/                            # 模型文件（需手动放置）
│   ├── weights.onnx                   # 火焰烟雾 ONNX 模型
│   ├── helmet_best.pt                 # 安全帽检测 YOLO 模型
│   └── smoking_best.pt                # 吸烟检测 YOLO 模型
├── face_database/                     # 人脸数据库（需手动放置）
│   └── {人员姓名}/
│       ├── feature.npy                # 人脸特征向量
│       └── face.jpg                   # 注册时保存的人脸图片
├── resource/
│   └── anomaly_detector               # ament index 标记文件
├── setup.py
├── setup.cfg
├── package.xml
├── requirements.txt
└── README.md
```

## 部署步骤

### 步骤 1：创建 Conda 虚拟环境

```bash
conda create -n anomaly_detect python=3.10 -y
conda activate anomaly_detect
```

### 步骤 2：安装基础依赖

```bash
pip install -r requirements.txt
```

### 步骤 3：安装 ROS2 Python 依赖

确保 ROS2 Humble 已正确安装并 source：

```bash
source /opt/ros/humble/setup.bash
pip install rclpy std_msgs
```

### 步骤 4：安装 InsightFace（人脸检测）

```bash
pip install insightface
```

### 步骤 5：安装 HyperLPR3（车牌检测）

```bash
pip install hyperlpr3
```

### 步骤 6：放置模型文件

将以下模型文件复制到 `models/` 目录：

| 文件 | 来源 | 说明 |
|------|------|------|
| `weights.onnx` | `fire_smoke_detect/models/weights.onnx` | 火焰烟雾 ONNX 模型 |
| `helmet_best.pt` | `safety_detect/models/helmet_best.pt` | 安全帽检测模型 |
| `smoking_best.pt` | `safety_detect/models/smoking_best.pt` | 吸烟检测模型 |

```bash
# 示例：从已有功能包复制模型
cp /home/jetson/ros2_ws/src/fire_smoke_detect/models/weights.onnx \
   /home/jetson/ros2_ws/src/anomaly_detector/models/

cp /home/jetson/ros2_ws/src/safety_detect/models/helmet_best.pt \
   /home/jetson/ros2_ws/src/anomaly_detector/models/

cp /home/jetson/ros2_ws/src/safety_detect/models/smoking_best.pt \
   /home/jetson/ros2_ws/src/anomaly_detector/models/
```

### 步骤 7：放置人脸数据库

将已注册的人脸数据复制到 `face_database/` 目录：

```bash
# 示例：复制已注册的人脸
cp -r /home/jetson/ros2_ws/src/face_detect/face_database/* \
      /home/jetson/ros2_ws/src/anomaly_detector/face_database/
```

人脸数据库结构：
```
face_database/
├── zhangsan/
│   ├── feature.npy    # 512维人脸特征向量
│   └── face.jpg       # 注册时的人脸截图
├── lisi/
│   ├── feature.npy
│   └── face.jpg
└── ...
```

### 步骤 8：安装中文字体（车牌标注用）

```bash
sudo apt install fonts-noto-cjk
```

如果系统没有该字体包，也可使用：
```bash
sudo apt install fonts-arphic-uming
```

### 步骤 9：编译功能包

```bash
cd /home/jetson/ros2_ws
colcon build --packages-select anomaly_detector
source install/setup.bash
```

### 步骤 10：运行

```bash
# 激活 conda 环境
conda activate anomaly_detect

# Source ROS2
source /opt/ros/humble/setup.bash
source /home/jetson/ros2_ws/install/setup.bash

# 运行节点
ros2 run anomaly_detector anomaly_detector
```

## 参数配置

可通过 ROS2 参数机制调整检测器行为：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `camera_id` | 0 | 摄像头设备 ID |
| `frame_width` | 640 | 采集分辨率 - 宽 |
| `frame_height` | 480 | 采集分辨率 - 高 |
| `fps` | 15 | 检测帧率 |
| `show_window` | True | 是否显示预览窗口 |
| `enable_fire_smoke` | True | 启用火焰烟雾检测 |
| `enable_safety` | True | 启用安全帽/吸烟检测 |
| `enable_plate` | True | 启用车牌检测 |
| `enable_face` | True | 启用人脸检测与识别 |
| `fire_smoke_model_dir` | models/ | 火焰烟雾模型目录 |
| `helmet_model_path` | models/helmet_best.pt | 安全帽模型路径 |
| `smoking_model_path` | models/smoking_best.pt | 吸烟模型路径 |
| `face_database_dir` | face_database/ | 人脸数据库目录 |

启动时修改参数示例：
```bash
ros2 run anomaly_detector anomaly_detector --ros-args -p enable_plate:=false
```

## 发布的话题

| 话题 | 类型 | 内容 |
|------|------|------|
| `/anomaly/fire_smoke` | `std_msgs/String` | 火焰/烟雾检测结果 (JSON) |
| `/anomaly/safety` | `std_msgs/String` | 安全帽/吸烟检测结果 (JSON) |
| `/anomaly/plate` | `std_msgs/String` | 车牌检测结果 (JSON) |
| `/anomaly/face` | `std_msgs/String` | 人脸检测结果 (JSON) |

JSON 格式示例：
```json
// /anomaly/fire_smoke
[{"class": "fire", "conf": 0.85, "box": [100, 200, 300, 400]}]

// /anomaly/safety
[{"class": "helmet", "conf": 0.95, "box": [50, 60, 200, 300]}]

// /anomaly/plate
[{"plate": "京A12345", "conf": 0.92, "box": [10, 20, 150, 80]}]

// /anomaly/face
[{"name": "zhangsan", "similarity": 0.78, "conf": 0.95, "box": [40, 30, 200, 250]}]
```

## 预览窗口说明

- 按 `q` 键退出程序
- 左上角图例：FS=火焰烟雾, SF=安全检测, LP=车牌检测, FD=人脸检测
- 各检测器使用不同颜色标注，便于区分

## 依赖关系总览

```
anomaly_detector
├── numpy
├── opencv-python
├── Pillow
├── onnxruntime          (火焰烟雾)
├── ultralytics          (安全帽/吸烟)
├── hyperlpr3            (车牌)
├── insightface          (人脸)
└── ROS2 (rclpy, std_msgs)
```

## 常见问题

**Q: 摄像头打不开？**
A: 检查 `camera_id` 参数，或确认摄像头设备路径 `/dev/video0` 是否存在。

**Q: 某个检测器报错或不工作？**
A: 使用 `--ros-args -p enable_xxx:=false` 单独关闭该检测器，排查问题。

**Q: 人脸识别不准确？**
A: 提高 `face_detector.py` 中的 `threshold` 参数（默认 0.50），或重新注册更清晰的人脸。