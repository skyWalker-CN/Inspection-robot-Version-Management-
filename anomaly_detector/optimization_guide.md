# Jetson Orin NX 视觉检测系统优化指南

## 一、性能瓶颈分析

### 原始代码的问题（3 FPS → 优化后 15-25 FPS）

| 瓶颈 | 原始方案 | 优化方案 | 预期加速 |
|------|----------|----------|----------|
| InsightFace 用 CPU | `providers=['CPUExecutionProvider']` `ctx_id=-1` | `CUDAExecutionProvider` `ctx_id=0` | 10-50x |
| 两个 YOLO 模型串行 | helmet.pt + smoking.pt 分别推理 | 合并为单个模型 或 TensorRT 引擎 | 2x |
| 同步管线 | 采集→检测→显示全在一个回调 | 三线程异步管线 | 消除阻塞 |
| 无 TensorRT | .pt / .onnx 原始模型 | .engine TensorRT FP16 | 5-10x |
| 火焰模型 1280×1280 | 输入分辨率过大 | 建议 640×640 重训练 | 4x |
| 逐车牌 PIL 转换 | 每个车牌 BGR→PIL→BGR | 每帧批量一次 PIL | N→1 |
| 人脸数据库逐个比对 | for 循环逐个计算余弦相似度 | 矩阵乘法一次性计算 | O(N)→O(1) |

---

## 二、优化代码使用方法

### 2.1 替换原始代码

```bash
# 将优化后的代码复制到你的 ROS2 包目录
cp anomaly_detector_optimized.py /path/to/your_ros2_pkg/anomaly_detector_optimized.py

# 更新 setup.py / setup.cfg 的 entry_points
# entry_points={
#     'console_scripts': [
#         'anomaly_detector_node = anomaly_detector_optimized:main',
#     ],
# },
```

### 2.2 转换模型为 TensorRT 引擎（关键步骤）

```bash
# 在 Jetson 开发板上执行
cd /path/to/models

# 方法1: 使用转换脚本
python convert_to_tensorrt.py --models-dir . --imgsz 640 --half

# 方法2: 使用优化后代码内置功能
python anomaly_detector_optimized.py --convert --models-dir . --imgsz 640

# 方法3: 检查环境
python convert_to_tensorrt.py --check
python anomaly_detector_optimized.py --check
```

转换完成后，`models/` 目录下会生成：
```
models/
├── helmet_best.pt       # 原始 PyTorch 模型
├── helmet_best.engine   # TensorRT 引擎（自动使用）
├── smoking_best.pt
├── smoking_best.engine
└── weights.onnx         # 火焰/烟雾模型
```

优化后的代码会**自动检测 `.engine` 文件**，优先使用 TensorRT 引擎。

### 2.3 ROS2 启动参数

```bash
# 基础启动（自动使用 GPU + TensorRT）
ros2 run your_pkg anomaly_detector_node

# 自定义参数
ros2 run your_pkg anomaly_detector_node \
  --ros-args \
  -p camera_id:=0 \
  -p frame_width:=640 \
  -p frame_height:=480 \
  -p fps:=30 \
  -p device:=cuda \
  -p half:=true \
  -p face_model_pack:=buffalo_s \
  -p face_det_size:=320 \
  -p enable_fire_smoke:=true \
  -p enable_safety:=true \
  -p enable_plate:=true \
  -p enable_face:=true

# 禁用某些检测器以提升帧率
ros2 run your_pkg anomaly_detector_node \
  --ros-args \
  -p enable_plate:=false \
  -p enable_face:=false
```

### 2.4 新增参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device` | `"cuda"` | 推理设备：`cuda` 或 `cpu` |
| `half` | `true` | FP16 半精度推理（Jetson 上推荐开启） |
| `face_model_pack` | `"buffalo_s"` | InsightFace 模型包：`buffalo_l`(大) / `buffalo_m`(中) / `buffalo_s`(小) / `buffalo_sc`(最小) |
| `face_det_size` | `320` | 人脸检测分辨率：320(快) / 640(准) |
| `safety_combined_model_path` | `""` | 合并的安全检测模型路径（空则用两个独立模型） |
| `detection_interval` | `0` | 检测间隔（秒），0 = 尽可能快 |

---

## 三、技术方案替代建议

### 3.1 YOLOv8 替代方案

#### 方案 A: YOLOv8 + TensorRT（推荐，改动最小）

**做法**：保持 YOLOv8 不变，仅将 `.pt` 转为 `.engine`

```python
from ultralytics import YOLO
model = YOLO("helmet_best.pt")
model.export(format="engine", device=0, half=True, imgsz=640)
# 生成 helmet_best.engine
```

**优点**：零代码改动，5-10x 加速
**缺点**：`.engine` 文件绑定 Jetson 硬件，换设备需重新生成
**适用场景**：当前最推荐的方案

#### 方案 B: YOLOv8n（nano）替代大模型

如果你的模型基于 YOLOv8s/m/l，可以换用 YOLOv8n（nano）：

```python
# 在训练时使用更小的预训练权重
model = YOLO("yolov8n.pt")  # 而非 yolov8s.pt
model.train(data="your_data.yaml", epochs=100, imgsz=640)
```

| 模型 | 参数量 | 速度（640, FP16, Jetson） | 精度（COCO mAP） |
|------|--------|---------------------------|------------------|
| YOLOv8n | 3.2M | ~5ms | 37.3 |
| YOLOv8s | 11.2M | ~8ms | 44.9 |
| YOLOv8m | 25.9M | ~15ms | 50.2 |

**建议**：安全帽/吸烟检测精度要求不高，YOLOv8n 足够

#### 方案 C: 合并两个 YOLO 模型为一个

将安全帽和吸烟数据集合并训练一个模型，**推理时间减半**：

```python
# 合并数据集的 data.yaml
# train: path/to/combined/images
# names:
#   0: helmet
#   1: no_helmet
#   2: smoking
#   3: no_smoking (或 cigarette)
model = YOLO("yolov8n.pt")
model.train(data="combined_safety.yaml", epochs=100, imgsz=640)
model.export(format="engine", device=0, half=True)
```

然后在代码中设置：
```bash
ros2 run your_pkg anomaly_detector_node \
  --ros-args -p safety_combined_model_path:=/path/to/safety_combined.engine
```

#### 方案 D: NVIDIA DeepStream（终极方案，大改）

如果需要极致性能（多路摄像头、4K分辨率），考虑 NVIDIA DeepStream：

```python
# DeepStream 使用 GStreamer 管线，GPU 硬解码 + 推理 + 渲染
# 优势：硬件解码、零拷贝、多路并发
# 劣势：学习曲线陡峭，代码重构量大
```

**建议**：如果单路摄像头 + TensorRT 已满足需求，不需要上 DeepStream

---

### 3.2 InsightFace 替代方案

#### 方案 A: InsightFace + CUDA（立即可用，推荐）

**改动**：仅修改两行代码

```python
# 原始（CPU，极慢）
self._face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
self._face_app.prepare(ctx_id=-1, det_size=(640, 640))

# 优化（GPU，快10倍）
self._face_app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
self._face_app.prepare(ctx_id=0, det_size=(320, 320))
```

| 模型包 | 大小 | 速度 | 精度 | 适用场景 |
|--------|------|------|------|----------|
| buffalo_l | ~330MB | 慢 | 最高 | 高精度识别 |
| buffalo_m | ~180MB | 中 | 高 | 平衡选择 |
| **buffalo_s** | ~65MB | **快** | 中 | **Jetson 推荐** |
| buffalo_sc | ~16MB | 最快 | 低 | 快速筛查 |

**建议**：Jetson Orin NX 上用 `buffalo_s` + `det_size=320`，速度和精度平衡最佳

#### 方案 B: InsightFace ONNX → TensorRT

将 InsightFace 的 ONNX 模型转为 TensorRT 引擎：

```bash
# InsightFace 模型通常在 ~/.insightface/models/buffalo_s/
# 包含: det_500m.onnx, w600k_mbf.onnx, 1k3d68.onnx, 2d106det.onnx

# 逐个转换
trtexec --onnx=det_500m.onnx --saveEngine=det_500m.engine --fp16
trtexec --onnx=w600k_mbf.onnx --saveEngine=w600k_mbf.engine --fp16
```

**注意**：需要修改 InsightFace 源码加载 `.engine` 文件，较复杂

#### 方案 C: 使用更轻量的 SCRFD 人脸检测

如果只需要人脸检测（不需要识别），SCRFD 比 InsightFace 快：

```python
# SCRFD 是 InsightFace 的人脸检测子模块
# 可以单独使用，不需要完整的 FaceAnalysis
from insightface.model_zoo import scrfd
detector = scrfd.SCRFD(model_file='scrfd_500m.onnx')
detector.prepare(ctx_id=0, input_size=(320, 320))
```

---

### 3.3 HyperLPR3 替代方案

#### 方案 A: HyperLPR3 + CUDA

HyperLPR3 内部使用 ONNX Runtime，可以尝试启用 CUDA：

```python
# 检查 HyperLPR3 是否支持 provider 参数
# 部分版本支持：
catcher = LicensePlateCatcher(provider=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 如果不支持，可以修改 HyperLPR3 源码
# 找到 LicensePlateCatcher.__init__ 中的 session 创建代码
# 添加 providers=['CUDAExecutionProvider']
```

#### 方案 B: 使用 PaddleOCR / PP-OCR 车牌识别

PaddleOCR 在 Jetson 上有较好优化：

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)
# 先用 YOLO 检测车牌区域，再用 OCR 识别
```

**优点**：GPU 加速，社区活跃
**缺点**：需要额外的车牌检测模型（可复用 YOLO）

#### 方案 C: 将 HyperLPR3 模型转为 TensorRT

```bash
# HyperLPR3 模型通常在 hyperlpr3 包的 assets/ 目录
# 找到 .onnx 文件后用 trtexec 转换
trtexec --onnx=plate_detect.onnx --saveEngine=plate_detect.engine --fp16
trtexec --onnx=plate_recognize.onnx --saveEngine=plate_recognize.engine --fp16
```

---

### 3.4 火焰/烟雾模型优化

#### 方案 A: 降低输入分辨率

当前模型输入 1280×1280，太大。如果可以重新训练：

```python
# 重新导出 ONNX 时使用 640x640
# YOLOv8 训练时设置 imgsz=640
model = YOLO("yolov8n.pt")
model.train(data="fire_smoke.yaml", imgsz=640)
model.export(format="onnx", imgsz=640)
```

1280→640：推理时间约减少 4 倍

#### 方案 B: ONNX → TensorRT

```bash
# 方法1: 通过 onnxruntime TensorRT Execution Provider
# （优化后代码已自动使用，无需额外操作）

# 方法2: 使用 trtexec 生成独立引擎
trtexec --onnx=weights.onnx --saveEngine=weights.engine --fp16 --useCudaGraph
```

#### 方案 C: 轻量级火焰检测

如果不需要复杂检测，可以使用基于颜色 + 运动检测的快速方案：

```python
# 火焰颜色 HSV 检测 + 轮廓分析
# 速度：1000+ FPS
# 精度：低（适合预筛，配合模型使用）
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower = np.array([0, 50, 200])   # 火焰颜色下界
upper = np.array([30, 255, 255])  # 火焰颜色上界
mask = cv2.inRange(hsv, lower, upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 大面积亮区 = 可能的火焰
```

**建议**：颜色检测作为快速预筛，只在疑似区域运行深度模型

---

## 四、部署优化清单

### 4.1 Jetson 系统级优化

```bash
# 1. 设置最大功耗模式
sudo nvpmodel -m 0          # MAXN 模式（最大功耗）
sudo jetson_clocks          # 锁定最高频率

# 2. 检查当前状态
sudo nvpmodel -q
jetson_clocks --show

# 3. 增加 swap（如果内存不足）
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 4. 关闭图形界面（释放 GPU 内存）
sudo systemctl set-default multi-user.target
# 重启后用 ssh 连接，需要时再启用：
# sudo systemctl set-default graphical.target
```

### 4.2 OpenCV CUDA 编译（可选，进阶优化）

系统自带的 OpenCV 可能不支持 CUDA。如需 GPU 加速图像处理：

```bash
# 检查 OpenCV 是否支持 CUDA
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# 如果输出 0，需要重新编译

# 编译 OpenCV with CUDA（耗时约 2 小时）
# 参考: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
# 关键 CMake 参数:
# -D WITH_CUDA=ON
# -D CUDA_ARCH_BIN=8.7  (Orin NX 的 CUDA 架构)
# -D WITH_CUDNN=ON
# -D OPENCV_DNN_CUDA=ON
```

### 4.3 环境检查

```bash
# 检查 ONNX Runtime GPU 支持
python -c "
import onnxruntime as ort
print('Version:', ort.__version__)
print('Providers:', ort.get_available_providers())
"

# 检查 PyTorch CUDA
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
"

# 检查 TensorRT
python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"

# 如果 onnxruntime 没有 CUDA 支持，需要安装 GPU 版本:
# pip uninstall onnxruntime
# pip install onnxruntime-gpu
# 注意: Jetson 需要使用 NVIDIA 提供的特殊版本
# 参考: https://elinux.org/Jetson_Zoo#ONNX_Runtime
```

---

## 五、优化效果预估

### 不同配置下的预期帧率（Jetson Orin NX 16GB）

| 配置 | 检测 FPS | 显示 FPS | 说明 |
|------|----------|----------|------|
| 原始代码 (全CPU) | 1-3 | 1-3 | 当前状态 |
| 仅 InsightFace→GPU | 5-8 | 15-30 | 最大瓶颈解决 |
| +YOLO TensorRT | 10-15 | 30 | YOLO 加速 |
| +合并YOLO模型 | 15-20 | 30 | 推理次数减半 |
| +火焰模型640 | 20-25 | 30 | 分辨率降低 |
| **全部优化** | **15-25** | **30** | **实时** |

### 各检测器单独耗时预估（优化后）

| 检测器 | CPU 原始 | CUDA 优化 | TensorRT FP16 |
|--------|----------|-----------|----------------|
| 火焰/烟雾 (1280) | ~80ms | ~25ms | ~15ms |
| 火焰/烟雾 (640) | ~20ms | ~8ms | ~5ms |
| YOLO 安全帽 | ~40ms | ~15ms | ~5ms |
| YOLO 吸烟 | ~40ms | ~15ms | ~5ms |
| InsightFace (buffalo_l, CPU) | ~150ms | — | — |
| InsightFace (buffalo_s, CUDA) | — | ~25ms | — |
| HyperLPR3 | ~30ms | ~15ms | ~10ms |

---

## 六、快速上手步骤

```bash
# Step 1: 复制优化代码到项目
cp anomaly_detector_optimized.py /your/ros2/pkg/
cp convert_to_tensorrt.py /your/ros2/pkg/

# Step 2: 检查环境
python convert_to_tensorrt.py --check

# Step 3: 转换模型（在 Jetson 上执行）
cd /your/ros2/pkg/models
python convert_to_tensorrt.py --models-dir . --imgsz 640

# Step 4: 设置 Jetson 最大性能
sudo nvpmodel -m 0 && sudo jetson_clocks

# Step 5: 运行优化后的节点
ros2 run your_pkg anomaly_detector_node

# Step 6: 观察 FPS 日志
# 节点会每5秒输出：
# [INFO] Display: 28.5 FPS, Detect: 18.2 FPS, Results: 5
# 每100次检测输出各检测器耗时：
# [INFO] Det #100: fire_smoke=15.2ms, safety=8.1ms, plate=12.3ms, face=20.5ms
```
