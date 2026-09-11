# 异常检测模型目录

本目录用于存放异常检测所需的模型文件。

## 需要的模型文件

### 1. 火焰烟雾检测模型
- **文件名**: `fire_smoke_best.pt`
- **类型**: YOLOv8 模型文件
- **功能**: 检测火焰和烟雾
- **来源**: 需要训练或下载预训练模型

### 2. 车牌检测模型
- **文件名**: `plate_best.pt`
- **类型**: YOLOv8 模型文件
- **功能**: 检测车牌区域
- **来源**: 需要训练或下载预训练模型

### 3. 人脸检测模型
- **文件名**: `yolov8n-face.pt`
- **类型**: YOLOv8 模型文件
- **功能**: 检测人脸区域
- **来源**: 可以从 ultralytics 官方下载

### 4. dlib 模型文件
- **文件名**: `shape_predictor_68_face_landmarks.dat`
- **类型**: dlib 人脸特征点检测模型
- **功能**: 提取人脸特征点
- **来源**: dlib 官方模型

- **文件名**: `dlib_face_recognition_resnet_model_v1.dat`
- **类型**: dlib 人脸识别模型
- **功能**: 提取人脸特征向量
- **来源**: dlib 官方模型

## 模型获取方式

### 方法1: 下载预训练模型
```bash
# 下载 YOLOv8 人脸检测模型
wget https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt -O models/yolov8n-face.pt

# 下载 dlib 模型文件
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

# 解压 dlib 模型文件
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

# 移动到模型目录
mv shape_predictor_68_face_landmarks.dat models/
mv dlib_face_recognition_resnet_model_v1.dat models/
```

### 方法2: 训练自定义模型
对于火焰烟雾检测和车牌检测，建议使用自定义数据集训练模型：

1. 收集数据集
2. 标注数据
3. 使用 YOLOv8 训练模型
4. 将训练好的模型文件放入此目录

## 模型文件结构
```
models/
├── fire_smoke_best.pt                    # 火焰烟雾检测模型
├── plate_best.pt                         # 车牌检测模型
├── yolov8n-face.pt                       # 人脸检测模型
├── shape_predictor_68_face_landmarks.dat # dlib 人脸特征点模型
├── dlib_face_recognition_resnet_model_v1.dat # dlib 人脸识别模型
└── README.md                             # 本说明文件
```

## 注意事项

1. 确保模型文件具有执行权限
2. 模型文件较大，建议使用 Git LFS 管理
3. 定期更新模型以获得更好的检测效果
4. 在生产环境中，建议使用 GPU 加速模型推理 