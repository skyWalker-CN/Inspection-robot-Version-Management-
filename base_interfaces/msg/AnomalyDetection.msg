# 异常检测消息定义
# 标准ROS2消息头
std_msgs/Header header

# 异常类型
string anomaly_type

# 时间戳
float64 timestamp

# 置信度 (0.0-1.0)
float64 confidence

# 位置信息
geometry_msgs/Point position

# 图像路径（如果有截图）
string image_path

# 描述信息
string description 