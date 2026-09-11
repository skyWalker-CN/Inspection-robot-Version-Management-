# 异常检测列表消息定义
# 标准ROS2消息头
std_msgs/Header header

# 异常检测总数
int32 total_count

# 帧时间戳
float64 frame_timestamp

# 异常检测列表
AnomalyDetection[] anomalies 