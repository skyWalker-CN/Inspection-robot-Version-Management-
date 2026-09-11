# 4 雷达参数接口介绍

**支持雷达型号:**
  WLR-716mini
  WLR-718h
  WLR-719
  WLR-719e
  WLR-720_16
  WLR-720_32
  WLR-722
  WLR-722f
  WLR-722h
  WLR-722z

## 4.1 雷达参数接口话题信息

```
std_msgs/Header header
uint16 cmd_id
uint8 cmd_type
uint32 repeat_interval
string data
```

``` header ```
雷达参数接口话题头信息:
ROS1包含内容有:序号(seq) 时间戳(timestamp) 坐标系(frame id);
ROS2包含内容有:时间戳(timestamp) 坐标系(frame id).

``` cmd_id ```
操作码:
0x0001 - 雷达工作模式;    (支持雷达型号: vanjee_722, vanjee_722f, vanjee_722h, vanjee_722z)
0x0002 - 雷达温度;        (支持雷达型号: vanjee_722f, vanjee_722h, vanjee_722z)
0x0003 - 雷达固件版本号;  (支持雷达型号: vanjee_716mini, vanjee_718h, vanjee_722f, vanjee_722h, vanjee_722z)
0x0004 - 雷达序列号;      (支持雷达型号: vanjee_722f, vanjee_722h, vanjee_722z)
0x0100 - 雷达状态;       (支持雷达型号: vanjee_716mini, vanjee_718h, vanjee_719, vanjee_719e, vanjee_720_16, vanjee_720_32, vanjee_722, vanjee_722f, vanjee_722h, vanjee_722z)

``` cmd_type ```
"操作码类型:
0 - 查询  (支持操作码: 0x0001, 0x0002, 0x0003, 0x0004)
1 - 设置  (支持操作码: 0x0001)
备注: 雷达状态驱动会主动上报，不需要查询.

``` repeat_interval ```
指令循环发送间隔, 单位：毫秒(ms)，参数需配置100ms的整数倍:
0 - 单次查询;
1000 - 循环查询, 1000ms 查询一次;

``` data ```
详细参数:
{
  "temperature": 20.0
}
备注: 数据封装为 JSON 格式.

``` 雷达参数接口话题头信息
  uint32 seq = 0;
  float64 timestamp = 0.0;
  string frame_id = "";
```

## 4.2 雷达参数接口信息

数据结构:
class LidarParameterInterface {
 public:
  uint32 seq;
  float64 timestamp;
  uint16 cmd_id;
  uint8 cmd_type;
  uint32 repeat_interval;
  std::string data;
};

``` 
接口函数:
void lidarParameterApi(const LidarParameterInterface& lidar_param);
``` 
设备类:
LidarDriver<PointCloudMsg> driver;

## 4.2.1 接收雷达参数指令的响应数据
回调函数注册:
driver.regLidarParameterInterfaceCallback(allocateLidarParameterInterfaceMemoryCallback, lidarParameterInterfaceCallback);
sdk通过 allocateLidarParameterInterfaceMemoryCallback 提供的内存空间存储雷达参数指令的响应数据
通过 lidarParameterInterfaceCallback 接收雷达参数指令的响应数据

## 4.2.2 发送雷达参数设置指令
雷达参数指令示例:
LidarParameterInterface lidar_param;
lidar_param.cmd_id = 2;
lidar_param.cmd_type = 0;
lidar_param.repeat_interval = 0;
lidar_param.data = "";


接口调用:
driver.lidarParameterApi(lidar_param);

**具体实现方法可参考demo用例**

## 4.3 各型号雷达参数指令说明

### 4.3.1 WLR-716mini

cmd_id: 3 -获取雷达固件版本号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 3 -回复雷达固件版本号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.2 WLR-718h

cmd_id: 3 -获取雷达固件版本号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 3 -回复雷达固件版本号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.3 WLR-719

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.4 WLR-719E

cmd_id: 0x0100-0x010f -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.5 WLR-720_16

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.6 WLR-720_32

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.7 WLR-722
cmd_id: 1 -获取雷达运行模式（下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 1 -回复雷达运行模式(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式 2-只开启25-28通道
}"

cmd_id: 1 -设置雷达运行模式（下行)：
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式 2-只开启25-28通道
}"

cmd_id: 1 -回复雷达运行模式(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式 2-只开启25-28通道
}"

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-故障
}"

### 4.3.8 WLR-722f

cmd_id: 1 -获取雷达运行模式（下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 1 -回复雷达运行模式(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式
}"

cmd_id: 1 -设置雷达运行模式（下行)：
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式
}"

cmd_id: 1 -回复雷达运行模式(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式
}"

cmd_id: 2 -获取雷达温度（下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 2 -回复雷达温度(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "temperature": 20.5
}"

cmd_id: 3 -获取雷达固件版本号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 3 -回复雷达固件版本号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "firmware_version": "v1.0"
}"

cmd_id: 4 -获取雷达SN号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 4 -回复雷达SN号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "sn": "123456"
}"

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码 (每个bit对应一个故障，如果需要确认故障类型可参考雷达产品手册)
  "fault_state": 1-警告 2-故障
}"

### 4.3.9 WLR-722h

cmd_id: 1 -获取雷达运行模式（下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 1 -回复雷达运行模式(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式
}"

cmd_id: 1 -设置雷达运行模式（下行)：
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式
}"

cmd_id: 1 -回复雷达运行模式(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-工作模式 1-待机模式
}"

cmd_id: 2 -获取雷达温度（下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 2 -回复雷达温度(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "temperature": 20.5
}"

cmd_id: 3 -获取雷达固件版本号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 3 -回复雷达固件版本号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "firmware_version": "v1.0"
}"

cmd_id: 4 -获取雷达SN号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 4 -回复雷达SN号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "sn": "123456"
}"

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码 (每个bit对应一个故障，如果需要确认故障类型可参考雷达产品手册)
  "fault_state": 1-警告 2-故障
}"

### 4.3.10 WLR-722z

cmd_id: 2 -获取雷达温度（下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 2 -回复雷达温度(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "temperature": 20.5
}"

cmd_id: 3 -获取雷达固件版本号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 3 -回复雷达固件版本号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "firmware_version": "v1.0"
}"

cmd_id: 4 -获取雷达SN号(下行)：
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 4 -回复雷达SN号(上行)：
cmd_type: 0
repeat_interval: 0
data: "{
  "sn": "123456"
}"

cmd_id: 0x0100 -雷达运行状态上报(上行)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-故障码
  "fault_state": 0-正常 1-警告 2-关机
}"

cmd_id: 5 -获取加速度量程(下行):
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 5 -回复加速度量程(上行):
cmd_type: 0
repeat_interval: 0
data: "{
  "acceleration_range": 0-±2G 1-±4G
}"

cmd_id: 1 -设置加速度量程(下行):
cmd_type: 1
repeat_interval: 0
data: "{
  "acceleration_range": 0-±2G 1-±4G
}"

cmd_id: 1 -回复加速度量程(上行):
cmd_type: 0
repeat_interval: 0
data: "{
  "acceleration_range": 0-±2G 1-±4G
}"
