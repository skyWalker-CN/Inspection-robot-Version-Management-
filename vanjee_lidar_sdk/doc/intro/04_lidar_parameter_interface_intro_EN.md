# 4 Introduction to Lidar Parameters Interface

**Support lidar type:**
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

## 4.1 Lidar parameter interface topic information

```
std_msgs/Header header
uint16 cmd_id
uint8 cmd_type
uint32 repeat_interval
string data
```

``` header ```
lidar parameter interface topic header information:
ROS1 includes: sequence number(seq), timestamp (timestamp), coordinate system (frame id);
ROS2 includes: timestamp(timestamp),coordinate system (frame id)

``` cmd_id ```
Command identification:
0x0001 - work mode;         (Support lidar type: vanjee_722, vanjee_722f, vanjee_722h, vanjee_722z)
0x0002 - temperature;       (Support lidar type: vanjee_722f, vanjee_722h, vanjee_722z)
0x0003 - firmware version;  (Support lidar type: vanjee_716mini, vanjee_718h, vanjee_722f, vanjee_722h, vanjee_722z)
0x0004 - sn;                (Support lidar type: vanjee_722f, vanjee_722h, vanjee_722z)
0x0100 - lidar state;       (Support lidar type: vanjee_716mini, vanjee_718h, vanjee_719, vanjee_719e, vanjee_720_16, vanjee_720_32, vanjee_722, vanjee_722f, vanjee_722h, vanjee_722z)


``` cmd_type ```
Command type:
the parameters of "set lidar work mode" include:
0 - query (Support cmd id: 0x0001, 0x0002, 0x0003, 0x0004)
1 - set   (Support cmd id: 0x0001)
Note: The lidar state driver will proactively report and does not require querying.

``` repeat_interval ```
Instruction loop sending interval, unit: milliseconds (ms). The parameter must be configured as an integer multiple of 100 ms:
0 -  single query (one-time only);
1000 - loop query (every 1000ms);

``` data ```
Detailed parameters:
{
  "temperature": 20.0
}
Note: The data is encapsulated in JSON format.

``` lidar parameter interface topic header information
  uint32 seq = 0;
  float64 timestamp = 0.0;
  string frame_id = "";
```

## 4.2 Lidar parameter interface information

data structure:
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
application program interface:
void lidarParameterApi(const LidarParameterInterface& lidar_param);
``` 
device class:
LidarDriver<PointCloudMsg> driver;

## 4.2.1 Receive the responded data of lidar parameter command
Register callback function:
driver.regLidarParameterInterfaceCallback(allocateLidarParameterInterfaceMemoryCallback, lidarParameterInterfaceCallback);

Sdk store the responded data of lidar parameter command into the memory provided by the allocateLidarParameterInterfaceMemoryCallback function
Receive the responded data of lidar parameter command by lidarParameterInterfaceCallback function

## 4.2.2 Send lidar parameter command
example of lidar parameter command:
LidarParameterInterface lidar_param;
lidar_param.cmd_id = 2;
lidar_param.cmd_type = 0;
lidar_param.repeat_interval = 0;
lidar_param.data = "";

Call interface:
driver.lidarParameterApi(lidar_param);

**refer the demo for the specific implementation method**

## 4.3 Lidar parameter commands for all lidars

### 4.3.1 WLR-716mini

cmd_id: 0x0003 -Get lidar firmware version (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0003 -The feedback of getting lidar firmware version (to user)
cmd_type: 0
repeat_interval: 0
data:"{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.2 WLR-718h

cmd_id: 0x0003 -Get lidar firmware version (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0003 -The feedback of getting lidar firmware version (to user)
cmd_type: 0
repeat_interval: 0
data:"{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.3 WLR-719

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.4 WLR-719E

cmd_id: 0x0100-0x010f -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.5 WLR-720_16

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.6 WLR-720_32

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.7 WLR-722

cmd_id: 1 -Get lidar work mode (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 1 -The feedback of getting lidar work mode command (to user)
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby; 2-only open channels 25-28
}"

cmd_id: 1 -Set lidar work mode (to lidar)
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-working mode; 1-standby mode; 2-only open channels 25-28
}"

cmd_id: 1 -The feedback of getting lidar work mode command (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby; 2-only open channels 25-28
}"

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Fault
}"

### 4.3.8 WLR-722f

cmd_id: 1 -Get lidar work mode (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 1 -The feedback of getting lidar work mode command (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby;
}"

cmd_id: 1 -Set lidar work mode (to lidar)
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby;
}"

cmd_id: 1 -The feedback of getting lidar work mode command (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby;
}"

cmd_id: 0x0002 -Get lidar temperature (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0002 -The feedback of getting lidar temperature (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "temperature": 20.5
}"

cmd_id: 0x0003 -Get lidar firmware version (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0003 -The feedback of getting lidar firmware version (to user)
cmd_type: 0
repeat_interval: 0
data:"{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0004 -Get lidar sn (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0004 -The feedback of getting lidar sn (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "sn": "123456"
}"

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-fault code (Each bit corresponds to a fault. If you need to confirm the fault type, please refer to the lidar product manual)
  "fault_state": 1-Warning 2-Fault
}"

### 4.3.9 WLR-722h

cmd_id: 1 -Get lidar work mode (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 1 -The feedback of getting lidar work mode command (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby;
}"

cmd_id: 1 -Set lidar work mode (to lidar)
cmd_type: 1
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby;
}"

cmd_id: 1 -The feedback of getting lidar work mode command (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "work_mode": 0-working; 1-standby;
}"

cmd_id: 0x0002 -Get lidar temperature (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0002 -The feedback of getting lidar temperature (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "temperature": 20.5
}"

cmd_id: 0x0003 -Get lidar firmware version (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0003 -The feedback of getting lidar firmware version (to user)
cmd_type: 0
repeat_interval: 0
data:"{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0004 -Get lidar sn (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0004 -The feedback of getting lidar sn (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "sn": "123456"
}"

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001-fault code (Each bit corresponds to a fault. If you need to confirm the fault type, please refer to the lidar product manual)
  "fault_state": 1-Warning 2-Fault
}"

### 4.3.10 WLR-722z

cmd_id: 0x0002 -Get lidar temperature (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0002 -The feedback of getting lidar temperature (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "temperature": 20.5
}"

cmd_id: 0x0003 -Get lidar firmware version (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0003 -The feedback of getting lidar firmware version (to user)
cmd_type: 0
repeat_interval: 0
data:"{
  "firmware_version": "v1.0"
}"

cmd_id: 0x0004 -Get lidar sn (to lidar)
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 0x0004 -The feedback of getting lidar sn (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "sn": "123456"
}"

cmd_id: 0x0100 -Lidar state reporting (to user)
cmd_type: 0
repeat_interval: 0
data: "{
  "fault_code": 0x0001
  "fault_state": 0-Normal 1-Warning 2-Shut down
}"

cmd_id: 5 -Get range of acceleration (to lidar):
cmd_type: 0
repeat_interval: 0
data: ""

cmd_id: 5 -The feedback of getting range of acceleration (to user):
cmd_type: 0
repeat_interval: 0
data: "{
  "acceleration_range": 0-±2G 1-±4G
}"

cmd_id: 1 -Set range of acceleration (to lidar):
cmd_type: 1
repeat_interval: 0
data: "{
  "acceleration_range": 0-±2G 1-±4G
}"

cmd_id: 1 -The feedback of getting range of acceleration (to user):
cmd_type: 0
repeat_interval: 0
data: "{
  "acceleration_range": 0-±2G 1-±4G
}"
