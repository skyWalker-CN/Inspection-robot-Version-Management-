# 3 Introduction to Device Control Parameters

**Support lidar type:**
  WLR-716mini
  WLR-718h
  WLR-719
  WLR-720_16
  WLR-720_32
  WLR-719e
  WLR-722
  WLR-722f
  WLR-722h
  WLR-722z

## 3.1 Device control topic information

``` 
std_msgs/Header header 
uint16 cmd_id
uint16 cmd_param
uint8 cmd_state
``` 

``` header ```
device control topic header information:
ROS1 includes: sequence number(seq), timestamp (timestamp), coordinate system (frame id);
ROS2 includes: timestamp(timestamp),coordinate system (frame id)

``` cmd_id ```
command identification
1 - set device work mode 

``` cmd_param ```
command parameters
the parameters of "set device work mode" include:
0 - working mode
1 - standby mode
2 - only open channels 25-28(only for vanjee_722,/vanjee_722f/vanjee_722h/vanjee_722z is not effective)

``` cmd_state ```
command responded state
If the device responded the corresponding device control command correctly.User shall ignore this field whiling issuing the device control topics
Unless otherwise specified, the response state of all device control commands include:
0 - failure
1- success


``` device control topic header information
  uint32 seq = 0;
  float64 timestamp = 0.0;
  string frame_id = "";
```

## 3.2 Device control interface information

data structure:
class DeviceCtrl {
 public:
  uint32 seq = 0;
  float64 timestamp = 0.0;
  uint16 cmd_id = 0;
  uint16 cmd_param = 0;
  uint8 cmd_state = 0;
};

``` 
application program interface:
void deviceCtrlApi(const DeviceCtrl& device_ctrl_cmd);
``` 
device class:
LidarDriver<PointCloudMsg> driver;

## 3.2.1 Receive the responded state of device control command
Register callback function:
driver.regDeviceCtrlCallback(allocateDeviceCtrlMemoryCallback, deviceCtrlCallback);

Sdk store the responded state of device control command into the memory provided by the allocateDeviceCtrlMemoryCallback function
Receive the responded state of device control command by deviceCtrlCallback function

## 3.2.2 Send device control command
example of device control command:
DeviceCtrl deviceCtrl;
deviceCtrl.cmd_id = 1;
deviceCtrl.cmd_param = 0; // working mode; 1-standby mode; 2-only open channels 25-28(only for vanjee_722,/vanjee_722f/vanjee_722h/vanjee_722z is not effective)
deviceCtrl.cmd_state = 0;

Call interface:
driver.deviceCtrlApi(deviceCtrl);

**refer the demo for the specific implementation method**

## 3.3 Device control commands for all lidars

### 3.3.1 WLR-716mini

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: 0-Normal 1-Warning 2-Fault
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.2 WLR-718h

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: 0-Normal 1-Warning 2-Fault
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.3 WLR-719

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: 0-Normal 1-Warning 2-Fault
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.4 WLR-719E

cmd_id: 0x0100-0x010f - Device operating state (to user)
cmd_param: fault code
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.5 WLR-720_16

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: fault code
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.6 WLR-720_32

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: fault code
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.7 WLR-722

cmd_id: 0x0001 - Set lidar work mode (to lidar)
cmd_param: 0-working mode; 1-standby mode; 2-only open channels 25-28
cmd_state: 0

cmd_id: 0x0001 - The feedback of setting lidar operating mode command (to user)
cmd_param: 0-working mode; 1-standby mode; 2-only open channels 25-28
cmd_state: 0-failure 1-success

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: fault code
cmd_state: 0-Normal 1-Warning 2-Fault

### 3.3.8 WLR-722f

cmd_id: 0x0001 - Set lidar work mode (to lidar)
cmd_param: 0-working mode; 1-standby mode;
cmd_state: 0

cmd_id: 0x0001 - The feedback of setting lidar operating mode command (to user)
cmd_param: 0-working mode; 1-standby mode;
cmd_state: 0-failure 1-success

cmd_id: 0x0002 - Get lidar temperature (to user)
cmd_param: temperature (need to convert to int16, the true value needs to be divided by 100)
cmd_state: 0-get param failure 1-get param success

cmd_id: 0x0100-0x010f - Device operating state (to user)
cmd_param: error code (Each bit corresponds to a fault. If you need to confirm the fault type, please refer to the lidar product manual)
cmd_state: 1-Warning 2-Fault

### 3.3.9 WLR-722h

cmd_id: 0x0001 - Set lidar work mode (to lidar)
cmd_param: 0-working mode; 1-standby mode;
cmd_state: 0

cmd_id: 0x0001 - The feedback of setting lidar operating mode command (to user)
cmd_param: 0-working mode; 1-standby mode;
cmd_state: 0-failure 1-success

cmd_id: 0x0002 - Get lidar temperature (to user)
cmd_param: temperature (need to convert to int16, the true value needs to be divided by 100)
cmd_state: 0-get param failure 1-get param success

cmd_id: 0x0100-0x010f - Device operating state (to user)
cmd_param: error code (Each bit corresponds to a fault. If you need to confirm the fault type, please refer to the lidar product manual)
cmd_state: 1-Warning 2-Fault

### 3.3.10 WLR-722z

cmd_id: 0x0002 - Get lidar temperature (to user)
cmd_param: temperature (need to convert to int16, the true value needs to be divided by 100)
cmd_state: 0-get param failure 1-get param success

cmd_id: 0x0100 - Device operating state (to user)
cmd_param: fault code
cmd_state: 0-Normal 1-Warning 2-Shut down
