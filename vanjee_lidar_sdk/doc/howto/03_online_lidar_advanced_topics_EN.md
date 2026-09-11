# 3 Online LiDAR - Advanced Topics

## 3.1 Introduction

+ The VanJee LiDAR can operate in the following scenarios:
  - Unicast/Multicast.
  - Connecting to multiple LiDARs.

This document describes how to configure the `vanjee_lidar_sdk` in these scenarios.

Before reading this document, please ensure that you have read:
+ LiDAR product Manual
+ [Parameter Introduction](../intro/02_parameter_intro_CN.md) 
+ [How to decode online lidar](./02_how_to_decode_online_lidar_CN.md)

## 3.2 Unicast, Multicast

### 3.2.1 Unicast

To reduce network load, it is recommended for the LiDAR to use unicast mode.
+ The LiDAR sends packets to `192.168.1.102:3001`, and the `vanjee_lidar_sdk` binds to port `3001`.

Below is the configuration in `config.yaml`, which is similar to the broadcast method.

```yaml
lidar:
  - driver:
      lidar_type: vajee_720_16           
      host_msop_port: 3001
      lidar_msop_port: 3333
      host_address: 192.168.2.88
      lidar_address: 192.168.2.86
```

### 3.2.2 Multicast

The LiDAR can also operate in multicast mode.

+ The LiDAR sends packets to `224.1.1.1:3001` in multicast mode.
+ The `vanjee_lidar_sdk` binds to port `3001`. Additionally, it joins the multicast group `224.1.1.1` on the local network interface with IP address `192.168.1.102`.

Below is the configuration in `config.yaml`

```yaml
lidar:
  - driver:
      lidar_type: vajee_720_16        
      host_msop_port: 3001  
      group_address: 224.1.1.1
      host_address: 192.168.2.88
```

## 3.3 The situation of multiple LiDAR

### 3.3.1 Different Target Port

If you have two or more LiDAR devices, the preferred configuration is to assign them different target ports to avoid conflicts.

+ The first LiDAR sends packets to `192.168.1.102:3001`, and the first driver node configured in the `vanjee_lidar_sdk` binds to port `3001`.
+ The second LiDAR sends packets to `192.168.1.102:3001`, and the second driver node configured in the `vanjee_lidar_sdk` binds to port `3001`.

Below is the configuration in `config.yaml`:

```yaml
lidar:
  - driver:
      lidar_type: vajee_720_16          
      host_msop_port: 3001
      lidar_msop_prot: 3333
      host_address: 192.168.2.88
      lidar_address: 192.168.2.85  
  - driver:
      lidar_type: vajee_720_16      
      host_msop_port: 3002
      lidar_msop_prot: 3333
      host_address: 192.168.2.88
      lidar_address: 192.168.2.86  
```
