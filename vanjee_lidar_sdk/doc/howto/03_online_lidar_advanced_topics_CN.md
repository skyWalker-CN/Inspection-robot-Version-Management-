# 3 在线雷达 - 高级主题

## 3.1 简介

万集激光雷达可以工作在如下的场景。

+ 单播/组播。
+ 接入多个雷达。

本文描述在这些场景下如何配置vanjee_lidar_sdk。

在阅读本文档之前， 请确保已经阅读过：
+ 雷达用户手册 
+ [参数介绍](../intro/02_parameter_intro_CN.md) 
+ [连接在线雷达](./02_how_to_decode_online_lidar_CN.md)

## 3.2 单播、组播

### 3.2.1 单播

为了减少网络负载，建议雷达使用单播模式。
+ 雷达发送Packet到 `192.168.2.88` : `3001`, vanjee_lidar_sdk绑定端口 `3001`。

如下是配置`config.yaml`的方式。这实际上与广播的方式一样。

```yaml
lidar:
  - driver:
      lidar_type: vajee_720_16           
      host_msop_port: 3001
      lidar_msop_port: 3333
      host_address: 192.168.2.88
      lidar_address: 192.168.2.86
```

### 3.2.2 组播

雷达也可以工作在组播模式。
+ 雷达发送Packet到 `224.1.1.1`:`3001` 
+ vanjee_lidar_sdk绑定到端口 `3001`。同时它将IP地址为`192.168.2.88`的本地网络接口加入组播组`224.1.1.1`。

如下是配置`config.yaml`的方式。

```yaml
lidar:
  - driver:
      lidar_type: vanjee_720_16        
      host_msop_port: 3001  
      group_address: 224.1.1.1
      host_address: 192.168.2.88
```

## 3.3 多个雷达的情况

### 3.3.1 不同的目标端口

如果有两个或多个雷达，首选的配置是让它们有不同的目标端口。
+ 第一个雷达发送Packet到 `192.168.2.88`:`3001`, 给vanjee_lidar_sdk配置的第一个driver节点绑定到`3001`。
+ 第二个雷达发送Packet到 `192.168.2.88`:`3002`, 给vanjee_lidar_sdk配置的第二个driver节点绑定到`3002`。

如下是配置`config.yaml`的方式。

```yaml
lidar:
  - driver:
      lidar_type: vanjee_720_16          
      host_msop_port: 3001
      lidar_msop_prot: 3333
      host_address: 192.168.2.88
      lidar_address: 192.168.2.85  
  - driver:
      lidar_type: vanjee_720_16      
      host_msop_port: 3002
      lidar_msop_prot: 3333
      host_address: 192.168.2.88
      lidar_address: 192.168.2.86  
```
