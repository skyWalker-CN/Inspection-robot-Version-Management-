# 1 How to change the definition of point types

## 1.1 Introduction

This document describes how to change the definition of point types.

Set the `POINT_TYPE` variable in the project's `CMakeLists.txt` file. After making changes, you will need to rebuild the entire project.

```cmake
#=======================================
# Custom Point Type (XYZI, XYZIRT)
#=======================================
set(POINT_TYPE XYZI)
```

## 1.2 XYZI

When `POINT_TYPE` is `XYZI`, the vanjee_lidar_sdk uses Vanjee's custom point type `PointXYZI`

```c++
struct PointXYZI
{
  float x;
  float y;
  float z;
  float intensity;
};
```

The `vanjee_lidar_sdk` converts point clouds based on `PointXYZI` into ROS `PointCloud2` messages and then publishes them.

```c++
 sensor_msgs::PointCloud2 ros_msg;
 addPointField(ros_msg, "x", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "y", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "z", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "intensity", 1, sensor_msgs::PointField::FLOAT32, offset);

 // 
 // copy points from point cloud of `PointXYZI` to `PointCloud2`
 //
 ...
```

In `PointCloud2`, the `intensity` field is of type `float` rather than `uint8_t`. This is because most ROS-based programs expect `intensity` to be represented as a `float` type, allowing for more precision and compatibility with common processing methods in ROS.

## 1.3 XYZIRT

When `POINT_TYPE` is set to `XYZIRT`, the `vanjee_lidar_sdk` utilizes the custom point type `PointXYZRT` defined by Vanjee.

```c++
struct PointXYZIRT
{
  float x;
  float y;
  float z;
  float intensity;
  uint16_t ring;
  double timestamp;
};
```

The `vanjee_lidar_sdk` converts point clouds based on `PointXYZIRT` into ROS `PointCloud2` messages and then publishes them.

```c++
 sensor_msgs::PointCloud2 ros_msg;
 addPointField(ros_msg, "x", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "y", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "z", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "intensity", 1, sensor_msgs::PointField::FLOAT32, offset);
#ifdef POINT_TYPE_XYZIRT
 addPointField(ros_msg, "ring", 1, sensor_msgs::PointField::UINT16, offset);
 addPointField(ros_msg, "timestamp", 1, sensor_msgs::PointField::FLOAT64, offset);
#endif

 // 
 // copy points from point cloud of `PointXYZIRT` to `PointCloud2`
 //
 ...
```

## 1.3 XYZIRTT

When `POINT_TYPE` is set to `XYZIRTT`, the `vanjee_lidar_sdk` utilizes the custom point type `PointXYZRTT` defined by Vanjee.

```c++
struct PointXYZIRTT
{
  float x;
  float y;
  float z;
  float intensity;
  uint16_t ring;
  double timestamp;
  uint8_t tag;
};
```

The `vanjee_lidar_sdk` converts point clouds based on `PointXYZIRTT` into ROS `PointCloud2` messages and then publishes them.

```c++
 sensor_msgs::PointCloud2 ros_msg;
 addPointField(ros_msg, "x", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "y", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "z", 1, sensor_msgs::PointField::FLOAT32, offset);
 addPointField(ros_msg, "intensity", 1, sensor_msgs::PointField::FLOAT32, offset);
#ifdef POINT_TYPE_XYZIRT
 addPointField(ros_msg, "ring", 1, sensor_msgs::PointField::UINT16, offset);
 addPointField(ros_msg, "timestamp", 1, sensor_msgs::PointField::FLOAT64, offset);
#endif

#ifdef POINT_TYPE_XYZIRTT
 addPointField(ros_msg, "ring", 1, sensor_msgs::PointField::UINT16, offset);
 addPointField(ros_msg, "timestamp", 1, sensor_msgs::PointField::FLOAT64, offset);
 addPointField(ros_msg, "tag", 1, sensor_msgs::PointField::UINT8, offset);
#endif

 // 
 // copy points from point cloud of `PointXYZIRTT` to `PointCloud2`
 //
 ...
```
