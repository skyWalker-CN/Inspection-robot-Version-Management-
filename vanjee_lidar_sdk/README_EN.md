# 1 **vanjee_lidar_sdk**

## 1.1 Project Overview

 **vanjee_lidar_sdk** is the LiDAR driver software package for VanJee LiDAR in the Ubuntu environment. It includes：

 + LiDAR drive kernel[vanjee_driver]， 
 + ROS extension functions

If you wish to perform secondary development based on ROS, you can use this software package.Working with the built-in visualization tool rviz in ROS, you can view point clouds.

If you wish to integrate the LiDAR driver into your own project for further secondary development, please develop based on vanjee_driver.

### 1.1.1 Supported LiDAR models

- vanjee_716mini
- vanjee_718h
- vanjee_719
- vanjee_719c
- vanjee_719e
- vanjee_720 / vanjee_720_16
- vanjee_720_32
- vanjee_721
- vanjee_722
- vanjee_722f
- vanjee_722h
- vanjee_722z
- vanjee_733
- vanjee_750
- vanjee_760

### 1.1.2 Supported point types

- XYZI - x, y, z, intensity
- XYZIRT - x, y, z, intensity, ring, timestamp
- XYZIRTT - x, y, z, intensity, ring, timestamp tag

## 1.2 Dependency introduction

### 1.3.1 ROS 

To use the LiDAR driver in the ROS environment, you need to install ROS-related dependent libraries.

+ Ubuntu 16.04 - ROS Kinetic desktop
+ Ubuntu 18.04 - ROS Melodic desktop
+ Ubuntu 20.04 - ROS Noetic desktop

Please refer to the installation methods provided at http://wiki.ros.org。

**It is strongly recommended to install the ROS desktop-full version. This process will automatically install some compatible versions of dependent libraries, such as the PCL library, etc. This can help avoid spending a lot of time installing and configuring them individually.**


### 1.3.2 ROS2

To use the LiDAR driver in the ROS2 environment, you need to install ROS2-related dependent libraries.

+ Ubuntu 16.04 - Not supported
+ Ubuntu 18.04 - ROS2 Eloquent desktop
+ Ubuntu 20.04 - ROS2 Galactic desktop
+ Ubuntu 22.04 - ROS2 Humble desktop
+ Ubuntu 24.04 - ROS2 Jazzy desktop

Please refer to the installation methods provided at https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Install-Debians/

**Please avoid installing ROS and ROS2 simultaneously on one computer to prevent potential version conflicts and the hassle of manually installing other libraries (such as Yaml)**

### 1.3.3 Yaml (indispensable)

Version: >= v0.5.2

*If ROS desktop-full is already installed, you can skip this step.*

Installation method is as follows:

```sh
sudo apt-get update
sudo apt-get install -y libyaml-cpp-dev
```

### 1.3.4 libpcap (indispensable)

Version: >= v1.7.4

Installation method is as follows:

```sh
sudo apt-get install -y  libpcap-dev
```

## 1.4 Compile and run

You can use three methods to compile and run vanjee_lidar_sdk.。
**Before starting the ROS node, please rename the configuration file for the corresponding LiDAR model (e.g., config721.yaml) in the config folder to config.yaml**

**If there is an error message indicating that "libVanJeeLaser760Filter_linux.so" cannot be found during compilation or execution, the following steps should be taken**
```sh
sudo cp <PROJECT_PATH>/src/vanjee_lidar_sdk/src/vanjee_driver/libVanJeeLaser760Filter_linux.so /usr/local/lib
sudo ldconfig
```
**If there is still an error message indicating that "libVanJeeLaser760Filter_linux.so" cannot be found during compilation or execution, the following steps should be taken**
To check the permissions of the dynamic library, you can execute the following command.
```sh
ls -la <PROJECT_PATH>/src/vanjee_lidar_sdk/src/vanjee_driver/libVanJeeLaser760Filter_linux.so
sudo chmod 777 <PROJECT_PATH>/src/vanjee_lidar_sdk/src/vanjee_driver/libVanJeeLaser760Filter_linux.so
sudo cp <PROJECT_PATH>/src/vanjee_lidar_sdk/src/vanjee_driver/libVanJeeLaser760Filter_linux.so /usr/local/lib
```
You can try compiling the driver with root privileges, and the executable command is as follows:
ROS1
```sh
sudo catkin_make
```
ROS2
```sh
sudo colcon build

### 1.4.1 Compile directly

(1) Open the *CMakeLists.txt* file in the project, and change the variable **COMPILE_METHOD** at the top of the file to **ORIGINAL**.

```cmake
#=======================================
# Compile setup (ORIGINAL,CATKIN,COLCON)
#=======================================
set(COMPILE_METHOD ORIGINAL)
```

(2) Compile and run the program directly in ROS1. 

First, start **roscore**, then run **vanjee_lidar_sdk_node**, and finally run **rviz** to visualize the point cloud.

```sh
cd vanjee_lidar_sdk
mkdir build && cd build
cmake .. && make -j4
./vanjee_lidar_sdk_node
```

### 1.4.2 Dependent on ROS-catkin compilation

(1) Open the *CMakeLists.txt* file in the project, and change the variable **COMPILE_METHOD** at the top of the file to **CATKIN**.


```cmake
#=======================================
# Compile setup (ORIGINAL,CATKIN,COLCON)
#=======================================
set(COMPILE_METHOD CATKIN)
```

(2) Create a new folder as the workspace, then create a folder named *src* within it. Place the vanjee_lidar_sdk project into the *src* folder.

(3) Return to the workspace directory, then execute the following commands to compile and run.

```sh
catkin_make
source devel/setup.bash
roslaunch vanjee_lidar_sdk start.launch
```

### 1.4.3 Dependent on ROS2-colcon compilation

(1) Open the *CMakeLists.txt* file in the project, and change the variable **COMPILE_METHOD** at the top of the file to **COLCON**.

```cmake
#=======================================
# Compile setup (ORIGINAL,CATKIN,COLCON)
#=======================================
set(COMPILE_METHOD COLCON)
```

(2) Rename the *package_ros2.xml* file in the vanjee_lidar_sdk project directory to *package.xml*.

(3) Create a new folder as the workspace, then create a folder named *src* within it. Place the vanjee_lidar_sdk project into the *src* folder.

(4) Define the lidar packet message in the ROS2 environment, extract the vanjee_lidar_msg project to the newly created * src * folder, and place it side by side with vanjee_lidar sdk

(5) Return to the workspace directory, then execute the following commands to compile and run. If using .zsh, replace the second line with *source install/setup.zsh*.

```sh
colcon build
source install/setup.bash
ros2 launch vanjee_lidar_sdk start.py
```

The format of start.py may vary depending on different ROS2 versions. Please use the corresponding version's start.py. For example, for ROS2 Eloquent, please use eloquent_start.py.

## 1.5 Parameter introduction

The functionalities of vanjee_lidar_sdk are implemented through configuration parameter files. Please read them carefully. 

[Parameter Introduction](doc/intro/02_parameter_intro_CN.md)


## 1.6 Quick Start

Here are some usage guidelines for commonly used functionalities.

[Connect to online LiDAR data and publish point clouds to ROS](doc/howto/02_how_to_decode_online_lidar_CN.md)

[Parse PCAP packets and publish point clouds to ROS](doc/howto/04_how_to_decode_pcap_file_CN.md)

[Switch point types](doc/howto/01_how_to_change_point_type_CN.md) 

