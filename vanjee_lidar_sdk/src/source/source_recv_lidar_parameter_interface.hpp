/*********************************************************************************************************************
Copyright (c) 2023 Vanjee
All rights reserved

By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.

License Agreement
For Vanjee LiDAR SDK Library
(3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the names of the Vanjee, nor Wanji Technology, nor the
names of other contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*********************************************************************************************************************/

#pragma once
#include <vanjee_driver/msg/lidar_parameter_interface_msg.hpp>

#include "source/source.hpp"
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <vanjee_lidar_sdk/LidarParameterInterface.h>
namespace vanjee {
namespace lidar {
inline std::shared_ptr<LidarParameterInterface> getRosMsg(const vanjee_lidar_sdk::LidarParameterInterface::ConstPtr &msg) {
  std::shared_ptr<LidarParameterInterface> vanjee_msg = std::shared_ptr<LidarParameterInterface>(new LidarParameterInterface());
  vanjee_msg->seq = msg->header.seq;
  vanjee_msg->timestamp = msg->header.stamp.toSec();
  vanjee_msg->cmd_id = msg->cmd_id;
  vanjee_msg->cmd_type = msg->cmd_type;
  vanjee_msg->repeat_interval = msg->repeat_interval;
  vanjee_msg->data = msg->data;

  return vanjee_msg;
}
/// @brief 'SourceLidarParameterInterfaceRos' subscribe lidar parameter through ROS topic
/// '/vanjee_lidar_parameter_sub'
class SourceLidarParameterInterfaceRos : public SourceLidarParameterInterface {
 private:
  std::shared_ptr<ros::NodeHandle> nh_;
  ros::Subscriber lidar_parameter_msg_sub_;
  std::string frame_id_;

 public:
  virtual ~SourceLidarParameterInterfaceRos() = default;
  /// @brief Initialize the instance of 'SourceLidarParameterInterfaceRos'
  virtual void init(const YAML::Node &config);

  /// @brief Subscribe lidar parameter through ROS topic '/vanjee_lidar_parameter_sub'
  void recvLidarParameter(const vanjee_lidar_sdk::LidarParameterInterface::ConstPtr &msg);
};

inline void SourceLidarParameterInterfaceRos::init(const YAML::Node &config) {
  yamlRead<std::string>(config["ros"], "ros_frame_id", frame_id_, "vanjee_lidar");
  std::string ros_recv_topic;
  yamlRead<std::string>(config["ros"], "ros_recv_lidar_parameter_cmd_topic", ros_recv_topic, "vanjee_lidar_parameter_sub");

  nh_ = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());
  lidar_parameter_msg_sub_ =
      nh_->subscribe<vanjee_lidar_sdk::LidarParameterInterface>(ros_recv_topic, 10, &SourceLidarParameterInterfaceRos::recvLidarParameter, this);
}
inline void SourceLidarParameterInterfaceRos::recvLidarParameter(const vanjee_lidar_sdk::LidarParameterInterface::ConstPtr &msg) {
  cached_message_.push(getRosMsg(msg));
}

}  // namespace lidar

}  // namespace vanjee

#endif
#ifdef ROS2_FOUND
#include <rclcpp/rclcpp.hpp>

#include "source/source_ros_msg_delegate.hpp"
#include "vanjee_lidar_msg/msg/lidar_parameter_interface.hpp"
namespace vanjee {
namespace lidar {
inline std::shared_ptr<LidarParameterInterface> getRosMsg(const vanjee_lidar_msg::msg::LidarParameterInterface::SharedPtr msg) {
  std::shared_ptr<LidarParameterInterface> vanjee_msg = std::shared_ptr<LidarParameterInterface>(new LidarParameterInterface());
  vanjee_msg->timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  vanjee_msg->cmd_id = msg->cmd_id;
  vanjee_msg->cmd_type = msg->cmd_type;
  vanjee_msg->repeat_interval = msg->repeat_interval;
  vanjee_msg->data = msg->data;
  return vanjee_msg;
}

/// @brief 'SourceLidarParameterInterfaceRos' subscribe lidar parameter through ROS topic
/// '/vanjee_lidar_parameter_sub'
class SourceLidarParameterInterfaceRos : virtual public SourceLidarParameterInterface {
 private:
  using LidarParameterInterfaceMsgSubPtr =
      std::shared_ptr<VanjeeLidarSdkSubscribeRosMsg<vanjee_lidar_msg::msg::LidarParameterInterface, vanjee::lidar::LidarParameterInterface>>;
  LidarParameterInterfaceMsgSubPtr lidar_parameter_msg_sub_ptr_;
  std::string frame_id_;

 public:
  virtual ~SourceLidarParameterInterfaceRos() = default;
  /// @brief Initialize the instance of 'SourceLidarParameterInterfaceRos'
  virtual void init(const YAML::Node &config);

  /// @brief Subscribe lidar parameter through ROS topic '/vanjee_lidar_parameter_sub'
  void recvLidarParameter(const vanjee_lidar_msg::msg::LidarParameterInterface::SharedPtr msg);
};

inline void SourceLidarParameterInterfaceRos::init(const YAML::Node &config) {
  yamlRead<std::string>(config["ros"], "ros_frame_id", frame_id_, "vanjee_lidar");
  std::string ros_recv_topic;
  yamlRead<std::string>(config["ros"], "ros_recv_lidar_parameter_cmd_topic", ros_recv_topic, "vanjee_lidar_parameter_sub");
  lidar_parameter_msg_sub_ptr_ = VanjeeLidarSdkNode::CreateInstance()->GetLidarParameterInterfaceMsgSubscriber(ros_recv_topic);
  lidar_parameter_msg_sub_ptr_->SubscribeTopic(std::bind(&SourceLidarParameterInterfaceRos::recvLidarParameter, this, std::placeholders::_1));
}
inline void SourceLidarParameterInterfaceRos::recvLidarParameter(const vanjee_lidar_msg::msg::LidarParameterInterface::SharedPtr msg) {
  cached_message_.push(getRosMsg(msg));
}

}  // namespace lidar

}  // namespace vanjee

#endif