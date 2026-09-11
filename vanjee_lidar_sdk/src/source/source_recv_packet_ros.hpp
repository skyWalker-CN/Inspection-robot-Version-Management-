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
#include <vanjee_driver/msg/device_ctrl_msg.hpp>

#include "source/source.hpp"
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <vanjee_lidar_sdk/VanjeelidarPacket.h>
namespace vanjee {
namespace lidar {
inline std::shared_ptr<Packet> getRosMsg(const vanjee_lidar_sdk::VanjeelidarPacket::ConstPtr &msg) {
  std::shared_ptr<Packet> vanjee_msg = std::shared_ptr<Packet>(new Packet());
  vanjee_msg->seq = msg->header.seq;
  vanjee_msg->timestamp = msg->header.stamp.toSec();

  // for (size_t i = 0; i < msg->data.size(); i++) {
  //   vanjee_msg->buf.emplace_back(msg->data[i]);
  // }
  vanjee_msg->buf.assign(msg->data.begin(), msg->data.end());

  return vanjee_msg;
}
/// @brief 'SourcePacketRos' subscribe packet through ROS topic
/// '/vanjee_packet'
class SourcePacketRos : public SourcePacket {
 private:
  std::shared_ptr<ros::NodeHandle> nh_;
  ros::Subscriber packet_sub_;
  std::string frame_id_;

 public:
  virtual ~SourcePacketRos() = default;
  /// @brief Initialize the instance of 'SourcePacketRos'
  virtual void init(const YAML::Node &config);

  /// @brief Subscribe packet through ROS topic '/vanjee_packet'
  void recvPacket(const vanjee_lidar_sdk::VanjeelidarPacket::ConstPtr &msg);
};

inline void SourcePacketRos::init(const YAML::Node &config) {
  yamlRead<std::string>(config["ros"], "ros_frame_id", frame_id_, "vanjee_lidar");
  std::string ros_recv_topic;
  yamlRead<std::string>(config["ros"], "ros_packet_topic", ros_recv_topic, "vanjee_lidar_packet");

  nh_ = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());
  packet_sub_ = nh_->subscribe<vanjee_lidar_sdk::VanjeelidarPacket>(ros_recv_topic, 100, &SourcePacketRos::recvPacket, this);
}
inline void SourcePacketRos::recvPacket(const vanjee_lidar_sdk::VanjeelidarPacket::ConstPtr &msg) {
  cached_message_.push(getRosMsg(msg));
}

}  // namespace lidar

}  // namespace vanjee

#endif
#ifdef ROS2_FOUND
#include <rclcpp/rclcpp.hpp>

#include "source/source_ros_msg_delegate.hpp"
#include "vanjee_lidar_msg/msg/vanjeelidar_packet.hpp"
namespace vanjee {
namespace lidar {
inline std::shared_ptr<Packet> getRosMsg(const vanjee_lidar_msg::msg::VanjeelidarPacket::SharedPtr msg) {
  std::shared_ptr<Packet> vanjee_msg = std::shared_ptr<Packet>(new Packet());
  vanjee_msg->timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

  // for (size_t i = 0; i < msg->data.size(); i++) {
  //   vanjee_msg->buf.emplace_back(msg->data[i]);
  // }
  vanjee_msg->buf.assign(msg->data.begin(), msg->data.end());
  return vanjee_msg;
}

/// @brief 'SourcePacketRos' subscribe packet through ROS topic
/// '/vanjee_packet'
class SourcePacketRos : virtual public SourcePacket {
 private:
  using PacketMsgSubPtr = std::shared_ptr<VanjeeLidarSdkSubscribeRosMsg<vanjee_lidar_msg::msg::VanjeelidarPacket, vanjee::lidar::Packet>>;
  PacketMsgSubPtr packet_msg_sub_ptr_;
  std::string frame_id_;

 public:
  virtual ~SourcePacketRos() = default;
  /// @brief Initialize the instance of 'SourcePacketRos'
  virtual void init(const YAML::Node &config);

  /// @brief subscribe packet through ROS topic '/vanjee_packet'
  void recvPacket(const vanjee_lidar_msg::msg::VanjeelidarPacket::SharedPtr msg);
};

inline void SourcePacketRos::init(const YAML::Node &config) {
  yamlRead<std::string>(config["ros"], "ros_frame_id", frame_id_, "vanjee_lidar");
  std::string ros_recv_topic;
  yamlRead<std::string>(config["ros"], "ros_packet_topic", ros_recv_topic, "vanjee_lidar_packet");
  packet_msg_sub_ptr_ = VanjeeLidarSdkNode::CreateInstance()->GetPacketMsgSubscriber(ros_recv_topic);
  packet_msg_sub_ptr_->SubscribeTopic(std::bind(&SourcePacketRos::recvPacket, this, std::placeholders::_1));
}
inline void SourcePacketRos::recvPacket(const vanjee_lidar_msg::msg::VanjeelidarPacket::SharedPtr msg) {
  cached_message_.push(getRosMsg(msg));
}

}  // namespace lidar

}  // namespace vanjee

#endif