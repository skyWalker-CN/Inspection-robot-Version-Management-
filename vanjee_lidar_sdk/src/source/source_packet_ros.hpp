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
#include "source/source.hpp"
#ifdef ROS_FOUND
#include <ros/ros.h>
#include <vanjee_lidar_sdk/VanjeelidarPacket.h>
namespace vanjee {
namespace lidar {
inline vanjee_lidar_sdk::VanjeelidarPacket toRosMsg(const Packet &vanjee_msg, const std::string &frame_id) {
  vanjee_lidar_sdk::VanjeelidarPacket ros_msg;

  ros_msg.header.seq = vanjee_msg.seq;
  ros_msg.header.stamp = ros_msg.header.stamp.fromSec(vanjee_msg.timestamp);
  ros_msg.header.frame_id = frame_id;

  for (size_t i = 0; i < vanjee_msg.buf.size(); i++) {
    ros_msg.data.emplace_back(vanjee_msg.buf[i]);
  }

  return ros_msg;
}
/// @brief 'DestinationPacketRos' publish pakcet command through ROS topic
/// '/vanjee_packet'
class DestinationPacketRos : public DestinationPacket {
 private:
  std::shared_ptr<ros::NodeHandle> nh_;
  ros::Publisher packet_pub_;
  std::string frame_id_;

 public:
  /// @brief Initialize the instance of 'DestinationPacketRos'
  virtual void init(const YAML::Node &config);
  /// @brief Publish packet command through ROS topic '/vanjee_packet'
  virtual void sendPacket(const Packet &msg);
  virtual ~DestinationPacketRos() = default;
};

inline void DestinationPacketRos::init(const YAML::Node &config) {
  yamlRead<std::string>(config["ros"], "ros_frame_id", frame_id_, "vanjee_lidar");
  std::string ros_send_topic;
  yamlRead<std::string>(config["ros"], "ros_packet_topic", ros_send_topic, "vanjee_lidar_packet");

  nh_ = std::unique_ptr<ros::NodeHandle>(new ros::NodeHandle());
  packet_pub_ = nh_->advertise<vanjee_lidar_sdk::VanjeelidarPacket>(ros_send_topic, 10);
}
inline void DestinationPacketRos::sendPacket(const Packet &msg) {
  packet_pub_.publish(toRosMsg(msg, frame_id_));
}
}  // namespace lidar

}  // namespace vanjee

#endif
#ifdef ROS2_FOUND
#include "rclcpp/rclcpp.hpp"
#include "source/source_ros_msg_delegate.hpp"
#include "vanjee_lidar_msg/msg/vanjeelidar_packet.hpp"
namespace vanjee {
namespace lidar {
inline vanjee_lidar_msg::msg::VanjeelidarPacket toRosMsg(const Packet &vanjee_msg, const std::string &frame_id) {
  vanjee_lidar_msg::msg::VanjeelidarPacket ros_msg;

  rclcpp::Time ros_time(vanjee_msg.timestamp * 1e9);
  ros_msg.header.stamp = ros_time;
  ros_msg.header.frame_id = frame_id;

  for (size_t i = 0; i < vanjee_msg.buf.size(); i++) {
    ros_msg.data.emplace_back(vanjee_msg.buf[i]);
  }

  return ros_msg;
}
// template class VanjeeLidarSdkPublishRosMsg<vanjee_lidar_msg::msg::VanjeelidarPacket>;
/// @brief 'DestinationPacketRos' publish packet command through ROS topic
/// '/vanjee_packet'
class DestinationPacketRos : virtual public DestinationPacket {
 private:
  using PacketMsgPubPtr = std::shared_ptr<VanjeeLidarSdkPublishRosMsg<vanjee_lidar_msg::msg::VanjeelidarPacket>>;
  std::string frame_id_;
  PacketMsgPubPtr packet_msg_pub_ptr_;

 public:
  /// @brief Initialize the instance of 'DestinationPacketRos'
  virtual void init(const YAML::Node &config);
  /// @brief Publish packet command through ROS topic '/vanjee_packet'
  virtual void sendPacket(const Packet &msg);
  virtual ~DestinationPacketRos() = default;
};

inline void DestinationPacketRos::init(const YAML::Node &config) {
  yamlRead<std::string>(config["ros"], "ros_frame_id", frame_id_, "vanjee_lidar");
  std::string ros_send_topic;
  yamlRead<std::string>(config["ros"], "ros_packet_topic", ros_send_topic, "vanjee_lidar_packet");
  packet_msg_pub_ptr_ = VanjeeLidarSdkNode::CreateInstance()->GetPacketMsgPublisher(ros_send_topic);
}
inline void DestinationPacketRos::sendPacket(const Packet &msg) {
  vanjee_lidar_msg::msg::VanjeelidarPacket packet = toRosMsg(msg, frame_id_);
  packet_msg_pub_ptr_->PublishMsg(packet);
}
}  // namespace lidar

}  // namespace vanjee

#endif