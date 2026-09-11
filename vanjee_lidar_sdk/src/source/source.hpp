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
#include <vanjee_driver/msg/imu_packet.hpp>
#include <vanjee_driver/msg/lidar_parameter_interface_msg.hpp>
#include <vanjee_driver/msg/packet.hpp>
#include <vanjee_driver/msg/scan_data_msg.hpp>
#include <vanjee_driver/utility/sync_queue.hpp>

#include "msg/lidar_point_cloud_msg.hpp"
#include "utility/yaml_reader.hpp"

namespace vanjee {
namespace lidar {
class DestinationPointCloud {
 public:
  typedef std::shared_ptr<DestinationPointCloud> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual void sendPointCloud(const LidarPointCloudMsg &msg) = 0;
  virtual ~DestinationPointCloud() = default;
};
class DestinationImuPacket {
 public:
  typedef std::shared_ptr<DestinationImuPacket> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual void sendImuPacket(const ImuPacket &msg) = 0;
  virtual ~DestinationImuPacket() = default;
};
class DestinationScanData {
 public:
  typedef std::shared_ptr<DestinationScanData> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual void sendScanData(const ScanData &msg) = 0;
  virtual ~DestinationScanData() = default;
};
class DestinationDeviceCtrl {
 public:
  typedef std::shared_ptr<DestinationDeviceCtrl> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual void sendDeviceCtrl(const DeviceCtrl &msg) = 0;
  virtual ~DestinationDeviceCtrl() = default;
};
class SourceDeviceCtrl {
 public:
  typedef std::shared_ptr<SourceDeviceCtrl> Ptr;
  SyncQueue<std::shared_ptr<DeviceCtrl>> cached_message_;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual ~SourceDeviceCtrl() = default;
};
class DestinationPacket {
 public:
  typedef std::shared_ptr<DestinationPacket> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual void sendPacket(const Packet &msg) = 0;
  virtual ~DestinationPacket() = default;
};
class SourcePacket {
 public:
  typedef std::shared_ptr<SourcePacket> Ptr;
  SyncQueue<std::shared_ptr<Packet>> cached_message_;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual ~SourcePacket() = default;
};
class DestinationLidarParameterInterface {
 public:
  typedef std::shared_ptr<DestinationLidarParameterInterface> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual void sendLidarParameter(const LidarParameterInterface &msg) = 0;
  virtual ~DestinationLidarParameterInterface() = default;
};
class SourceLidarParameterInterface {
 public:
  typedef std::shared_ptr<SourceLidarParameterInterface> Ptr;
  SyncQueue<std::shared_ptr<LidarParameterInterface>> cached_message_;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  virtual ~SourceLidarParameterInterface() = default;
};

enum SourceType { MSG_FROM_LIDAR = 1, MSG_FROM_PCAP = 2, MSG_FROM_PACKET = 3 };
class Source {
 protected:
  /// @brief Point cloud sending function
  void sendPointCloud(std::shared_ptr<LidarPointCloudMsg> msg);
  void sendImuPacket(const ImuPacket &msg);
  void sendScanData(const ScanData &msg);
  void sendDeviceCtrlState(const DeviceCtrl &msg);
  void sendPacket(const Packet &msg);
  void sendLidarParameter(const LidarParameterInterface &msg);

  SourceType src_type_;

  std::vector<DestinationPointCloud::Ptr> pc_cb_vec_;
  std::vector<DestinationImuPacket::Ptr> imu_pkt_cb_vec_;
  std::vector<DestinationScanData::Ptr> scan_data_cb_vec_;
  std::vector<DestinationDeviceCtrl::Ptr> device_ctrl_cb_vec_;
  std::shared_ptr<SourceDeviceCtrl> device_ctrl_ptr_;
  std::vector<DestinationPacket::Ptr> packet_cb_vec_;
  std::shared_ptr<SourcePacket> packet_ptr_;
  std::vector<DestinationLidarParameterInterface::Ptr> lidar_param_cb_vec_;
  std::shared_ptr<SourceLidarParameterInterface> lidar_param_ptr_;

 public:
  bool send_point_cloud_ros_;
  bool send_imu_packet_ros_;
  bool send_laser_scan_ros_;
  bool send_device_ctrl_state_ros_;
  bool recv_device_ctrl_cmd_ros_;
  bool send_packet_ros_;
  bool recv_packet_ros_;
  bool send_lidar_param_ros_;
  bool recv_lidar_param_cmd_ros_;

  typedef std::shared_ptr<Source> Ptr;
  virtual void init(const YAML::Node &config) {
  }
  virtual void start() {
  }
  virtual void stop() {
  }
  /// @brief Register point cloud callback function
  virtual void regPointCloudCallback(DestinationPointCloud::Ptr dst);
  /// @brief Register IMU packet callback function
  virtual void regImuPacketCallback(DestinationImuPacket::Ptr dst);
  /// @brief Register LaserScan packet callback function
  virtual void regScanDataCallback(DestinationScanData::Ptr dst);
  /// @brief Register DeviceCtrl packet callback function
  virtual void regDeviceCtrlCallback(DestinationDeviceCtrl::Ptr dst);
  virtual void setDeviceCtrlPtr(std::shared_ptr<SourceDeviceCtrl> source_device_ctrl);
  /// @brief Register packet callback function
  virtual void regPacketCallback(DestinationPacket::Ptr dst);
  virtual void setPacketPtr(std::shared_ptr<SourcePacket> source_packet);
  /// @brief Register lidar parameter callback function
  virtual void regLidarParameterInterfaceCallback(DestinationLidarParameterInterface::Ptr dst);
  virtual void setLidarParameterInterfacePtr(std::shared_ptr<SourceLidarParameterInterface> source_lidar_param);
  virtual ~Source() = default;
  Source(SourceType src_type);
};

inline Source::Source(SourceType src_type) : src_type_(src_type) {
}
inline void Source::regPointCloudCallback(DestinationPointCloud::Ptr dst) {
  pc_cb_vec_.emplace_back(dst);
}
inline void Source::regImuPacketCallback(DestinationImuPacket::Ptr dst) {
  imu_pkt_cb_vec_.emplace_back(dst);
}
inline void Source::regScanDataCallback(DestinationScanData::Ptr dst) {
  scan_data_cb_vec_.emplace_back(dst);
}
inline void Source::regDeviceCtrlCallback(DestinationDeviceCtrl::Ptr dst) {
  device_ctrl_cb_vec_.emplace_back(dst);
}
inline void Source::regPacketCallback(DestinationPacket::Ptr dst) {
  packet_cb_vec_.emplace_back(dst);
}
inline void Source::regLidarParameterInterfaceCallback(DestinationLidarParameterInterface::Ptr dst) {
  lidar_param_cb_vec_.emplace_back(dst);
}

inline void Source::sendPointCloud(std::shared_ptr<LidarPointCloudMsg> msg) {
  for (auto iter : pc_cb_vec_) {
    iter->sendPointCloud(*msg);
  }
}
inline void Source::sendImuPacket(const ImuPacket &msg) {
  for (auto iter : imu_pkt_cb_vec_) {
    iter->sendImuPacket(msg);
  }
}
inline void Source::sendScanData(const ScanData &msg) {
  for (auto iter : scan_data_cb_vec_) {
    iter->sendScanData(msg);
  }
}
inline void Source::sendDeviceCtrlState(const DeviceCtrl &msg) {
  for (auto iter : device_ctrl_cb_vec_) {
    iter->sendDeviceCtrl(msg);
  }
}
inline void Source::sendLidarParameter(const LidarParameterInterface &msg) {
  for (auto iter : lidar_param_cb_vec_) {
    iter->sendLidarParameter(msg);
  }
}

inline void Source::setDeviceCtrlPtr(std::shared_ptr<SourceDeviceCtrl> source_device_ctrl) {
  device_ctrl_ptr_ = source_device_ctrl;
}
inline void Source::sendPacket(const Packet &msg) {
  for (auto iter : packet_cb_vec_) {
    iter->sendPacket(msg);
  }
}
inline void Source::setPacketPtr(std::shared_ptr<SourcePacket> source_packet) {
  packet_ptr_ = source_packet;
}
inline void Source::setLidarParameterInterfacePtr(std::shared_ptr<SourceLidarParameterInterface> source_lidar_param) {
  lidar_param_ptr_ = source_lidar_param;
}

}  // namespace lidar

}  // namespace vanjee
