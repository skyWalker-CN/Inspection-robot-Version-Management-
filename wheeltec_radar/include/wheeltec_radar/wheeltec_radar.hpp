
#ifndef __WHEELTEC_RADAR_HPP_
#define __WHEELTEC_RADAR_HPP_

#include <iostream>
#include <string.h>
#include <string> 
#include <iostream>
#include <math.h> 
#include <stdlib.h>    
#include <unistd.h> 

#include "rclcpp/rclcpp.hpp"
#include <rcl/types.h>
#include <sys/stat.h>
#include <fcntl.h>          
#include <stdbool.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

      
#include <sys/types.h>

#include <serial/serial.h>
#include "std_msgs/msg/byte_multi_array.hpp"
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "radar_msgs/msg/radar_scan.hpp"
#include "radar_msgs/msg/radar_return.hpp"
#include "radar_msgs/msg/radar_tracks.hpp"
#include "unique_identifier_msgs/msg/uuid.hpp"
#include <vector>
#include <random>
#include "std_msgs/msg/byte_multi_array.hpp"

using namespace std;
using sensor_msgs::msg::PointCloud2;


class turn_on_radar : public rclcpp::Node
{
	public:
		turn_on_radar();  //Constructor //构造函数
		~turn_on_radar(); //Destructor //析构函数
		serial::Serial radar_Serial; //Declare a serial object //声明串口对象 
		void GenerateUUIDTable();
		void Get_Data();
		pcl::PointXYZI getRadarPointXYZI(const radar_msgs::msg::RadarReturn & radar, float intensity);
		sensor_msgs::msg::PointCloud2 toPointcloud2(const radar_msgs::msg::RadarScan & radar_scan);



        string usart_port_name; //Define the related variables //定义相关变量
        int serial_baud_rate;      //Serial communication baud rate //串口通信波特率
        uint8_t Receive_Data[17];

        rclcpp::Node::SharedPtr node_handle = nullptr;
        rclcpp::Publisher<radar_msgs::msg::RadarScan>::SharedPtr pos_Pub;
        rclcpp::Publisher<radar_msgs::msg::RadarTracks>::SharedPtr track_Pub;
        rclcpp::Publisher<PointCloud2>::SharedPtr pub_pointcloud_{};
        std::vector<unique_identifier_msgs::msg::UUID> UUID_table_;
        radar_msgs::msg::RadarScan scan_msg;
        radar_msgs::msg::RadarTracks tracks_msg;
};
#endif
