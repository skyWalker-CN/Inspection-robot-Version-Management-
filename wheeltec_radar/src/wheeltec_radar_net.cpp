#include "wheeltec_radar/wheeltec_radar_net.hpp"

using std::placeholders::_1;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv); //ROS initializes and sets the node name //ROS初始化 并设置节点名称 
  turn_on_radar radar_Control;//Instantiate an object //实例化一个对象
  radar_Control.Get_Data();//Loop through data collection and publish the topic //循环执行数据采集和发布话题等操作
  return 0;  
} 

pcl::PointXYZI turn_on_radar::getRadarPointXYZI(const radar_msgs::msg::RadarReturn & radar, float intensity)
{
  pcl::PointXYZI point;
  const float r_xy = radar.range * std::cos(radar.elevation);
  point.x = r_xy * std::cos(radar.azimuth);
  point.y = r_xy * std::sin(radar.azimuth);
  point.z = radar.range * std::sin(radar.elevation);
  point.intensity = intensity;
  return point;
}

sensor_msgs::msg::PointCloud2 turn_on_radar::toPointcloud2(const radar_msgs::msg::RadarScan & radar_scan)
{
  pcl::PointCloud<pcl::PointXYZI> pcl;
  for (const auto & radar : radar_scan.returns) {
    pcl.push_back(getRadarPointXYZI(radar, radar.amplitude));
  }
  sensor_msgs::msg::PointCloud2 pointcloud_msg;
  pcl::toROSMsg(pcl, pointcloud_msg);
  pointcloud_msg.header = radar_scan.header;
  return pointcloud_msg;
}

void turn_on_radar::GenerateUUIDTable() 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    UUID_table_.resize(100 + 1);  // 确保 UUID 表的大小足够

    for (size_t i = 0; i <= 100; ++i) {
        unique_identifier_msgs::msg::UUID uuid;
        for (size_t j = 0; j < uuid.uuid.size(); ++j) {
            uuid.uuid[j] = dist(gen);
        }
        UUID_table_[i] = uuid;
    }
}
void turn_on_radar::Get_Data()
{
  int count = 0;
  while(rclcpp::ok())
  {
    try {
      count = ReadFromIO(Receive_Data,2048);
    } catch (const std::exception& e) {
      //ROS_ERROR("Serial read failed: %s", e.what());
      continue;
    }
  if(count == 14)
  {
    count=0;
    if(Receive_Data[12] == 0x55 && Receive_Data[13] == 0x55)
    {
      if(Receive_Data[3]==0x06 && Receive_Data[2]==0x0b)
      {
      float datay = (Receive_Data[5] * 32 + (Receive_Data[6] >> 3)) * 0.1 - 500;
      float datax = (((Receive_Data[6] & 0x07) * 256 + Receive_Data[7]) * 0.1 - 102.3);    
      float datavelocity_y = (Receive_Data[8]<<2+(Receive_Data[9]>>6))*0.25-128;
      float datavelocity_x = ((Receive_Data[9]&0x3F)*8+(Receive_Data[10]>>5))*0.25-64;
      //ROS 右手坐标系
      float xx = datay;
      float yy = -datax;
      float velocity_x = datavelocity_y;
      float velocity_y = -datavelocity_x;
      float dis = sqrt(xx * xx + yy * yy);
      float vel = sqrt(velocity_x * velocity_x + velocity_y * velocity_y);
      float angle = atan2(yy, xx);


      radar_msgs::msg::RadarReturn radarreturn;
      radarreturn.range = dis; 
      radarreturn.azimuth = angle; 
      radarreturn.elevation = 0; 
      radarreturn.doppler_velocity = vel; 
      radarreturn.amplitude = 0;
      scan_msg.returns.push_back(radarreturn);
      
      radar_msgs::msg::RadarTrack out_object;
      out_object.uuid = UUID_table_[Receive_Data[4]];
      out_object.position.y = yy;
      out_object.position.x = xx;
      out_object.velocity.y = velocity_y;
      out_object.velocity.x = velocity_x;
      out_object.velocity_covariance.at(0) = 0.1;
      out_object.size.x = 0.5;
      out_object.size.y = 0.5;
      out_object.size.z = 1.0;
      out_object.classification = 32000;  //UNKNOWN_ID
      tracks_msg.tracks.push_back(out_object);
    }
    else if(Receive_Data[3]==0x06 && Receive_Data[2]==0x0a)
    {
      scan_msg.header.stamp = rclcpp::Node::now();
      scan_msg.header.frame_id = "radar";
      pos_Pub->publish(scan_msg);

      pub_pointcloud_->publish(toPointcloud2(scan_msg));
      
      tracks_msg.header.stamp = rclcpp::Node::now();
      tracks_msg.header.frame_id = "radar";
      track_Pub->publish(tracks_msg);
      
      scan_msg.returns.clear();
      tracks_msg.tracks.clear();
      
    }

    }
  }
    
  }
  return;
}
int turn_on_radar::ReadFromIO(uint8_t *rx_buf, uint32_t rx_buf_len) {
        //double time1 = rclcpp::Node::now().toSec();
        int q = 0;
        struct pollfd fds[1];
        fds[0].fd = sockfd_;
        fds[0].events = POLLIN;
        static const int POLL_TIMEOUT = 2000; // one second (in msec)

        sockaddr_in sender_address;
        socklen_t sender_address_len = sizeof(sender_address);
        // while (true)
        while (rclcpp::ok())
        {
            // poll() until input available
            do
            {
                int retval = poll(fds, 1, POLL_TIMEOUT);
                if (retval < 0) // poll() error?
                {
                    if (errno != EINTR)
                        printf("poll() error: %s", strerror(errno));
                    return 0;
                }
                if (retval == 0) // poll() timeout?
                {
                    RCLCPP_INFO(this->get_logger(),"lslidar poll() timeout");
                    return 0;
                }
                if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) || (fds[0].revents & POLLNVAL)) // device error?
                {
                    RCLCPP_ERROR(this->get_logger(),"poll() reports lslidar error");
                    return 0;
                }
            } while ((fds[0].revents & POLLIN) == 0);

            // Receive packets that should now be available from the
            // socket using a blocking read.
            ssize_t nbytes = recvfrom(sockfd_, rx_buf, rx_buf_len, 0,
                                      (sockaddr *)&sender_address, &sender_address_len);
            //        ROS_DEBUG_STREAM("incomplete lslidar packet read: "
            //                         << nbytes << " bytes");
            q = (int)nbytes;
            if (nbytes < 0)
            {
                if (errno != EWOULDBLOCK)
                {
                    perror("recvfail");
                    printf("recvfail");
                    return 1;
                }
            }
            else if ((size_t)nbytes <= rx_buf_len || (size_t)nbytes >= 50)
            {

                // read successful,
                // if packet is not from the lidar scanner we selected by IP,
                // continue otherwise we are done
                if (devip_str_ != "" && sender_address.sin_addr.s_addr != devip_.s_addr)
                    continue;
                else
                    break; // done
            }
            
        }
        // if (flag == 0)
        // {
        //     abort();
        // }
        // this->getFPGA_GPSTimeStamp(packet);

        // Average the times at which we begin and end reading.  Use that to
        // estimate when the scan occurred.
        //double time2 = rclcpp::Node::now().toSec();
        // packet->stamp = ros::Time((time2 + time1) / 2.0);
        // packet->stamp = this->timeStamp;
        //for(int i=0;i<rx_buf_len;i++)printHexData(rx_buf[i]);
        return q;
}

bool turn_on_radar::WriteToIo(const uint8_t *tx_buf, uint32_t tx_buf_len,
                                  uint32_t *tx_len) {
        sockaddr_in server_sai;
        server_sai.sin_family = AF_INET; // IPV4 协议族
        server_sai.sin_port = htons(UDP_PORT_NUMBER_DIFOP);
        server_sai.sin_addr.s_addr = inet_addr(devip_str_.c_str());
        int rtn = 0;
        rtn = sendto(sockfd_, tx_buf, tx_buf_len, 0, (struct sockaddr *)&server_sai, sizeof(struct sockaddr));
        if (rtn < 0)
        {
            printf("start scan error !\n");
        }
        else
        {
            usleep(3000000);
            return 1;
        }
        return 0;
}

turn_on_radar::turn_on_radar():rclcpp::Node ("wheeltec_radar")
{
  this->declare_parameter<std::string>("device_ip","192.168.1.200");
  this->declare_parameter<std::string>("device_ip_difop","192.168.1.102");
  this->declare_parameter<bool>("add_multicast",false);
  this->declare_parameter<std::string>("group_ip","224.1.1.2");
  this->declare_parameter<int>("device_port",2369);
  this->declare_parameter<int>("difop_port",2368);

  this->get_parameter("device_port", UDP_PORT_NUMBER_DIFOP);
  this->get_parameter("device_ip", devip_str_);
  this->get_parameter("add_multicast", add_multicast);
  this->get_parameter("group_ip", group_ip);
  this->get_parameter("difop_port", sid);
  this->get_parameter("device_ip_difop", devip_str_difop);

  pos_Pub = create_publisher<radar_msgs::msg::RadarScan>("/radarscan", 10);
  track_Pub = create_publisher<radar_msgs::msg::RadarTracks>("/radartrack", 10);
  pub_pointcloud_ = create_publisher<PointCloud2>("/radarpointcloud", 10);

  try
  { 
    sockfd_ = -1;
    if (!devip_str_.empty())
    {
        inet_aton(devip_str_.c_str(), &devip_);
        inet_aton(devip_str_difop.c_str(), &devip_difop);
    }
    RCLCPP_INFO(this->get_logger(),"Opening UDP socket ");
    sockfd_ = socket(PF_INET, SOCK_DGRAM, 0);
    if (sockfd_ == -1)
    {
        printf("socket"); // TODO: ROS_ERROR errno
        return;
    }
    int opt = 1;
    if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, (const void *)&opt, sizeof(opt)))
    {
        RCLCPP_ERROR(this->get_logger(),"setsockopt error!\n");
        return;
    }

    sockaddr_in my_addr;                         // my address information
    memset(&my_addr, 0, sizeof(my_addr));        // initialize to zeros
    my_addr.sin_family = AF_INET;                // host byte order
    my_addr.sin_port = htons(sid);              // port in network byte order
    my_addr.sin_addr.s_addr = htonl(INADDR_ANY); // automatically fill in my IP

    if (bind(sockfd_, (sockaddr *)&my_addr, sizeof(sockaddr)) == -1)
    {
        printf("bind"); // TODO: ROS_ERROR errno
        return;
    }
    if (add_multicast)
    {
        struct ip_mreq group;
        // char *group_ip_ = (char *) group_ip.c_str();
        group.imr_multiaddr.s_addr = inet_addr(group_ip.c_str());
        // group.imr_interface.s_addr =  INADDR_ANY;
        group.imr_interface.s_addr = htonl(INADDR_ANY);
        // group.imr_interface.s_addr = inet_addr("192.168.1.102");

        if (setsockopt(sockfd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&group, sizeof(group)) < 0)
        {
            RCLCPP_ERROR(this->get_logger(),"Adding multicast group error ");
            close(sockfd_);
            exit(1);
        }
        else
            printf("Adding multicast group...OK.\n");
    }

    if (fcntl(sockfd_, F_SETFL, O_NONBLOCK | FASYNC) < 0)
    {
        printf("non-block");
        return;
    }
  }
  catch (std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(),"wheeltec_radar can not open! "); //If opening the serial port fails, an error message is printed //如果开启串口失败，打印错误信息
  }
  GenerateUUIDTable();
}

turn_on_radar::~turn_on_radar()
{
  (void)close(sockfd_);  
  RCLCPP_INFO(this->get_logger(),"Shutting down"); //Prompt message //提示信息
}

