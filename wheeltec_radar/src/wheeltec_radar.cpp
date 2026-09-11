#include "wheeltec_radar/wheeltec_radar.hpp"

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
  while(rclcpp::ok())
  {
  uint8_t Receive_Data_Pr[1]; //Temporary variable to save the data of the lower machine //临时变量，保存下位机数据
  static int count; //Static variable for counting //静态变量，用于计数
  radar_Serial.read(Receive_Data_Pr,sizeof(Receive_Data_Pr)); //Read the data sent by the lower computer through the serial port //通过串口读取下位机发送过来的数据
    
  Receive_Data[count] = Receive_Data_Pr[0];
  if(Receive_Data_Pr[0] == 0xAA || count >0)
    count++;
  else
    count=0;
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


turn_on_radar::turn_on_radar():rclcpp::Node ("wheeltec_radar")
{
  this->declare_parameter<int>("serial_baud_rate");
  this->declare_parameter<std::string>("usart_port_name", "/dev/wheeltec_lidar");

  this->get_parameter("serial_baud_rate", serial_baud_rate);//Communicate baud rate 115200 to the lower machine //和下位机通信波特率115200
  this->get_parameter("usart_port_name", usart_port_name);//Fixed serial port number //固定串口号

  pos_Pub = create_publisher<radar_msgs::msg::RadarScan>("/radarscan", 10);
  track_Pub = create_publisher<radar_msgs::msg::RadarTracks>("/radartrack", 10);
  pub_pointcloud_ = create_publisher<PointCloud2>("/radarpointcloud", 10);

  try
  { 
    //Attempts to initialize and open the serial port //尝试初始化与开启串口
    radar_Serial.setPort(usart_port_name); //Select the serial port number to enable //选择要开启的串口号
    radar_Serial.setBaudrate(serial_baud_rate); //Set the baud rate //设置波特率
    serial::Timeout _time = serial::Timeout::simpleTimeout(2000); //Timeout //超时等待
    radar_Serial.setTimeout(_time);
    radar_Serial.open(); //Open the serial port //开启串口
    radar_Serial.flushInput();
  }
  catch (serial::IOException& e)
  {
    RCLCPP_ERROR(this->get_logger(),"wheeltec_radar can not open serial port,Please check the serial port cable! "); //If opening the serial port fails, an error message is printed //如果开启串口失败，打印错误信息
  }
  if(radar_Serial.isOpen())
  {
    RCLCPP_INFO(this->get_logger(),"wheeltec_radar serial port opened"); //Serial port opened successfully //串口开启成功提示
  }
  GenerateUUIDTable();
}

turn_on_radar::~turn_on_radar()
{
  radar_Serial.close(); //Close the serial port //关闭串口  
  RCLCPP_INFO(this->get_logger(),"Shutting down"); //Prompt message //提示信息
}

