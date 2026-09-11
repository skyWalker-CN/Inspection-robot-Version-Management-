import os 
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction,IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration,PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


#def launch(launch_descriptor, argv):
def generate_launch_description():

    

    # Extract common parameters
    common_params = {
        'group_ip': "224.1.1.2",                                    #组播ip
        'add_multicast': False ,                                    #是否启动组播
        'device_ip': "192.168.1.200"  ,                              #雷达源ip
        'device_ip_difop': "192.168.1.102"  ,                        #雷达目的IP
        'device_port': 2369  ,                                      #雷达源端口号
        'difop_port': 2368 ,                                       #雷达目的端口号
    }

    turn_on_radar = Node(
        package='wheeltec_radar', 
        executable='wheeltec_radar_net_node', 
        name='wheeltec_radar_net_node',
        output='screen',
        parameters=[common_params],
    )
    base_to_radar = Node(
        package='tf2_ros', 
        executable='static_transform_publisher', 
        name='base_to_radar',
        arguments=['0', '0', '0','0', '0','0','base_footprint','radar'],
    )


    ld = LaunchDescription()
    ld.add_action(turn_on_radar)
    #ld.add_action(base_to_radar)

    return ld
