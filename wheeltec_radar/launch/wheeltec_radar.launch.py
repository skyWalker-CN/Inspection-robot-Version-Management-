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
        'usart_port_name': '/dev/ttyACM0',
        'serial_baud_rate': 115200,
    }

    turn_on_radar = Node(
        package='wheeltec_radar', 
        executable='wheeltec_radar_node', 
        name='wheeltec_radar_node',
        output='screen',
        parameters=[common_params],
    )
    #base_to_radar = Node(
        #package='tf2_ros', 
        #executable='static_transform_publisher', 
        #name='base_to_radar',
        #arguments=['0', '0', '0','0', '0','0','base_footprint','radar'],
    #)


    ld = LaunchDescription()
    ld.add_action(turn_on_radar)
    #ld.add_action(base_to_radar)

    return ld
