from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    config_path=get_package_share_directory('vanjee_lidar_sdk')+'/config/config_autoware.yaml'
    rviz_config=get_package_share_directory('vanjee_lidar_sdk')+'/rviz/rviz2.rviz'

    return LaunchDescription([
        Node(namespace='vanjee_lidar_sdk', package='vanjee_lidar_sdk', executable='vanjee_lidar_sdk_node', output='screen', parameters=[{'config_path':config_path}]),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='velodyne_to_base_link',
            arguments=['0.09', '0', '0.15', '0', '0', '0','base_link','vanjee_lidar'],
            ),
        Node(
            package='pointcloud_converter',
            executable='converter_node',
            name='pointcloud_converter',
            ),

    ])
