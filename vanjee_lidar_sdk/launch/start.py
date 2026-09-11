#from launch import LaunchDescription
#from launch_ros.actions import Node
#from ament_index_python.packages import get_package_share_directory

#def generate_launch_description():

    #config_path=get_package_share_directory('vanjee_lidar_sdk')+'/config/config.yaml'
    #rviz_config=get_package_share_directory('vanjee_lidar_sdk')+'/rviz/rviz2.rviz'

    #return LaunchDescription([
        #Node(namespace='vanjee_lidar_sdk', package='vanjee_lidar_sdk', executable='vanjee_lidar_sdk_node', output='screen', parameters=[{'config_path':config_path}]),
        #Node(
            #package='tf2_ros',
            #executable='static_transform_publisher',
            #name='velodyne_to_base_link',
            #arguments=['0.09', '0', '0.15', '0', '0', '0','base_link','vanjee_lidar'],
            #),
        #Node(namespace='rviz2', package='rviz2', executable='rviz2', arguments=['-d ',rviz_config])
    #])




#from launch import LaunchDescription
#from launch_ros.actions import Node
#from ament_index_python.packages import get_package_share_directory

#def generate_launch_description():

    #config_path = get_package_share_directory('vanjee_lidar_sdk') + '/config/config.yaml'
    #rviz_config = get_package_share_directory('vanjee_lidar_sdk') + '/rviz/rviz2.rviz'

    #return LaunchDescription([
        # 原始雷达驱动（命名空间 /vanjee_lidar_sdk）
        #Node(
            #namespace='vanjee_lidar_sdk',
            #package='vanjee_lidar_sdk',
            #executable='vanjee_lidar_sdk_node',
            #output='screen',
            #parameters=[{'config_path': config_path}]
        #),

        # 3D点云 → 2D激光扫描
        #Node(
            #package='pointcloud_to_laserscan',
            #executable='pointcloud_to_laserscan_node',
            #name='pointcloud_to_laserscan',
            #output='screen',
            #parameters=[{
                #'target_frame': 'base_link',
                #'min_height': -0.4,
                #'max_height': 0.0,
                #'angle_min': -3.14159,
                #'angle_max': 3.14159,
                #'angle_increment': 0.0087,
                #'scan_time': 0.1,
                #'range_min': 0.05,
                #'range_max': 30.0,
                #'use_inf': True,
            #}],
            #remappings=[
                #('cloud_in', '/rslidar_points'),   # 你的实际点云话题
                # 输出 /scan 不变
            #]
        #),

        # RViz（按需取消注释）
        # Node(
        #     namespace='rviz2',
        #     package='rviz2',
        #     executable='rviz2',
        #     arguments=['-d', rviz_config]
        # )
    #])


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_path = get_package_share_directory('vanjee_lidar_sdk') + '/config/config.yaml'

    return LaunchDescription([
        # 1. 原始雷达驱动
        Node(
            namespace='vanjee_lidar_sdk',
            package='vanjee_lidar_sdk',
            executable='vanjee_lidar_sdk_node',
            output='screen',
            parameters=[{'config_path': config_path}]
        ),

        # 2. 点云降采样（独立 Python 脚本）
        ExecuteProcess(
            cmd=['python3', 
                 '/home/jetson/ros2_ws/src/vanjee_lidar_sdk/scripts/pointcloud_downsample.py'],
            name='pointcloud_downsampler',
            output='screen'
        ),

        # 3. 点云转激光扫描（AMCL 用）
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            parameters=[{
                'target_frame': 'base_link',
                'min_height': 0.5,
                'max_height': 3.0,
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.0087,
                'scan_time': 0.1,
                'range_min': 0.05,
                'range_max': 30.0,
                'use_inf': True,
                'qos_overrides': {
                    '/rslidar_points_filtered': {
                        'subscription': {'reliability': 'reliable', 'history': 'keep_last', 'depth': 10}
                    },
                    '/scan': {
                        'publication': {'reliability': 'reliable', 'history': 'keep_last', 'depth': 10}
                    }
                }
            }],
            remappings=[
                ('cloud_in', '/rslidar_points_filtered'),  # 订阅降采样点云
            ]
        ),
    ])
