import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import AnyLaunchDescriptionSource

def generate_launch_description():

    #rplidar = IncludeLaunchDescription(
        #PythonLaunchDescriptionSource(
            #os.path.join(get_package_share_directory('rplidar_ros'),
                         #'launch', 'rplidar_s3_launch.py')
        #)
    #)

    rplidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('vanjee_lidar_sdk'),
                         'launch', 'start.py')
        )
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('orbbec_camera'),
                         'launch', 'gemini_330_series.launch.py')
        )
    )

    udf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'),
                         'launch', 'udf2tf.launch.py')
        )
    )

    ackerman = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ranger_bringup'),
                         'launch', 'ranger_mini_v3.launch.py')
        )
    )

    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'),
                         'launch', 'ekf.launch.py')
        )
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('nav2_bringup'),
                         'launch', 'navigation2.launch.py')
        ),
        launch_arguments = {
            'use_rviz': 'false',
            'log_level': 'info'
        }.items()
    )

    #followPoints = Node(
            #package = 'example_python',
            #executable = 'followPointsV6',
            #output = 'screen',
    #)

    hkws = Node(
            package = 'HKWS',
            executable = 'new_hksdk_1',
            output = 'screen',
            )

    detect = Node(
            package = 'anomaly_detector',
            executable = 'new_ptz_detect_warn',
            output = 'screen',
            )

    rosbridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rosbridge_server'),
                         'launch', 'rosbridge_websocket_launch.xml')
        )
    )

    web_detect = Node(
            package = 'example_python',
            executable = 'web_hard_1',
            output = 'screen',
            )

    web_hot = Node(
            package = 'example_python',
            executable = 'new_hotCam',
            output = 'screen',
            )


    return LaunchDescription([
        rosbridge,
        TimerAction(period = 5.0, actions=[hkws]),
        TimerAction(period = 15.0, actions=[detect]),
        TimerAction(period = 45.0, actions=[rplidar]),
        #TimerAction(period = 3.0, actions=[camera]),
        TimerAction(period = 55.0, actions=[ackerman]),
        TimerAction(period = 58.0, actions=[udf]),
        TimerAction(period = 61.0, actions=[ekf]),
        TimerAction(period = 64.0, actions=[nav2]),
        #TimerAction(period = 38.0, actions=[followPoints]),
        TimerAction(period = 85.0, actions=[web_detect]),
        TimerAction(period = 90.0, actions=[web_hot]),
    ])
