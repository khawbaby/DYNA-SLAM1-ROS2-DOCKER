from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        output='screen',
        parameters=[
            '/root/colcon_ws/src/realsense_handler/config/realsense.yaml'
        ]
    )

    camera_handler_node = Node(
        package='realsense_handler',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    return LaunchDescription([
        realsense_node,
        camera_handler_node
    ])