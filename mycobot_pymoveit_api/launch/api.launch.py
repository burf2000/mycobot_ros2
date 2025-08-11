from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
          package='mycobot_pymoveit_api',
          executable='api',
          name='mycobot_api', output='screen'
        )
    ])