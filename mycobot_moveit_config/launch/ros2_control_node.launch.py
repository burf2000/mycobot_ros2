#!/usr/bin/env python3
"""
Launch a standalone ros2_control controller manager (no Gazebo).

Uses the robot URDF (with mock_components/GenericSystem hardware) and
the ros2_controllers.yaml config so that MoveIt has real controller
action servers (arm_controller, gripper_action_controller) to talk to.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')

    # --- URDF via xacro (use_gazebo:=false → mock hardware) ---
    urdf_path = PathJoinSubstitution([
        FindPackageShare('mycobot_description'), 'urdf', 'robots',
        'mycobot_280.urdf.xacro'
    ])

    robot_description_content = ParameterValue(
        Command([
            'xacro ', urdf_path,
            ' robot_name:=mycobot_280',
            ' prefix:=',
            ' add_world:=true',
            ' base_link:=base_link',
            ' base_type:=g_shape',
            ' flange_link:=link6_flange',
            ' gripper_type:=adaptive_gripper',
            ' use_camera:=false',
            ' use_gazebo:=false',
            ' use_gripper:=true',
        ]),
        value_type=str,
    )

    # --- Controller config ---
    controllers_yaml = PathJoinSubstitution([
        FindPackageShare('mycobot_moveit_config'), 'config',
        'ros2_controllers.yaml'
    ])

    # --- Controller manager node ---
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description_content},
            controllers_yaml,
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation clock if true'),
        ros2_control_node,
    ])
