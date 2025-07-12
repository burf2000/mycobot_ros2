import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("mycobot", package_name="mycobot_moveit_config")
        .robot_description(file_path=os.path.join(
            get_package_share_directory("mycobot_description"),
            "urdf",
            "robots",
            "mycobot_280.urdf.xacro"
        ))
        .robot_description_semantic(file_path="config/mycobot_280/mycobot_280.srdf")
        .planning_pipelines()
        .joint_limits(file_path="config/joint_limits.yaml")
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .trajectory_execution(file_path="config/mycobot_280/moveit_controllers.yaml")
        # .moveit_cpp(file_path="config/planning_python_api.yaml")  # this must exist
        .moveit_cpp(
            file_path=get_package_share_directory("mycobot_moveit_config")
            + "/config/planning_python_api.yaml"
        )
        .to_moveit_configs()
    )


    api_server = Node(
        name="moveit_py",
        package="mycobot_python_api",
        executable="moveit_http_server",
        output="both",
        parameters=[moveit_config.to_dict(), {"use_sim_time": True}]
    )

    return LaunchDescription([
        api_server
    ])
