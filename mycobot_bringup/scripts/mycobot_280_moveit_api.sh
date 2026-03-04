#!/bin/bash
# Launch mycobot with MoveIt 2 for real hardware (no Gazebo).
#
# Uses mock_components/GenericSystem so ros2_control controllers exist
# for MoveIt to talk to, then sync_plan forwards joint_states to the
# physical arm over serial.

# Clear snap environment variables that conflict with RViz/GUI rendering
unset LOCPATH
unset GTK_PATH
unset GTK_IM_MODULE_FILE
unset GTK_EXE_PREFIX

cleanup() {
  echo "Cleaning up..."
  sleep 5
  pkill -9 -f "ros2|robot_state_publisher|move_group|ros2_control_node|pymoveit_api"
}
trap 'cleanup' SIGINT SIGTERM

# 1. Robot state publisher (URDF, no Gazebo)
ros2 launch mycobot_description robot_state_publisher.launch.py \
  use_sim_time:=false use_gazebo:=false use_rviz:=false jsp_gui:=false &

# 2. Standalone controller manager (mock hardware)
sleep 2
ros2 launch mycobot_moveit_config ros2_control_node.launch.py \
  use_sim_time:=false &

# 3. Load controllers (joint_state_broadcaster -> arm_controller -> gripper)
sleep 5
ros2 launch mycobot_moveit_config load_ros2_controllers.launch.py &

# 4. MoveIt move_group
sleep 8
ros2 launch mycobot_moveit_config move_group.launch.py \
  use_sim_time:=false &

# 5. Flask HTTP API
sleep 5
ros2 launch mycobot_pymoveit_api api.launch.py &

# 6. Forward joint_states to real hardware over serial
sleep 5
ros2 run mycobot_pymoveit_api sync_plan &

wait
