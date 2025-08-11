#!/bin/bash
# Single script to launch mycobot with full ros2_control support

cleanup() {
  echo "Cleaning up..."
  sleep 5
  pkill -9 -f "ros2||robot_state_publisher|move_group|gz_ros2_control|ros2_control_node|*pymoveit_api*"
}
trap 'cleanup' SIGINT SIGTERM

# 1️⃣ Start robot_state_publisher (URDF)
ros2 launch mycobot_description robot_state_publisher.launch.py \
  use_sim_time:=false sim:=false use_gazebo:=false use_rviz:=false jsp_gui:=false &

# 2️⃣ Launch controller manager (ros2_control_node) for hardware mode
ros2 launch mycobot_moveit_config ros2_control_node.launch.py \
  use_sim_time:=false sim:=false &

# 3️⃣ Load joint_state_broadcaster, arm_controller, gripper controller
sleep 3
ros2 launch mycobot_moveit_config load_ros2_controllers.launch.py \
  use_sim_time:=false sim:=false &

# 4️⃣ Start MoveIt move_group
sleep 3
ros2 launch mycobot_moveit_config move_group.launch.py &

# 5️⃣ Optional Gazebo camera control (will be a no-op if Gazebo isn't running)
echo "Adjusting camera position..."
gz service -s /gui/move_to/pose --reqtype gz.msgs.GUICamera \
  --reptype gz.msgs.Boolean --timeout 2000 \
  --req "pose: {position: {x: 1.36, y: -0.58, z: 0.95}, orientation: {x: -0.26, y: 0.1, z: 0.89, w: 0.35}}"

# 6️⃣ Start your Flask API
sleep 5
ros2 launch mycobot_pymoveit_api api.launch.py &

# 7️⃣ Launch your hardware sync node
sleep 5
ros2 run mycobot_pymoveit_api sync_plan_hardware &

wait
