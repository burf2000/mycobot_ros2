#!/bin/bash
# Launch mycobot arm hardware + USB cameras WITHOUT MoveIt.
#
# Use this when MoveIt (move_group) runs on a separate machine.
# This script starts ros2_control, controllers, sync_plan (serial
# forwarding to the physical arm), and the USB camera nodes.
# MoveIt on the remote machine talks to the controllers over DDS.
#
# Requires: sudo apt-get install -y ros-jazzy-v4l2-camera

# Clear snap environment variables that conflict with RViz/GUI rendering
unset LOCPATH
unset GTK_PATH
unset GTK_IM_MODULE_FILE
unset GTK_EXE_PREFIX

cleanup() {
  echo "Cleaning up..."
  sleep 5
  pkill -9 -f "ros2|robot_state_publisher|ros2_control_node|v4l2_camera"
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

# 4. Forward joint_states to real hardware over serial
sleep 8
ros2 run mycobot_pymoveit_api sync_plan &

# 5. USB Camera Nodes
sleep 2

echo "Launching overhead camera (USB 2.0 Camera @ /dev/video0)..."
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
  -r __ns:=/camera/overhead \
  -p video_device:="/dev/video0" \
  -p image_size:="[640,480]" \
  -p camera_frame_id:="overhead_camera" &

sleep 2

echo "Launching gripper camera (HD Camera @ /dev/video2)..."
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
  -r __ns:=/camera/gripper \
  -p video_device:="/dev/video2" \
  -p image_size:="[640,480]" \
  -p camera_frame_id:="gripper_camera" &

echo ""
echo "Arm hardware + cameras ready (no MoveIt)."
echo "Run MoveIt (move_group) on the remote machine."
echo ""
echo "Camera topics available:"
echo "  Overhead: /camera/overhead/image_raw"
echo "  Gripper:  /camera/gripper/image_raw"
echo ""

wait
