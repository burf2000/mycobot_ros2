#!/bin/bash
# Launch mycobot with MoveIt 2 for real hardware (no Gazebo) + USB cameras.
#
# Uses mock_components/GenericSystem so ros2_control controllers exist
# for MoveIt to talk to, then sync_plan forwards joint_states to the
# physical arm over serial.
#
# Requires: sudo apt-get install -y ros-jazzy-v4l2-camera

# Clear snap environment variables that conflict with RViz/GUI rendering
unset LOCPATH
unset GTK_PATH
unset GTK_IM_MODULE_FILE
unset GTK_EXE_PREFIX

# Isolate this robot to loopback. The myCobot pick is entirely single-host;
# without this, another ROS2 machine on the LAN (DDS domain 0) leaks its
# /joint_states (foreign joint names + empty msgs) into this graph, which
# floods pymoveit2 with "Joint states not available" and can feed sync_plan
# bad data. Loopback-only removes that cross-talk.
export ROS_LOCALHOST_ONLY=1

cleanup() {
  echo "Cleaning up..."
  sleep 5
  pkill -9 -f "ros2|robot_state_publisher|move_group|ros2_control_node|pymoveit_api|v4l2_camera"
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

# 7. USB Camera Nodes
sleep 2

echo "Launching overhead camera (USB 2.0 Camera @ /dev/video0)..."
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
  -r __ns:=/camera/overhead \
  -p video_device:="/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001-video-index0" \
  -p image_size:="[640,480]" \
  -p camera_frame_id:="overhead_camera" &

sleep 2

echo "Launching gripper camera (HD Camera @ /dev/video2)..."
ros2 run v4l2_camera v4l2_camera_node \
  --ros-args \
  -r __ns:=/camera/gripper \
  -p video_device:="/dev/v4l/by-id/usb-Suyin_HD_Camera_200910120001-video-index0" \
  -p image_size:="[640,480]" \
  -p camera_frame_id:="gripper_camera" &

echo ""
echo "Camera topics available:"
echo "  Overhead: /camera/overhead/image_raw"
echo "  Gripper:  /camera/gripper/image_raw"
echo ""

wait
