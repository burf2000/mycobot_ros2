# mycobot_ros2 #
![OS](https://img.shields.io/ubuntu/v/ubuntu-wallpapers/noble)
![ROS_2](https://img.shields.io/ros/v/jazzy/rclcpp)


## Overview
This is a fork of the great work by https://github.com/automaticaddison/mycobot_ros2

The idea of this repo is to extend what has been done by AutomaticAddison by learning MoveIt2 and the API's (Python) and to get it to work with real hardware.
 - First, I have added a MoveIt2 Python API, which has been a real challenge.  Latee I discovered that the C one works a lot better
 - Added a PyMoveIt API which seems to be more inline with the C API and a lot easier to get going
 -
rqt
rqt_graph
ros2 node info /node

## Steps to run

### Option A: Gazebo simulation + real hardware
Launches Gazebo, MoveIt, RViz, the HTTP API, and `sync_plan` (which mirrors
Gazebo joint states to the physical arm over USB serial).

```bash
$ bash ~/ros2_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_gazebo_and_moveit.sh
```

### Option B: Real hardware only (no Gazebo)
Launches a standalone ros2_control controller manager with mock hardware,
MoveIt, RViz, the HTTP API, and `sync_plan` (which forwards planned joint
states to the physical arm over USB serial). No Gazebo window is opened.

```bash
$ bash ~/ros2_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_moveit_api.sh
```

**Requirements for Option B:**
- myCobot 280 M5 connected via USB (`/dev/ttyUSB0`)
- Atom firmware >= 6.5, pymycobot >= 3.6.1

### Option C: Real hardware + USB cameras (no Gazebo)
Same as Option B but also launches overhead and gripper USB cameras as ROS2
topics via `v4l2_camera`.

```bash
$ bash ~/ros2_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_moveit_api_camera.sh
```

**Additional requirements for Option C:**
- `sudo apt-get install -y ros-jazzy-v4l2-camera`
- Overhead camera at `/dev/video0`, gripper camera at `/dev/video2`
- Camera topics: `/camera/overhead/image_raw`, `/camera/gripper/image_raw`


### Pick-and-place: two-camera red-brick pick

`pick_red_brick_ros` is the working pure-ROS2 two-camera pick: the **overhead
camera** gives a coarse brick position, the **gripper (wrist) camera** fine-aligns,
then the arm descends and grips. No HTTP API - it reads the camera topics
directly and drives the arm via MoveIt + `sync_plan`.

```bash
# 1. Bring up arm + MoveIt + cameras + sync_plan (Option C)
$ bash ~/ros2_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_moveit_api_camera.sh

# 2. Place the red brick inside the black workspace circle, then:
$ ros2 run mycobot_pymoveit_api pick_red_brick_ros
```

**How it works**
1. Overhead camera detects the red brick and maps its pixel to a robot XY using
   the workspace-circle calibration in `vision_circle_calib.json`.
2. The arm moves to that coarse XY; the gripper camera fine-aligns on the brick.
3. The arm descends, closes the gripper, retracts, and verifies with the overhead camera.

**Vision calibration - `vision_circle_calib.json`** maps the workspace-circle
centre/radius (pixels) to metres. **If the overhead camera is moved it MUST be
re-fitted**, or the coarse move will be off. The circle centre often sits off the
top of the frame, so fit a circle to the *visible arc* (least-squares / RANSAC) -
a full-circle Hough detect clamps the centre to the frame edge and is wrong.

**Notes**
- The bringup script exports `ROS_LOCALHOST_ONLY=1` so another ROS2 machine on the
  LAN cannot leak its `/joint_states` into this graph (foreign joint names flood
  pymoveit2 and can feed `sync_plan` bad data).
- `move_group` is launched headless in the bringup; if you run it separately over
  SSH use `use_rviz:=false` (RViz cannot open a display and its exit otherwise
  shuts down `move_group`).
- Set `PICK_DEBUG_DIR=/tmp/dbg` to dump gripper-camera frames + red masks per
  fine-align iteration when debugging detection.

The older `pick_red_brick_v2` (HTTP API + `cv2.VideoCapture`) is kept for reference;
`pick_red_brick_ros` is the maintained two-camera path.

### Common commands

```bash
# run visual script
$ ros2 run mycobot_pymoveit_api display

# Home arm
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":0.06,"posY":0.079,"posZ":0.41,"rotX":0.03,"rotY":-0.382,"rotZ":0.001,"rotW":1}'

# Arm movement
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":0.1133,"posY":0.0133,"posZ":0.3,"rotX":1,"rotY":0,"rotZ":0,"rotW":1}'

# arm parallel to ground
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":0.029,"posY":0.186,"posZ":0.1285,"rotX":0.004,"rotY":0.042,"rotZ":0.000,"rotW":1}'

# pickup Y (forward)
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":-0.002,"posY":0.24,"posZ":0.08,"rotX":0.020,"rotY":-0.659,"rotZ":0.613,"rotW":0.000}'

#pick up X (Right)
curl -X POST http://localhost:8080/move -H "Content-Type: application/json" -d '{
  "posX": 0.24, "posY": 0.00, "posZ": 0.08,
  "rotX": -0.491, "rotY": -0.503, "rotZ": 0.520, "rotW": 0.483
}'

#pick up - X (Left)
curl -X POST http://localhost:8080/move -H "Content-Type: application/json" -d '{
  "posX": -0.24, "posY": 0.00, "posZ": 0.08,
  "rotX": -0.491, "rotY": -0.503, "rotZ": 0.520, "rotW": 0.483
}'

# Gripper Movement
$ curl -X POST http://localhost:8080/gripper/close   -H "Content-Type: application/json" 
```


## Overview (from fork)
This repository contains ROS 2 packages for simulating and controlling the myCobot robotic arm using ROS 2 Control and MoveIt 2. It provides support for Gazebo simulation and visualization in RViz. Gazebo simulation also includes simulated 3D point cloud data from the depth camera (RGBD) sensor plugin for vision.

![Gazebo Pick and Place Task Simulation](https://automaticaddison.com/wp-content/uploads/2024/12/pick-place-gazebo-800-fast.gif)

![Pick and Place with Perception](https://automaticaddison.com/wp-content/uploads/2024/12/pick-place-demo-rviz-800-fast.gif)

## Features
- Gazebo simulation of the myCobot robotic arm
- RViz visualization for robot state and motion planning
- MoveIt 2 integration for motion planning and control
- Pick and place task implementation using the MoveIt Task Constructor (MTC)
- 3D perception and object segmentation using point cloud data
- Automatic planning scene generation from perceived objects
- Support for various primitive shapes (cylinders, boxes) in object detection
- Integration with tf2 for coordinate transformations
- Custom service for retrieving planning scene information
- Advanced object detection algorithms:
  - RANSAC (Random Sample Consensus) for robust model fitting
  - Hough transform for shape recognition
- CPU-compatible implementation, no GPU required. 
- Real-time perception and planning capabilities for responsive robot operation

![Setup Planning Scene](https://automaticaddison.com/wp-content/uploads/2024/12/creating-planning-scene-800.gif)

## Getting Started
For a complete step-by-step walkthrough on how to build this repository from scratch, start with this tutorial:
[Create and Visualize a Robotic Arm with URDF](https://automaticaddison.com/create-and-visualize-a-robotic-arm-with-urdf-ros-2-jazzy/)

This guide will take you through the entire process of setting up and understanding the mycobot_ros2 project.

![3D Point Cloud RViz](https://automaticaddison.com/wp-content/uploads/2024/12/800_3d-point-cloud.jpg)

![mycobot280_rviz](./mycobot_description/urdf/mycobot280_rviz.png)