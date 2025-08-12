# mycobot_ros2 #
![OS](https://img.shields.io/ubuntu/v/ubuntu-wallpapers/noble)
![ROS_2](https://img.shields.io/ros/v/jazzy/rclcpp)


## Overview
This is a fork of the great work by https://github.com/automaticaddison/mycobot_ros2

The idea of this repo is to extend what has been done by AutomaticAddison by learning MoveIt2 and the API's (Python) and to get it to work with real hardware.
 - First, I have added a MoveIt2 Python API, which has been a real challenge.  Latee I discovered that the C one works a lot better
 - Added a PyMoveIt API which seems to be more inline with the C API and a lot easier to get going
 - Last was to get it working with a real arm


## Steps to run
```bash
# Launches everything (RVIZ, MoveIt, Gazebo, API, USB connection to real arm)
$ bash ~/ros2_ws/src/mycobot_ros2/mycobot_bringup/scripts/mycobot_280_gazebo_and_moveit.sh

# Arm movement
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":0.1133,"posY":0.0133,"posZ":0.3,"rotX":1,"rotY":0,"rotZ":0,"rotW":1}'

# arm parallel to ground
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":0.029,"posY":0.186,"posZ":0.1285,"rotX":0.004,"rotY":0.042,"rotZ":0.000,"rotW":1}'

# pickup
$ curl -X POST http://localhost:8080/move   -H "Content-Type: application/json"   -d '{"posX":-0.002,"posY":-0.24,"posZ":0.06,"rotX":0.009,"rotY":-0.699,"rotZ":0.713,"rotW":0.027}'


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