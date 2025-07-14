#!/usr/bin/env python3
import rclpy

from flask import Flask, request, jsonify
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit.core.robot_state import RobotState
from moveit.core.kinematic_constraints import construct_joint_constraint
# moveit python library
from moveit.planning import (
    MoveItPy,
)

app = Flask(__name__)

moveit: MoveItPy = None
planning_component = None




@app.route("/gripper/<action>", methods=["POST"])
def gripper(action):
    if action not in ("open", "close"):
        return jsonify(error="Invalid action"), 400



    # # Create RobotState and set gripper joint(s) value
    robot_model = moveit.get_robot_model()
    st = RobotState(robot_model)

    if action == "open":
        planning_component_gripper.set_goal_state(configuration_name="open")
    else:
        planning_component_gripper.set_goal_state(configuration_name="closed")



    plan_result = planning_component_gripper.plan()
    if plan_result and plan_result.trajectory:
        moveit.execute(plan_result.trajectory, controllers=[])
        return jsonify(status="success"), 200

    return jsonify(status="planning_failed"), 500


@app.route("/move_to_xyz", methods=["POST"])
def move_to_xyz():
    data = request.get_json()
    x, y, z = (data.get("x"), data.get("y"), data.get("z"))
    if None in (x, y, z):
        return jsonify({"error": "Missing x/y/z"}), 400

    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "base_link"
    pose_goal.pose.orientation.w = 1.0
    pose_goal.pose.position.x = x
    pose_goal.pose.position.y = y
    pose_goal.pose.position.z = z
    planning_component.set_goal_state(pose_stamped_msg=pose_goal, pose_link="gripper_base")

    plan_result = planning_component.plan()

    robot_trajectory = plan_result.trajectory

    if plan_result and robot_trajectory:
        moveit.execute(robot_trajectory, controllers=[])
        return jsonify({"status": "success"}), 200
    return jsonify({"status": "planning_failed"}), 500

def main():
    global moveit, planning_component, planning_component_gripper
    rclpy.init()
    moveit = MoveItPy(node_name="moveit_http_server")   # loads MoveIt params
    planning_component = moveit.get_planning_component("arm")
    planning_component_gripper = moveit.get_planning_component("gripper")
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()