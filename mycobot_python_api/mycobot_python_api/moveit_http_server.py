#!/usr/bin/env python3

import rclpy
from flask import Flask, request, jsonify
from geometry_msgs.msg import PoseStamped
from threading import Thread
from moveit.planning import MoveItPy

app = Flask(__name__)
moveit = None
planning_component = None

@app.route('/move_to_xyz', methods=['POST'])
def move_to_xyz():
    data = request.get_json()
    x, y, z = data.get("x"), data.get("y"), data.get("z")

    if x is None or y is None or z is None:
        return jsonify({"error": "Missing x/y/z"}), 400

    goal = PoseStamped()
    goal.header.frame_id = "base_link"
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.w = 1.0

    # IMPORTANT: specify the link name; adjust 'ee_link' to your actual tool frame
    pose_link = "gripper_base"  # e.g., "end_effector_link"

    try:
        planning_component.set_goal_state(pose_stamped_msg=goal, pose_link=pose_link)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = planning_component.plan()
    if result and result.trajectory:
        planning_component.execute()
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "planning_failed"}), 500

def main():
    global moveit, planning_component

    rclpy.init()
    moveit = MoveItPy(node_name="moveit_http_server")
    planning_component = moveit.get_planning_component("arm")

    # The MoveItPy node is already spinning internally—no need for your own thread

    app.run(host="0.0.0.0", port=5000)

if __name__ == '__main__':
    main()
