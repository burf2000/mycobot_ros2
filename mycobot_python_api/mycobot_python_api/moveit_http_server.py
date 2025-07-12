#!/usr/bin/env python3
import rclpy
from flask import Flask, request, jsonify
from geometry_msgs.msg import PoseStamped

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)

app = Flask(__name__)

moveit: MoveItPy = None
planning_component = None

@app.route("/move_to_xyz", methods=["POST"])
def move_to_xyz():
    data = request.get_json()
    x, y, z = (data.get("x"), data.get("y"), data.get("z"))
    if None in (x, y, z):
        return jsonify({"error": "Missing x/y/z"}), 400

    goal = PoseStamped()
    goal.header.frame_id = "base_link"
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = z
    goal.pose.orientation.w = 1.0

    planning_component.set_goal_state(pose_stamped_msg=goal,
                                      pose_link="gripper_base")

    # params = PlanRequestParameters(moveit, "ompl")  # MoveItPy instance + pipeline
    # params.planning_time = 5.0                      # 5 s planning timeout

    # initialise multi-pipeline plan request parameters
    multi_plan_parameters = MultiPipelinePlanRequestParameters(
            moveit, ["ompl_rrtc", "pilz_lin", "chomp", "ompl_rrt_star"]
    )

    plan_result = planning_component.plan(
        multi_plan_parameters=multi_plan_parameters
    )

    # 🚨  POSitional call – no keyword
    #plan_result = planning_component.plan()

    if plan_result and plan_result.trajectory:
        planning_component.execute()
        return jsonify({"status": "success"}), 200
    return jsonify({"status": "planning_failed"}), 500


def main():
    global moveit, planning_component
    rclpy.init()
    moveit = MoveItPy(node_name="moveit_http_server")   # loads MoveIt params
    planning_component = moveit.get_planning_component("arm")
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
