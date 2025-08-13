#!/usr/bin/env python3
import rclpy
from threading import Thread
from flask import Flask, request, jsonify
import time
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import GripperInterface
from pymoveit2 import MoveIt2

app = Flask(__name__)
moveit2: MoveIt2 = None
gripper_interface: GripperInterface = None


@app.route("/gripper/<action>", methods=["POST"])
def gripper(action):
    if action not in ("open", "close"):
        return jsonify(error="Invalid action"), 400
    
    app.logger.info(f'Performing gripper action "{action}"')

    if "open" == action:
        gripper_interface.open()
        time.sleep(2)
        #gripper_interface.wait_until_executed()
    elif "close" == action:
        gripper_interface.close()
        time.sleep(2)
        #gripper_interface.wait_until_executed()

    return jsonify(status="success"), 200

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    pos = [data.get(k) for k in ('posX','posY','posZ')]
    quat = [data.get(k) for k in ('rotX','rotY','rotZ','rotW')]
    if None in pos or None in quat:
        return jsonify(error="Missing fields"), 400


    # Plan to the requested pose
    traj = moveit2.plan(position=pos, quat_xyzw=quat)
    if traj is None:
        app.logger.error("Planning failed or trajectory invalid")
        return jsonify(status="planning_failed"), 500

    # Execute the planned trajectory
    moveit2.execute(traj)
    exec_ok = moveit2.wait_until_executed()

    if not exec_ok:
        app.logger.error("Execution returned failure {exec_ok}")
        return jsonify(status="execution_failed"), 500

    # Compute FK to get current end-effector pose
    ee_pose_stamped = moveit2.compute_fk()
    if ee_pose_stamped is None:
        app.logger.error("FK service returned no result")
        return jsonify(status="no_fk_response"), 500

    p = ee_pose_stamped.pose.position
    o = ee_pose_stamped.pose.orientation
    return jsonify({
        "status": "success",
        "ee_position": {"x": p.x, "y": p.y, "z": p.z},
        "ee_orientation": {"x": o.x, "y": o.y, "z": o.z, "w": o.w}
    }), 200


def main():
    global moveit2, gripper_interface
    rclpy.init()
    node = rclpy.create_node('mycobot_pymoveit_api')
    cbg = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=['link1_to_link2','link2_to_link3','link3_to_link4','link4_to_link5','link5_to_link6','link6_to_link6_flange'],
        base_link_name='base_link',
        end_effector_name='gripper_base',
        group_name='arm',
        callback_group=cbg
    )

    gripper_interface = GripperInterface(
        node=node,
        gripper_joint_names=['gripper_controller'],
        open_gripper_joint_positions=[0.0],
        closed_gripper_joint_positions=[-0.6],
        # open_gripper_joint_positions=[0.1],
        # closed_gripper_joint_positions=[0.0],
        gripper_group_name='gripper',
        callback_group=cbg,
        gripper_command_action_name="gripper_action_controller/gripper_cmd",
    )


    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()

    app.run(host='0.0.0.0', port=8080)

    rclpy.shutdown()
    thread.join()

if __name__ == '__main__':
    main()