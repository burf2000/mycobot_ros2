#!/usr/bin/env python3
import math
import threading
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

# TF2 (optional but recommended for end-effector pose)
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

# --- minimal GUI (stdlib) ---
import tkinter as tk
from tkinter import ttk


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def euler_from_quaternion(x: float, y: float, z: float, w: float):
    """
    Convert quaternion -> roll, pitch, yaw (radians), ROS convention (XYZ -> RPY).
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # clamp to handle numerical drift
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class MyCobotState(Node):
    """
    ROS 2 node that:
      - Subscribes to /joint_states
      - Optionally looks up TF from base_frame -> ee_frame
    Frame names are configurable via parameters: base_frame, ee_frame.
    """
    def __init__(self):
        super().__init__('mycobot_state_gui')

        # Parameters for frame names (adjust to your myCobot setup)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'gripper_base')  # sometimes 'wrist_3_link', 'end_effector_link', etc.
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.ee_frame = self.get_parameter('ee_frame').get_parameter_value().string_value

        # Latest joint data
        self.joint_names: List[str] = []
        self.joint_pos: List[float] = []
        self.joint_vel: List[float] = []
        self.joint_eff: List[float] = []
        self.last_joint_stamp: Optional[float] = None

        # Latest EE pose (in world/base frame)
        self.ee_translation = (None, None, None)  # x, y, z (meters)
        self.ee_quat = (None, None, None, None)   # x, y, z, w

        # Subscribers & TF
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)

        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        # periodic timer to refresh TF (don’t block callback thread)
        self.create_timer(0.05, self._update_tf)  # 20 Hz

    def _on_joint_states(self, msg: JointState):
        self.joint_names = list(msg.name)
        self.joint_pos = list(msg.position) if msg.position else []
        self.joint_vel = list(msg.velocity) if msg.velocity else []
        self.joint_eff = list(msg.effort) if msg.effort else []
        self.last_joint_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _lookup_transform(self) -> Optional[TransformStamped]:
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=self.base_frame,
                source_frame=self.ee_frame,
                time=rclpy.time.Time())  # latest
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _update_tf(self):
        tf = self._lookup_transform()
        if tf is None:
            return
        t = tf.transform.translation
        q = tf.transform.rotation
        self.ee_translation = (t.x, t.y, t.z)
        self.ee_quat = (q.x, q.y, q.z, q.w)


class RosThread:
    """
    Runs the ROS node & executor on a background thread so Tkinter remains responsive.
    """
    def __init__(self):
        rclpy.init(args=None)
        self.node = MyCobotState()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self._running = False

    def start(self):
        self._running = True
        self.thread.start()

    def _spin(self):
        while self._running:
            self.executor.spin_once(timeout_sec=0.05)

    def stop(self):
        self._running = False
        time.sleep(0.1)
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        rclpy.shutdown()


class App(tk.Tk):
    def __init__(self, ros: RosThread):
        super().__init__()
        self.title("myCobot – Joint States & EE Pose (ROS 2)")
        self.geometry("760x520")
        self.ros = ros

        # --- Joint table ---
        joint_frame = ttk.LabelFrame(self, text="Joint States (/joint_states)")
        joint_frame.pack(fill="both", expand=False, padx=10, pady=10)

        columns = ("name", "pos_rad", "pos_deg", "vel", "eff")
        self.tree = ttk.Treeview(joint_frame, columns=columns, show="headings", height=12)
        for col, text, w in [
            ("name", "Joint", 180),
            ("pos_rad", "Position (rad)", 130),
            ("pos_deg", "Position (deg)", 130),
            ("vel", "Velocity (rad/s)", 140),
            ("eff", "Effort", 100),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=tk.CENTER)
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)

        # --- End-effector pose ---
        ee_frame = ttk.LabelFrame(self, text="End-Effector Pose ({} → {}) via TF".format(
            self.ros.node.base_frame, self.ros.node.ee_frame))
        ee_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.lbl_xyz = ttk.Label(ee_frame, text="XYZ (m): –")
        self.lbl_xyz.pack(anchor="w", padx=8, pady=4)

        self.lbl_quat = ttk.Label(ee_frame, text="Quaternion [x, y, z, w]: –")
        self.lbl_quat.pack(anchor="w", padx=8, pady=4)

        self.lbl_rpy = ttk.Label(ee_frame, text="RPY (deg): –")
        self.lbl_rpy.pack(anchor="w", padx=8, pady=4)

        self.status = ttk.Label(self, text="Waiting for /joint_states…", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

        # Start update loop
        self.after(100, self._refresh_gui)

        # Clean shutdown
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _refresh_gui(self):
        node = self.ros.node

        # Update joint table
        self.tree.delete(*self.tree.get_children())
        if node.joint_names and node.joint_pos:
            for i, name in enumerate(node.joint_names):
                pos = node.joint_pos[i] if i < len(node.joint_pos) else float("nan")
                vel = node.joint_vel[i] if i < len(node.joint_vel) else float("nan")
                eff = node.joint_eff[i] if i < len(node.joint_eff) else float("nan")
                self.tree.insert(
                    "", "end",
                    values=(
                        name,
                        f"{pos: .5f}",
                        f"{rad2deg(pos): .2f}",
                        f"{vel: .4f}" if not math.isnan(vel) else "–",
                        f"{eff: .4f}" if not math.isnan(eff) else "–",
                    ),
                )

        # Update EE pose (if TF available)
        tx, ty, tz = node.ee_translation
        qx, qy, qz, qw = node.ee_quat
        if None not in (tx, ty, tz):
            self.lbl_xyz.config(text=f"XYZ (m): [{tx: .4f}, {ty: .4f}, {tz: .4f}]")
        else:
            self.lbl_xyz.config(text="XYZ (m): – (TF not available)")

        if None not in (qx, qy, qz, qw):
            self.lbl_quat.config(text=f"Quaternion [x, y, z, w]: [{qx: .5f}, {qy: .5f}, {qz: .5f}, {qw: .5f}]")
            r, p, y = euler_from_quaternion(qx, qy, qz, qw)
            self.lbl_rpy.config(text=f"RPY (deg): [{rad2deg(r): .2f}, {rad2deg(p): .2f}, {rad2deg(y): .2f}]")
        else:
            self.lbl_quat.config(text="Quaternion [x, y, z, w]: –")
            self.lbl_rpy.config(text="RPY (deg): –")

        # Status line
        if node.last_joint_stamp is not None:
            age = time.time() - node.last_joint_stamp
            self.status.config(text=f"/joint_states age: {age: .2f}s   base_frame='{node.base_frame}'  ee_frame='{node.ee_frame}'")
        else:
            self.status.config(text="Waiting for /joint_states…")

        # schedule next refresh
        self.after(100, self._refresh_gui)

    def _on_close(self):
        try:
            self.ros.stop()
        finally:
            self.destroy()


def main():
    ros = RosThread()
    ros.start()
    app = App(ros)
    app.mainloop()


if __name__ == "__main__":
    main()