#!/usr/bin/env python3
"""
Two-camera visual-servoing brick pick — pure ROS2 version.

Uses:
  - pymoveit2 directly (no Flask HTTP API)
  - ROS2 image topic subscribers (no cv2.VideoCapture)

Requires the camera launch script to publish:
  /camera/overhead/image_raw
  /camera/gripper/image_raw

Phases:
  1. Detect brick with overhead camera, move arm ONCE to approach height
  2. Fine alignment with gripper-mounted camera (live feed)
  3. Pick: descend -> grip -> retract
  4. Verify with overhead camera
"""

import os, time, math, json, signal
from typing import Tuple, Optional
from threading import Thread, Lock

import cv2
import numpy as np

# Headless support: if OpenCV has no GUI backend (headless build / no display),
# neuter the GUI calls so the pick runs autonomously without a preview window.
HEADLESS = False
try:
    cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL); cv2.destroyWindow("__probe__")
except Exception:
    HEADLESS = True
    cv2.namedWindow = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.destroyWindow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.getWindowProperty = lambda *a, **k: 1.0   # report "window visible" so should_quit() doesn't false-fire
    print("[GUI] OpenCV has no display backend - running HEADLESS (auto-start on brick detection).")

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
# Note: ReentrantCallbackGroup used by MoveIt, cameras get their own node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rclpy.action import ActionClient
from control_msgs.action import GripperCommand

from pymoveit2 import MoveIt2

# ─── Graceful Ctrl-C handling ───
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    print("\n[SIGINT] Shutting down...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def interruptible_sleep(seconds):
    """Sleep in small increments so Ctrl-C / window close is responsive."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if _shutdown:
            return
        time.sleep(min(0.1, end - time.monotonic()))


# ─── Poses ───
HIDE_POSE = {
    "posX": 0.06, "posY": 0.005, "posZ": 0.444,
    "rotX": 0.352, "rotY": -0.354, "rotZ": -0.143, "rotW": 0.854,
}

# ─── Heights (m) ───
Z_APPROACH = 0.12
Z_PRE_PICK = 0.08

# ─── Safe tilt quaternion (will be yaw-rotated) ───
SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

# ─── Workspace limits ───
REACH_RADIUS_M = 0.28
KEEP_OUT_R_M   = 0.05
ALLOW_NEG_Y    = False

# ─── Yaw ───
GRIPPER_WORLD_YAW_DEG = -90.0

# ─── Calibration trims (overwritten by load_cfg) ───
X_BIAS  = 0.0
Y_BIAS  = 0.0
X_SCALE = 1.0
Y_SCALE = 1.0
X_ORIGIN_SHIFT_M = 0.0
Y_ORIGIN_SHIFT_M = 0.0

# ─── Detection parameters ───
DETECT_RETRIES    = 3

# ─── Gripper camera fine-alignment parameters ───
GRIPPER_CAM_GAIN_X        =  0.00005
GRIPPER_CAM_GAIN_Y        = -0.00005
GRIPPER_CAM_TARGET_U_FRAC = 0.5
GRIPPER_CAM_TARGET_V_FRAC = 0.85
GRIPPER_CAM_CONVERGE_PX   = 15
GRIPPER_CAM_MAX_ITERS     = 12
GRIPPER_CAM_SERVO_GAIN    = 0.4

# How long to wait after a MoveIt move for the real arm to finish (seconds).
# MoveIt mock hardware completes instantly; sync_plan needs time to physically
# move the arm.
ARM_SETTLE_SECS = 2.5        # Large moves (go_hide, coarse positioning)
GRIPPER_SETTLE_SECS = 1.0    # Small gripper-servo corrections

# ─── ROS2 camera topics ───
OVERHEAD_TOPIC = "/camera/overhead/image_raw"
GRIPPER_TOPIC  = "/camera/gripper/image_raw"

# ─── Window names ───
WIN_OVERHEAD = "Overhead Camera"
WIN_GRIPPER  = "Gripper Camera"

# ─── Gripper joint states ───
GRIPPER_OPEN   = [0.0]
GRIPPER_CLOSED = [-0.50]

# ─── Optional debug frame dump (set PICK_DEBUG_DIR to enable) ───
DEBUG_DIR = os.environ.get("PICK_DEBUG_DIR")
if DEBUG_DIR:
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        print(f"[DEBUG] Dumping gripper-cam frames to {DEBUG_DIR}")
    except Exception:
        DEBUG_DIR = None

# ─── Calibration file ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(_SCRIPT_DIR, "..", "..", "vision_circle_calib.json")
CAL = {
    "cx": None, "cy": None, "r_px": None, "m_per_px": None,
    "x_origin_shift_m": 0.0, "y_origin_shift_m": 0.0,
    "x_bias": 0.0, "y_bias": 0.0, "x_scale": 1.0, "y_scale": 1.0,
}


# ═══════════════════════  ROS2 CAMERA SUBSCRIBER  ═══════════════════════

class CameraSubscriber:
    """Thread-safe wrapper that subscribes to a ROS2 image topic.

    Uses its own ReentrantCallbackGroup so camera callbacks are never
    blocked by MoveIt or other node callbacks.
    """

    def __init__(self, node: Node, topic: str):
        self._bridge = CvBridge()
        self._lock = Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq: int = 0
        self._topic = topic

        qos = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self._sub = node.create_subscription(
            Image, topic, self._callback, qos)
        print(f"[CAM-SUB] Subscribed to {topic} (RELIABLE)")

    def _callback(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
        except Exception as e:
            print(f"[CAM-SUB] cv_bridge error on {self._topic}: {e}")
            return
        with self._lock:
            self._frame = frame
            self._seq += 1
            if self._seq == 1:
                print(f"[CAM-SUB] First frame received from {self._topic}!")

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame (or None if nothing received yet)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def wait_for_frame(self, timeout=5.0) -> Optional[np.ndarray]:
        """Block until a frame arrives or timeout."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            f = self.get_frame()
            if f is not None:
                return f
            time.sleep(0.05)
        return None

    def wait_for_new_frame(self, old_seq: int, timeout=5.0) -> Optional[np.ndarray]:
        """Block until a frame with seq > old_seq arrives."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if _shutdown:
                return None
            with self._lock:
                if self._frame is not None and self._seq > old_seq:
                    return self._frame.copy()
            time.sleep(0.05)
        return None


# ═══════════════════════  UTILITIES  ═══════════════════════

def load_cfg():
    global CAL, X_ORIGIN_SHIFT_M, Y_ORIGIN_SHIFT_M
    global X_BIAS, Y_BIAS, X_SCALE, Y_SCALE
    try:
        with open(CFG_FILE, "r") as f:
            d = json.load(f)
        CAL.update(d)
        X_ORIGIN_SHIFT_M = float(CAL.get("x_origin_shift_m", 0.0))
        Y_ORIGIN_SHIFT_M = float(CAL.get("y_origin_shift_m", 0.0))
        X_BIAS  = float(CAL.get("x_bias", 0.0))
        Y_BIAS  = float(CAL.get("y_bias", 0.0))
        X_SCALE = float(CAL.get("x_scale", 1.0))
        Y_SCALE = float(CAL.get("y_scale", 1.0))
        print(f"[CFG] Loaded {CFG_FILE}: center=({CAL['cx']},{CAL['cy']}), "
              f"r_px={CAL['r_px']}, m/px={CAL['m_per_px']:.6f}")
    except Exception:
        print("[CFG] No saved calibration; will auto-detect circle.")


def yaw_about_z(q, yaw_rad):
    x1, y1, z1, w1 = q
    cz, sz = math.cos(yaw_rad / 2.0), math.sin(yaw_rad / 2.0)
    x2, y2, z2, w2 = 0.0, 0.0, sz, cz
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1
    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    return (x, y, z, w)


# ─── Robot moves (direct MoveIt2) ───

def move_and_settle(moveit2: MoveIt2, x, y, z, q):
    """Plan, execute, then wait for the physical arm to reach the target."""
    pos = [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
    quat = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    print(f"[MOVE] -> pos={pos} quat={quat}")
    traj = moveit2.plan(position=pos, quat_xyzw=quat)
    if traj is None:
        print("[MOVE] Planning failed!")
        return False
    moveit2.execute(traj)
    ok = moveit2.wait_until_executed()
    if not ok:
        print(f"[MOVE] Execution failed: {ok}")
        return False
    # MoveIt mock hardware reports done instantly; real arm needs time.
    interruptible_sleep(ARM_SETTLE_SECS)
    return True


def move_pose_and_settle(moveit2: MoveIt2, pose):
    q = (pose["rotX"], pose["rotY"], pose["rotZ"], pose["rotW"])
    return move_and_settle(moveit2, pose["posX"], pose["posY"], pose["posZ"], q)


class GripperDriver:
    """Drives the real gripper via its GripperActionController.

    The gripper is a position_controllers/GripperActionController exposing the
    control_msgs/action/GripperCommand action at /gripper_action_controller/gripper_cmd.
    Driving it through MoveIt (plan/execute on the "gripper" group) hangs, so we
    talk to the action server directly. Position is the joint target
    (0.0 = open, GRIPPER_CLOSED[0] = closed); the node's background executor
    services the futures, so we just poll them.
    """

    ACTION = "/gripper_action_controller/gripper_cmd"

    def __init__(self, node: Node):
        self._node = node
        self._client = ActionClient(node, GripperCommand, self.ACTION)

    def _send(self, position: float, max_effort: float = 50.0, timeout: float = 8.0) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            print(f"[GRIPPER] action server {self.ACTION} not available")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        deadline = time.time() + timeout
        send_fut = self._client.send_goal_async(goal)
        while not send_fut.done() and time.time() < deadline and not _shutdown:
            time.sleep(0.05)
        if not send_fut.done():
            print("[GRIPPER] goal send timed out")
            return False
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            print("[GRIPPER] goal rejected")
            return False
        res_fut = gh.get_result_async()
        while not res_fut.done() and time.time() < deadline and not _shutdown:
            time.sleep(0.05)
        if not res_fut.done():
            print("[GRIPPER] result timed out (goal sent, motion may still finish)")
            return False
        return True

    def open(self) -> bool:
        print("[GRIPPER] Opening...")
        ok = self._send(GRIPPER_OPEN[0])
        print("[GRIPPER] Open done" if ok else "[GRIPPER] Open failed")
        return ok

    def close(self) -> bool:
        print("[GRIPPER] Closing...")
        ok = self._send(GRIPPER_CLOSED[0])
        print("[GRIPPER] Close done" if ok else "[GRIPPER] Close failed")
        return ok


def gripper_open(gripper: "GripperDriver"):
    gripper.open()


def gripper_close(gripper: "GripperDriver"):
    gripper.close()


def go_hide(moveit2: MoveIt2):
    print("[HIDE] Moving arm out of camera view")
    move_pose_and_settle(moveit2, HIDE_POSE)


# ═══════════════════════  DETECTION  ═══════════════════════

def detect_circle(img_bgr) -> Optional[Tuple[float, float, float]]:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 5)
    edges = cv2.Canny(g, 50, 120)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
        param1=120, param2=40, minRadius=150, maxRadius=900,
    )
    if circles is not None:
        c = circles[0][0]
        return float(c[0]), float(c[1]), float(c[2])
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    if r > 150:
        return float(x), float(y), float(r)
    return None


def detect_red_centroid(img_bgr) -> Optional[Tuple[float, float]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 80), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 90, 80), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 60:
        return None
    M = cv2.moments(c)
    m00 = M["m00"]
    if m00 == 0:
        return None
    return float(M["m10"] / m00), float(M["m01"] / m00)


def img_to_robot(u, v) -> Optional[Tuple[float, float]]:
    cx, cy, r = CAL["cx"], CAL["cy"], CAL["r_px"]
    m_per_px = CAL["m_per_px"]
    if None in (cx, cy, r, m_per_px):
        return None
    X = ((cx - u) * m_per_px) * X_SCALE + X_BIAS + X_ORIGIN_SHIFT_M
    Y = ((v  - cy) * m_per_px) * Y_SCALE + Y_BIAS + Y_ORIGIN_SHIFT_M
    return X, Y


# ═══════════════════════  CAMERA / DISPLAY  ═══════════════════════

def draw_overlay(img, status_text="", brick_uv=None, robot_xy=None):
    h, w = img.shape[:2]
    cx, cy, r = CAL["cx"], CAL["cy"], CAL["r_px"]
    if None not in (cx, cy, r):
        cv2.circle(img, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
        cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)
    if brick_uv is not None:
        u, v = brick_uv
        cv2.circle(img, (int(u), int(v)), 10, (0, 165, 255), -1)
        cv2.circle(img, (int(u), int(v)), 12, (0, 0, 255), 2)
        if robot_xy is not None:
            rx, ry = robot_xy
            cv2.putText(img, f"({rx:+.3f},{ry:+.3f})m",
                        (int(u) + 15, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if status_text:
        cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(img, status_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if CAL["m_per_px"] is not None:
        cv2.putText(img, f"m/px={CAL['m_per_px']:.6f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def should_quit(win_name=None):
    if _shutdown:
        return True
    if win_name is not None:
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
    return False


def check_key_quit(key):
    global _shutdown
    if key in (ord('q'), ord('x'), ord('e'), 27):
        _shutdown = True
        return True
    return False


def show_frame(img, win_name, status_text="", brick_uv=None, robot_xy=None, wait_ms=1):
    disp = img.copy()
    draw_overlay(disp, status_text, brick_uv, robot_xy)
    cv2.imshow(win_name, disp)
    key = cv2.waitKey(wait_ms) & 0xFF
    check_key_quit(key)
    return key


def detect_brick_world(cam: CameraSubscriber, show=True, status="Detecting..."):
    """Grab the latest frame and detect the brick.  Returns (x, y, u, v, frame) or None."""
    frame = cam.get_frame()
    if frame is None:
        print("[WARN] No frame from camera")
        return None
    if CAL["m_per_px"] is None:
        circ = detect_circle(frame)
        if circ is None:
            if show:
                show_frame(frame, WIN_OVERHEAD, "WARN: no workspace circle")
            print("[WARN] Cannot detect workspace circle")
            return None
        CAL["cx"], CAL["cy"], CAL["r_px"] = circ
        CAL["m_per_px"] = REACH_RADIUS_M / circ[2]
        print(f"[CAL] Auto-detected circle: center=({circ[0]:.1f},{circ[1]:.1f}), "
              f"r={circ[2]:.1f}, m/px={CAL['m_per_px']:.6f}")
    uv = detect_red_centroid(frame)
    if uv is None:
        if show:
            show_frame(frame, WIN_OVERHEAD, f"{status} - no brick found")
        return None
    u, v = uv
    xy = img_to_robot(u, v)
    if xy is None:
        if show:
            show_frame(frame, WIN_OVERHEAD, f"{status} - mapping failed", brick_uv=(u, v))
        return None
    if show:
        show_frame(frame, WIN_OVERHEAD, status, brick_uv=(u, v), robot_xy=xy)
    return xy[0], xy[1], u, v, frame


def is_safe(x, y) -> bool:
    r = math.hypot(x, y)
    if r < KEEP_OUT_R_M:
        print(f"[GUARD] Inside keep-out zone (r={r:.3f} m)")
        return False
    if r > REACH_RADIUS_M:
        print(f"[GUARD] Outside reach (r={r:.3f} m)")
        return False
    if (not ALLOW_NEG_Y) and y < 0:
        print(f"[GUARD] Negative Y={y:.3f} not allowed")
        return False
    return True


# ═══════════════════════  GRIPPER CAMERA FINE ALIGNMENT  ═══════════════════════

def draw_gripper_overlay(img, brick_uv=None, target_uv=None, status_text=""):
    h, w = img.shape[:2]
    if target_uv is not None:
        tu, tv = int(target_uv[0]), int(target_uv[1])
        cv2.drawMarker(img, (tu, tv), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(img, (tu, tv), 20, (0, 255, 0), 1)
    if brick_uv is not None:
        bu, bv = int(brick_uv[0]), int(brick_uv[1])
        cv2.circle(img, (bu, bv), 8, (0, 165, 255), -1)
        cv2.circle(img, (bu, bv), 10, (0, 0, 255), 2)
        if target_uv is not None:
            tu, tv = int(target_uv[0]), int(target_uv[1])
            cv2.line(img, (bu, bv), (tu, tv), (0, 0, 255), 2)
            err = math.hypot(brick_uv[0] - target_uv[0], brick_uv[1] - target_uv[1])
            cv2.putText(img, f"err={err:.1f}px", (bu + 12, bv - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if status_text:
        cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(img, status_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def gripper_cam_fine_align(moveit2: MoveIt2, cam_gripper: CameraSubscriber,
                           current_x, current_y, pick_q):
    """
    Phase 2: Fine alignment using the gripper-mounted camera.

    Gradually descends from Z_APPROACH toward Z_PRE_PICK while centering
    the brick in the gripper camera.  Shows a continuously updating live
    feed throughout.

    Returns (final_x, final_y, final_z, converged).
    """
    print("\n=== Phase 2: Gripper camera fine alignment ===")
    total = cam_gripper.seq
    print(f"[DIAG] Gripper camera frames received so far: {total}")
    if total == 0:
        print(f"[DIAG] Waiting up to 10s for first gripper frame...")
        frame = cam_gripper.wait_for_frame(timeout=10.0)
        if frame is None:
            print(f"[ERROR] Cannot receive frames from {GRIPPER_TOPIC}.")
            return current_x, current_y, Z_APPROACH, False
        print(f"[DIAG] Got first gripper frame!")

    cv2.namedWindow(WIN_GRIPPER, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    x, y = current_x, current_y
    z = Z_APPROACH
    # Descend evenly from Z_APPROACH to Z_PRE_PICK over all iterations
    z_step = (Z_APPROACH - Z_PRE_PICK) / GRIPPER_CAM_MAX_ITERS
    converged = False

    try:
        for iteration in range(1, GRIPPER_CAM_MAX_ITERS + 1):
            if should_quit(WIN_GRIPPER) or _shutdown:
                break

            print(f"\n--- Gripper servo iteration {iteration}/{GRIPPER_CAM_MAX_ITERS}  Z={z:.3f} ---")

            # Show live feed while waiting for arm to settle
            print(f"[GRIPPER-CAM] Settling ({GRIPPER_SETTLE_SECS}s)...")
            settle_end = time.monotonic() + GRIPPER_SETTLE_SECS
            last_uv = None
            _dbg_saved = False
            while time.monotonic() < settle_end:
                if _shutdown or should_quit(WIN_GRIPPER):
                    break
                frame = cam_gripper.get_frame()
                if frame is not None:
                    if DEBUG_DIR and not _dbg_saved:
                        try:
                            hsv_d = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            m1d = cv2.inRange(hsv_d, (0, 90, 80), (10, 255, 255))
                            m2d = cv2.inRange(hsv_d, (170, 90, 80), (180, 255, 255))
                            cv2.imwrite(f"{DEBUG_DIR}/grip_iter{iteration:02d}.png", frame)
                            cv2.imwrite(f"{DEBUG_DIR}/grip_iter{iteration:02d}_redmask.png",
                                        cv2.bitwise_or(m1d, m2d))
                            _dbg_saved = True
                        except Exception:
                            pass
                    disp = frame.copy()
                    h, w = frame.shape[:2]
                    target_uv = (w * GRIPPER_CAM_TARGET_U_FRAC,
                                 h * GRIPPER_CAM_TARGET_V_FRAC)
                    uv = detect_red_centroid(frame)
                    if uv is not None:
                        last_uv = uv
                    draw_gripper_overlay(disp, brick_uv=uv, target_uv=target_uv,
                                         status_text=f"iter {iteration}/{GRIPPER_CAM_MAX_ITERS}  Z={z:.3f}")
                    cv2.imshow(WIN_GRIPPER, disp)
                key = cv2.waitKey(30) & 0xFF
                check_key_quit(key)

            if _shutdown:
                break

            # Use the last detected brick position from the settle feed
            if last_uv is None:
                print(f"[GRIPPER-CAM] No brick detected in iteration {iteration}")
                # Still descend even if we can't see it
                z -= z_step
                pos = [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
                quat = [float(pick_q[0]), float(pick_q[1]), float(pick_q[2]), float(pick_q[3])]
                traj = moveit2.plan(position=pos, quat_xyzw=quat)
                if traj is not None:
                    moveit2.execute(traj)
                    moveit2.wait_until_executed()
                continue

            brick_u, brick_v = last_uv
            frame = cam_gripper.get_frame()
            h, w = frame.shape[:2] if frame is not None else (480, 640)
            target_u = w * GRIPPER_CAM_TARGET_U_FRAC
            target_v = h * GRIPPER_CAM_TARGET_V_FRAC

            err_u = brick_u - target_u
            err_v = brick_v - target_v
            err_px = math.hypot(err_u, err_v)
            print(f"[GRIPPER-CAM] Brick=({brick_u:.0f},{brick_v:.0f}) "
                  f"Target=({target_u:.0f},{target_v:.0f}) "
                  f"err={err_px:.1f}px")

            if err_px < GRIPPER_CAM_CONVERGE_PX:
                print(f"[GRIPPER-CAM] Converged! err={err_px:.1f}px < {GRIPPER_CAM_CONVERGE_PX}px")
                converged = True
                break

            # Compute XY correction
            dx_robot = GRIPPER_CAM_GAIN_X * GRIPPER_CAM_SERVO_GAIN * err_u
            dy_robot = GRIPPER_CAM_GAIN_Y * GRIPPER_CAM_SERVO_GAIN * err_v
            x += dx_robot
            y += dy_robot

            if not is_safe(x, y):
                print("[GRIPPER-CAM] Outside safe workspace, reverting")
                x -= dx_robot
                y -= dy_robot
                break

            # Descend one step
            z -= z_step

            print(f"[GRIPPER-CAM] Moving to ({x:+.4f}, {y:+.4f}, Z={z:.3f})")
            pos = [round(float(x), 3), round(float(y), 3), round(float(z), 3)]
            quat = [float(pick_q[0]), float(pick_q[1]), float(pick_q[2]), float(pick_q[3])]
            traj = moveit2.plan(position=pos, quat_xyzw=quat)
            if traj is None:
                print("[GRIPPER-CAM] Planning failed!")
                break
            moveit2.execute(traj)
            moveit2.wait_until_executed()

        if not converged:
            print(f"[GRIPPER-CAM] Did not converge after {GRIPPER_CAM_MAX_ITERS} iterations. "
                  "Using best estimate.")

        # Always descend to pick height at the final XY position
        if not _shutdown:
            print(f"[GRIPPER-CAM] Final descent to Z={Z_PRE_PICK}")
            pos = [round(float(x), 3), round(float(y), 3), round(float(Z_PRE_PICK), 3)]
            quat = [float(pick_q[0]), float(pick_q[1]), float(pick_q[2]), float(pick_q[3])]
            traj = moveit2.plan(position=pos, quat_xyzw=quat)
            if traj is not None:
                moveit2.execute(traj)
                moveit2.wait_until_executed()
                interruptible_sleep(GRIPPER_SETTLE_SECS)
            z = Z_PRE_PICK

    finally:
        cv2.destroyWindow(WIN_GRIPPER)

    return x, y, z, converged


# ═══════════════════════  MAIN PICK SEQUENCE  ═══════════════════════

def visual_servo_pick(moveit2: MoveIt2, gripper: "GripperDriver",
                      cam_overhead: CameraSubscriber, cam_gripper: CameraSubscriber):
    """Full autonomous two-camera visual-servoing pick sequence."""

    pick_q = yaw_about_z(SAFE_Q, math.radians(GRIPPER_WORLD_YAW_DEG))

    # ── Phase 1: Detect with overhead camera, move arm ONCE ──
    print("\n=== Phase 1: Overhead detect + coarse move ===")
    try:
        gripper_open(gripper)
    except Exception as e:
        print(f"[WARN] gripper open: {e}")
    if _shutdown:
        return False

    go_hide(moveit2)
    if _shutdown:
        return False

    # Detect brick (with retries)
    detection = None
    for attempt in range(1, DETECT_RETRIES + 1):
        if _shutdown:
            return False
        detection = detect_brick_world(
            cam_overhead, status=f"Detecting brick ({attempt}/{DETECT_RETRIES})")
        if detection is not None:
            break
        print(f"[DETECT] Attempt {attempt}/{DETECT_RETRIES}: no brick found, retrying...")
        interruptible_sleep(1.0)

    if _shutdown:
        return False
    if detection is None:
        print("[ABORT] Could not detect red brick after retries.")
        return False

    target_x, target_y, u, v, _ = detection
    print(f"[DETECT] Brick at pixel ({u:.1f},{v:.1f}) -> robot ({target_x:+.4f},{target_y:+.4f})")

    if not is_safe(target_x, target_y):
        print("[ABORT] Target outside safe workspace.")
        return False

    # Single descent to approach height
    print(f"\n[COARSE] Moving to ({target_x:+.4f}, {target_y:+.4f}) at Z={Z_APPROACH}")
    move_and_settle(moveit2, target_x, target_y, Z_APPROACH, pick_q)
    if _shutdown:
        return False

    # ── Phase 2: Fine alignment + gradual descent (gripper camera) ──
    target_x, target_y, target_z, fine_converged = gripper_cam_fine_align(
        moveit2, cam_gripper, target_x, target_y, pick_q
    )
    if fine_converged:
        print(f"[FINE] Converged at ({target_x:+.4f}, {target_y:+.4f}, Z={target_z:.3f})")
    else:
        print(f"[FINE] Best estimate ({target_x:+.4f}, {target_y:+.4f}, Z={target_z:.3f})")

    if _shutdown:
        return False

    # ── Phase 3: Pick ──
    print("\n=== Phase 3: Pick ===")
    if not is_safe(target_x, target_y):
        print("[ABORT] Final target outside safe workspace.")
        return False

    # If not already at pick height, descend the last bit
    if target_z > Z_PRE_PICK + 0.005:
        print(f"[PICK] Final descent to Z={Z_PRE_PICK}...")
        move_and_settle(moveit2, target_x, target_y, Z_PRE_PICK, pick_q)
        if _shutdown:
            return False
    else:
        print(f"[PICK] Already at pick height Z={target_z:.3f}")

    print("[PICK] Closing gripper...")
    try:
        gripper_close(gripper)
    except Exception as e:
        print(f"[WARN] gripper close: {e}")
    interruptible_sleep(1.0)
    if _shutdown:
        return False

    print("[PICK] Retracting...")
    move_and_settle(moveit2, target_x, target_y, Z_APPROACH, pick_q)
    if _shutdown:
        return False

    print("[PICK] Returning to hide pose...")
    go_hide(moveit2)

    # ── Phase 4: Verify (overhead camera) ──
    print("\n=== Phase 4: Verify ===")
    if _shutdown:
        return False
    detection = detect_brick_world(cam_overhead, status="Verifying pick...")
    if detection is None:
        print("[VERIFY] Brick no longer visible - pick likely succeeded!")
        return True
    else:
        vx, vy, _, _, _ = detection
        print(f"[VERIFY] Brick still detected at ({vx:+.4f},{vy:+.4f}). "
              "Pick may have failed.")
        return False


# ═══════════════════════  MAIN  ═══════════════════════

def main():
    load_cfg()

    rclpy.init()
    node = rclpy.create_node("pick_red_brick_ros")
    cbg = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=[
            "link1_to_link2", "link2_to_link3", "link3_to_link4",
            "link4_to_link5", "link5_to_link6", "link6_to_link6_flange",
        ],
        base_link_name="base_link",
        end_effector_name="gripper_base",
        group_name="arm",
        callback_group=cbg,
    )

    # ── Separate camera node + executor ──
    # pymoveit2 monopolises the MoveIt executor during planning/execution,
    # which starves any other callbacks on the same node.  A dedicated
    # camera node with its own executor guarantees frames keep flowing.
    cam_node = rclpy.create_node("pick_cameras")
    cam_overhead = CameraSubscriber(cam_node, OVERHEAD_TOPIC)
    cam_gripper = CameraSubscriber(cam_node, GRIPPER_TOPIC)

    cam_executor = MultiThreadedExecutor()
    cam_executor.add_node(cam_node)
    cam_spin = Thread(target=cam_executor.spin, daemon=True)
    cam_spin.start()

    # The gripper is driven directly via its GripperActionController (GripperCommand
    # action), NOT via MoveIt — the MoveIt "gripper" group route hangs on this robot.
    # It is attached to cam_node (not the MoveIt node): MoveIt monopolises its own
    # executor during planning/execution, which would starve the gripper action's
    # response future and make close() time out. cam_executor is always free.
    gripper = GripperDriver(cam_node)

    moveit_executor = MultiThreadedExecutor()
    moveit_executor.add_node(node)
    moveit_spin = Thread(target=moveit_executor.spin, daemon=True)
    moveit_spin.start()

    print("[ROS2] Nodes started. Waiting for camera topics...")
    print(f"  Overhead: {OVERHEAD_TOPIC}")
    print(f"  Gripper:  {GRIPPER_TOPIC}")

    frame = cam_overhead.wait_for_frame(timeout=10.0)
    if frame is None:
        print(f"[ERROR] No frames from {OVERHEAD_TOPIC} after 10s. Is the camera running?")
        rclpy.shutdown()
        return
    print("[ROS2] Overhead camera connected.")

    # Also check gripper camera early
    gframe = cam_gripper.wait_for_frame(timeout=5.0)
    if gframe is not None:
        print("[ROS2] Gripper camera connected.")
    else:
        print(f"[WARN] No frames from {GRIPPER_TOPIC} yet (will retry later).")

    cv2.namedWindow(WIN_OVERHEAD, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    try:
        # Auto-detect circle if no saved calibration
        if CAL["m_per_px"] is None:
            print("[CAL] No saved calibration, detecting workspace circle...")
            circ = detect_circle(frame)
            if circ is not None:
                CAL["cx"], CAL["cy"], CAL["r_px"] = circ
                CAL["m_per_px"] = REACH_RADIUS_M / circ[2]
                print(f"[CAL] Circle: center=({circ[0]:.1f},{circ[1]:.1f}), "
                      f"r={circ[2]:.1f}, m/px={CAL['m_per_px']:.6f}")
            else:
                print("[CAL] Circle not found in first frame; will retry during detection.")

        # ── Live preview: press 's' to start, 'q' to quit ──
        print("[PREVIEW] Showing overhead camera. Press 's' to start pick, 'q' to quit.")
        while True:
            if should_quit(WIN_OVERHEAD):
                return

            frame = cam_overhead.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            uv = detect_red_centroid(frame)
            brick_uv = None
            robot_xy = None
            if uv is not None:
                brick_uv = uv
                xy = img_to_robot(uv[0], uv[1])
                if xy is not None:
                    robot_xy = xy

            key = show_frame(frame, WIN_OVERHEAD,
                             "PREVIEW - 's'=start  'q/x/e/ESC'=quit",
                             brick_uv=brick_uv, robot_xy=robot_xy, wait_ms=30)

            if _shutdown:
                return
            if HEADLESS:
                # No keypress possible: auto-start once the brick is detected.
                if robot_xy is not None:
                    print(f"[START] Headless auto-start - brick detected at {robot_xy}.")
                    break
                continue
            if key == ord('s'):
                print("[START] Beginning pick sequence...")
                break

        success = visual_servo_pick(moveit2, gripper,
                                    cam_overhead, cam_gripper)

        if success:
            print("\n*** Pick completed successfully! ***")
        else:
            print("\n*** Pick did not succeed. ***")

        if not _shutdown and not HEADLESS:
            print("[DONE] Press any key to exit.")
            while not _shutdown:
                key = cv2.waitKey(200) & 0xFF
                if key != 255 or should_quit(WIN_OVERHEAD):
                    break

    finally:
        cv2.destroyAllWindows()
        print("[ROS2] Shutting down...")
        rclpy.shutdown()
        cam_spin.join(timeout=2.0)
        moveit_spin.join(timeout=2.0)
        print("[ROS2] Done.")


if __name__ == "__main__":
    main()
