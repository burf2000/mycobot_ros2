#!/usr/bin/env python3
"""
Two-camera visual-servoing brick pick (v2).

Phases:
  1. Coarse positioning with overhead camera (same as v1 phases 1-3)
  2. Fine alignment with gripper-mounted camera (new)
  3. Pick: descend -> grip -> retract
  4. Verify with overhead camera
"""

import os, time, math, json, signal
from typing import Tuple, Optional
import cv2
import numpy as np
import requests

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

# ─── Robot endpoints ───
MOVE_URL               = "http://localhost:8080/move"
GRIPPER_OPEN_URL       = "http://localhost:8080/gripper_moveit/open"
GRIPPER_CLOSE_URL      = "http://localhost:8080/gripper_moveit/closed"

# ─── Poses ───
HOME_POSE = {
    "posX": 0.06, "posY": 0.079, "posZ": 0.411,
    "rotX": 0.03, "rotY": -0.382, "rotZ": 0.001, "rotW": 1.0,
}
HIDE_POSE = {
    "posX": 0.06, "posY": 0.005, "posZ": 0.444,
    "rotX": 0.352, "rotY": -0.354, "rotZ": -0.143, "rotW": 0.854,
}

# ─── Heights (m) ───
Z_APPROACH = 0.12
Z_PRE_PICK = 0.09    # ~1cm above surface (gripper fingers swing down to grab)

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

# ─── Visual-servoing parameters (overhead) ───
SERVO_GAIN        = 0.7
CONVERGE_THRESH_M = 0.005
MAX_SERVO_ITERS   = 8
FLUSH_FRAMES      = 5
DETECT_RETRIES    = 3

# ─── Camera indices ───
CAM_INDEX_OVERHEAD = 2
CAM_INDEX_GRIPPER  = 4

# ─── Gripper camera fine-alignment parameters ───
# Separate gains per axis (m/pixel). Flip sign if axis moves wrong way.
GRIPPER_CAM_GAIN_X       =  0.00005  # pixel-U error -> robot X (flip sign if wrong)
GRIPPER_CAM_GAIN_Y       = -0.00005  # pixel-V error -> robot Y (flip sign if wrong)
GRIPPER_CAM_TARGET_U_FRAC = 0.5     # horizontal center
GRIPPER_CAM_TARGET_V_FRAC = 0.85    # bottom portion of frame
GRIPPER_CAM_CONVERGE_PX  = 15       # pixel convergence threshold
GRIPPER_CAM_MAX_ITERS    = 12
GRIPPER_CAM_SERVO_GAIN   = 0.4      # damping factor (lower = slower, more cautious)
GRIPPER_CAM_PREVIEW_SECS = 1.0      # seconds to show camera feed between moves

# ─── Window names ───
WIN_OVERHEAD = "Overhead Camera"
WIN_GRIPPER  = "Gripper Camera"

# ─── Calibration file ───
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_FILE = os.path.join(_SCRIPT_DIR, "..", "..", "vision_circle_calib.json")
CAL = {
    "cx": None, "cy": None, "r_px": None, "m_per_px": None,
    "x_origin_shift_m": 0.0, "y_origin_shift_m": 0.0,
    "x_bias": 0.0, "y_bias": 0.0, "x_scale": 1.0, "y_scale": 1.0,
}

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


# ─── Robot moves ───

def call_move_pose(pose, timeout=15.0):
    body = dict(pose)
    for k in ("posX", "posY", "posZ"):
        body[k] = round(float(body[k]), 3)
    print(f"[MOVE] -> {body}")
    r = requests.post(MOVE_URL, json=body, timeout=timeout)
    if r.status_code >= 400:
        print(f"[MOVE][HTTP] {r.status_code} {r.text[:300]}")
    r.raise_for_status()
    return r.json() if r.text else {"ok": True}


def call_move(x, y, z, q, timeout=15.0):
    return call_move_pose({
        "posX": float(x), "posY": float(y), "posZ": float(z),
        "rotX": q[0], "rotY": q[1], "rotZ": q[2], "rotW": q[3],
    }, timeout=timeout)


def gripper_open():
    print("[GRIPPER] Opening...")
    r = requests.post(GRIPPER_OPEN_URL, timeout=8)
    print(f"[GRIPPER] Open response: {r.status_code} {r.text[:200]}")
    r.raise_for_status()


def gripper_close():
    print("[GRIPPER] Closing...")
    r = requests.post(GRIPPER_CLOSE_URL, timeout=8)
    print(f"[GRIPPER] Close response: {r.status_code} {r.text[:200]}")
    r.raise_for_status()


def go_hide():
    print("[HIDE] Moving arm out of camera view")
    call_move_pose(HIDE_POSE)
    interruptible_sleep(1.5)


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


# ─── Pixel -> robot coordinates (overhead camera) ───

def img_to_robot(u, v) -> Optional[Tuple[float, float]]:
    cx, cy, r = CAL["cx"], CAL["cy"], CAL["r_px"]
    m_per_px = CAL["m_per_px"]
    if None in (cx, cy, r, m_per_px):
        return None
    X = ((cx - u) * m_per_px) * X_SCALE + X_BIAS + X_ORIGIN_SHIFT_M
    Y = ((v  - cy) * m_per_px) * Y_SCALE + Y_BIAS + Y_ORIGIN_SHIFT_M
    return X, Y


# ═══════════════════════  CAMERA / DISPLAY  ═══════════════════════

def flush_camera(cap, n=FLUSH_FRAMES):
    for _ in range(n):
        cap.read()


def grab_fresh_frame(cap):
    flush_camera(cap)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    return frame


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
    """Return True if Ctrl-C was pressed or the given window was closed."""
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
    """Return True if the key is a quit key (q, x, e, or ESC)."""
    global _shutdown
    if key in (ord('q'), ord('x'), ord('e'), 27):  # 27 = ESC
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


def detect_brick_world(cap, show=True, status="Detecting..."):
    frame = grab_fresh_frame(cap)
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


# ─── Workspace safety check ───

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

def draw_gripper_overlay(img, brick_uv=None, target_uv=None, status_text="", iteration=0):
    """Draw target crosshair, brick marker, and error line on gripper camera frame."""
    h, w = img.shape[:2]

    # Target crosshair
    if target_uv is not None:
        tu, tv = int(target_uv[0]), int(target_uv[1])
        cv2.drawMarker(img, (tu, tv), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
        cv2.circle(img, (tu, tv), 20, (0, 255, 0), 1)

    # Brick centroid
    if brick_uv is not None:
        bu, bv = int(brick_uv[0]), int(brick_uv[1])
        cv2.circle(img, (bu, bv), 8, (0, 165, 255), -1)
        cv2.circle(img, (bu, bv), 10, (0, 0, 255), 2)

        # Error line from brick to target
        if target_uv is not None:
            tu, tv = int(target_uv[0]), int(target_uv[1])
            cv2.line(img, (bu, bv), (tu, tv), (0, 0, 255), 2)
            err = math.hypot(brick_uv[0] - target_uv[0], brick_uv[1] - target_uv[1])
            cv2.putText(img, f"err={err:.1f}px", (bu + 12, bv - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Status bar
    if status_text:
        cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(img, status_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def gripper_cam_fine_align(current_x, current_y, pick_q):
    """
    Phase 2: Fine alignment using the gripper-mounted camera.

    Opens the gripper camera, detects the brick, and iteratively adjusts
    the arm position until the brick centroid aligns with the target
    position in the gripper camera frame.

    Returns (final_x, final_y, converged).
    """
    print("\n=== Phase 2: Gripper camera fine alignment ===")

    cap_gripper = cv2.VideoCapture(CAM_INDEX_GRIPPER)
    if not cap_gripper.isOpened():
        print(f"[WARN] Cannot open gripper camera (index {CAM_INDEX_GRIPPER}). "
              "Skipping fine alignment.")
        return current_x, current_y, False

    cv2.namedWindow(WIN_GRIPPER, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    print(f"[CAM] Opened gripper camera (index {CAM_INDEX_GRIPPER})")

    x, y = current_x, current_y
    converged = False

    try:
        for iteration in range(1, GRIPPER_CAM_MAX_ITERS + 1):
            if should_quit(WIN_GRIPPER):
                print("[QUIT] Shutdown requested during gripper servo.")
                break

            print(f"\n--- Gripper servo iteration {iteration}/{GRIPPER_CAM_MAX_ITERS} ---")

            # Show live camera feed for a few seconds so user can see
            print(f"[GRIPPER-CAM] Showing live feed for {GRIPPER_CAM_PREVIEW_SECS}s...")
            preview_end = time.monotonic() + GRIPPER_CAM_PREVIEW_SECS
            last_uv = None
            last_frame = None
            while time.monotonic() < preview_end:
                if _shutdown or should_quit(WIN_GRIPPER):
                    break
                ok, frame = cap_gripper.read()
                if not ok:
                    continue
                last_frame = frame
                h, w = frame.shape[:2]
                target_u = w * GRIPPER_CAM_TARGET_U_FRAC
                target_v = h * GRIPPER_CAM_TARGET_V_FRAC
                uv = detect_red_centroid(frame)
                disp = frame.copy()
                if uv is not None:
                    last_uv = uv
                    draw_gripper_overlay(disp, brick_uv=uv,
                                         target_uv=(target_u, target_v),
                                         status_text=f"Gripper iter {iteration}/{GRIPPER_CAM_MAX_ITERS} - LIVE")
                else:
                    draw_gripper_overlay(disp, target_uv=(target_u, target_v),
                                         status_text=f"Gripper iter {iteration}/{GRIPPER_CAM_MAX_ITERS} - no brick")
                cv2.imshow(WIN_GRIPPER, disp)
                key = cv2.waitKey(30) & 0xFF
                check_key_quit(key)

            if _shutdown:
                break

            # Use the last detected position
            if last_uv is None or last_frame is None:
                print(f"[GRIPPER-CAM] No brick detected in iteration {iteration}")
                continue

            brick_u, brick_v = last_uv
            h, w = last_frame.shape[:2]
            target_u = w * GRIPPER_CAM_TARGET_U_FRAC
            target_v = h * GRIPPER_CAM_TARGET_V_FRAC

            print(f"[GRIPPER-CAM] Brick at pixel ({brick_u:.1f}, {brick_v:.1f}), "
                  f"target ({target_u:.1f}, {target_v:.1f})")

            # Compute pixel error
            err_u = brick_u - target_u
            err_v = brick_v - target_v
            err_px = math.hypot(err_u, err_v)
            print(f"[GRIPPER-CAM] Pixel error: du={err_u:.1f} dv={err_v:.1f} |err|={err_px:.1f} px")

            # Check convergence
            if err_px < GRIPPER_CAM_CONVERGE_PX:
                print(f"[GRIPPER-CAM] Converged! Error {err_px:.1f} px < "
                      f"{GRIPPER_CAM_CONVERGE_PX} px threshold")
                converged = True
                break

            # Simple per-axis mapping (no yaw rotation — tune signs empirically)
            dx_robot = GRIPPER_CAM_GAIN_X * GRIPPER_CAM_SERVO_GAIN * err_u
            dy_robot = GRIPPER_CAM_GAIN_Y * GRIPPER_CAM_SERVO_GAIN * err_v
            print(f"[GRIPPER-CAM] Robot delta: dx={dx_robot:+.5f} dy={dy_robot:+.5f} m")

            # Apply correction
            x += dx_robot
            y += dy_robot
            print(f"[GRIPPER-CAM] New position: ({x:+.4f}, {y:+.4f})")

            if not is_safe(x, y):
                print("[GRIPPER-CAM] Corrected position outside safe workspace, reverting")
                x -= dx_robot
                y -= dy_robot
                break

            # Move to corrected position
            print("[GRIPPER-CAM] Moving arm...")
            call_move(x, y, Z_APPROACH, pick_q)
            interruptible_sleep(1.5)

        if not converged:
            print(f"[GRIPPER-CAM] Did not converge after {GRIPPER_CAM_MAX_ITERS} iterations. "
                  "Proceeding with best estimate.")

    finally:
        cap_gripper.release()
        cv2.destroyWindow(WIN_GRIPPER)
        print("[CAM] Gripper camera released.")

    return x, y, converged


# ═══════════════════════  VISUAL SERVOING  ═══════════════════════

def visual_servo_pick(cap_overhead):
    """Full autonomous two-camera visual-servoing pick sequence."""

    pick_q = yaw_about_z(SAFE_Q, math.radians(GRIPPER_WORLD_YAW_DEG))

    # ── Phase 1: Coarse positioning (overhead camera) ──
    print("\n=== Phase 1: Prepare ===")
    try:
        gripper_open()
    except Exception as e:
        print(f"[WARN] gripper open: {e}")
    if _shutdown:
        return False
    go_hide()
    if _shutdown:
        return False

    # Initial detection (with retries)
    print("\n--- Initial detection ---")
    detection = None
    for attempt in range(1, DETECT_RETRIES + 1):
        if _shutdown:
            return False
        detection = detect_brick_world(
            cap_overhead, status=f"Detecting brick ({attempt}/{DETECT_RETRIES})")
        if detection is not None:
            break
        print(f"[DETECT] Attempt {attempt}/{DETECT_RETRIES}: no brick found, retrying...")
        interruptible_sleep(0.5)

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

    # Overhead servoing loop
    print("\n--- Overhead visual servoing loop ---")
    converged = False

    for iteration in range(1, MAX_SERVO_ITERS + 1):
        if should_quit(WIN_OVERHEAD):
            print("[QUIT] Shutdown requested during overhead servo.")
            return False

        print(f"\n--- Overhead servo iteration {iteration}/{MAX_SERVO_ITERS} ---")
        print(f"[TARGET] ({target_x:+.4f}, {target_y:+.4f})")

        if not is_safe(target_x, target_y):
            print("[ABORT] Corrected target outside safe workspace.")
            return False

        call_move(target_x, target_y, Z_APPROACH, pick_q)
        interruptible_sleep(1.5)
        if _shutdown:
            return False

        go_hide()
        if _shutdown:
            return False

        detection = detect_brick_world(
            cap_overhead, status=f"Servo iter {iteration}/{MAX_SERVO_ITERS}")
        if detection is None:
            print("[WARN] Lost sight of brick during servoing. Trying once more...")
            interruptible_sleep(0.5)
            detection = detect_brick_world(
                cap_overhead, status=f"Servo iter {iteration} - retry")
            if detection is None:
                print("[WARN] Brick still not visible. Proceeding with last known target.")
                converged = True
                break

        new_x, new_y, u, v, _ = detection
        print(f"[DETECT] Brick now at pixel ({u:.1f},{v:.1f}) "
              f"-> robot ({new_x:+.4f},{new_y:+.4f})")

        dx = new_x - target_x
        dy = new_y - target_y
        error = math.hypot(dx, dy)
        print(f"[SERVO] Correction needed: dx={dx:+.4f} dy={dy:+.4f} |err|={error:.4f} m")

        if error < CONVERGE_THRESH_M:
            print(f"[SERVO] Converged! Error {error*1000:.1f} mm < "
                  f"{CONVERGE_THRESH_M*1000:.1f} mm threshold")
            target_x = new_x
            target_y = new_y
            converged = True
            break

        target_x += SERVO_GAIN * dx
        target_y += SERVO_GAIN * dy
        print(f"[SERVO] Updated target: ({target_x:+.4f}, {target_y:+.4f}) "
              f"(gain={SERVO_GAIN})")

    if not converged:
        print(f"[WARN] Overhead servo did not converge after {MAX_SERVO_ITERS} iterations. "
              "Continuing to fine alignment.")

    if _shutdown:
        return False

    # Move arm to target at approach height for Phase 2
    print(f"\n[COARSE] Moving to coarse target ({target_x:+.4f}, {target_y:+.4f}) "
          f"at Z={Z_APPROACH}")
    call_move(target_x, target_y, Z_APPROACH, pick_q)
    interruptible_sleep(1.5)
    if _shutdown:
        return False

    # ── Phase 2: Fine alignment (gripper camera) ──
    target_x, target_y, fine_converged = gripper_cam_fine_align(
        target_x, target_y, pick_q
    )
    if fine_converged:
        print(f"[FINE] Gripper camera alignment converged at "
              f"({target_x:+.4f}, {target_y:+.4f})")
    else:
        print(f"[FINE] Using best estimate ({target_x:+.4f}, {target_y:+.4f})")

    if _shutdown:
        return False

    # ── Phase 3: Pick ──
    print("\n=== Phase 3: Pick ===")
    if not is_safe(target_x, target_y):
        print("[ABORT] Final target outside safe workspace.")
        return False

    print(f"[PICK] Final target: ({target_x:+.4f}, {target_y:+.4f})")

    # Move to pick height (1cm above surface — gripper fingers swing down)
    print(f"[PICK] Moving to pick height (Z={Z_PRE_PICK})...")
    call_move(target_x, target_y, Z_PRE_PICK, pick_q)
    interruptible_sleep(1.5)
    if _shutdown:
        return False

    # Close gripper via MoveIt
    print("[PICK] Closing gripper (MoveIt)...")
    try:
        gripper_close()
        print("[PICK] Gripper close done")
    except Exception as e:
        print(f"[WARN] gripper close: {e}")
    interruptible_sleep(2.0)

    if _shutdown:
        return False

    # Retract to approach height
    print("[PICK] Retracting...")
    call_move(target_x, target_y, Z_APPROACH, pick_q)
    interruptible_sleep(1.0)
    if _shutdown:
        return False

    # Move to hide/home
    print("[PICK] Returning to hide pose...")
    go_hide()

    # ── Phase 4: Verify (overhead camera) ──
    print("\n=== Phase 4: Verify ===")
    interruptible_sleep(1.0)
    if _shutdown:
        return False
    detection = detect_brick_world(cap_overhead, status="Verifying pick...")
    if detection is None:
        print("[VERIFY] Brick no longer visible - pick likely succeeded!")
        return True
    else:
        vx, vy, vu, vv, _ = detection
        print(f"[VERIFY] Brick still detected at ({vx:+.4f},{vy:+.4f}). "
              "Pick may have failed.")
        return False


# ═══════════════════════  MAIN  ═══════════════════════

def main():
    load_cfg()

    cv2.namedWindow(WIN_OVERHEAD, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cap = cv2.VideoCapture(CAM_INDEX_OVERHEAD)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open overhead camera index {CAM_INDEX_OVERHEAD}")
    print(f"[CAM] Opened overhead camera {CAM_INDEX_OVERHEAD}")

    try:
        # Auto-detect circle if no saved calibration
        if CAL["m_per_px"] is None:
            print("[CAL] No saved calibration, detecting workspace circle...")
            flush_camera(cap)
            ok, frame = cap.read()
            if ok:
                circ = detect_circle(frame)
                if circ is not None:
                    CAL["cx"], CAL["cy"], CAL["r_px"] = circ
                    CAL["m_per_px"] = REACH_RADIUS_M / circ[2]
                    print(f"[CAL] Circle: center=({circ[0]:.1f},{circ[1]:.1f}), "
                          f"r={circ[2]:.1f}, m/px={CAL['m_per_px']:.6f}")
                else:
                    print("[CAL] Circle not found in first frame; "
                          "will retry during detection.")

        # ── Live preview loop: press 's' to start, 'q' to quit ──
        print("[PREVIEW] Showing camera. Press 's' to start pick, 'q' to quit.")
        while True:
            if should_quit(WIN_OVERHEAD):
                print("[QUIT] Shutdown requested.")
                return

            ok, frame = cap.read()
            if not ok:
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
                print("[QUIT] User cancelled.")
                return
            if key == ord('s'):
                print("[START] Beginning two-camera visual servo pick...")
                break

        success = visual_servo_pick(cap)

        if success:
            print("\n*** Pick completed successfully! ***")
        else:
            print("\n*** Pick did not succeed. ***")

        if not _shutdown:
            print("[DONE] Press any key in the camera window to exit.")
            while not _shutdown:
                key = cv2.waitKey(200) & 0xFF
                if key != 255 or should_quit(WIN_OVERHEAD):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[CAM] Cameras released.")


if __name__ == "__main__":
    main()
