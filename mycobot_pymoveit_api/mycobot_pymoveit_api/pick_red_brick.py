#!/usr/bin/env python3
"""
Autonomous visual-servoing brick pick.

Strategy ("look-then-move"):
  1. Hide arm  -> capture frame -> detect red brick -> pixel-to-world
  2. Move arm to estimated position at approach height
  3. Hide arm again -> re-capture -> re-detect -> compute correction
  4. Repeat until correction < 5 mm threshold (converged)
  5. Execute pick: approach -> descend -> grip -> retract -> verify
"""

import os, time, math, json
from typing import Tuple, Optional
import cv2
import numpy as np
import requests

# ─── Robot endpoints ───
MOVE_URL          = "http://localhost:8080/move"
GRIPPER_OPEN_URL  = "http://localhost:8080/gripper/open"
GRIPPER_CLOSE_URL = "http://localhost:8080/gripper/close"

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
Z_PICK     = 0.08

# ─── Safe tilt quaternion (will be yaw-rotated) ───
SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

# ─── Workspace limits ───
REACH_RADIUS_M = 0.28
KEEP_OUT_R_M   = 0.05
ALLOW_NEG_Y    = False

# ─── Yaw ───
GRIPPER_WORLD_YAW_DEG = 45.0

# ─── Calibration trims (overwritten by load_cfg) ───
X_BIAS  = 0.0
Y_BIAS  = 0.0
X_SCALE = 1.0
Y_SCALE = 1.0
X_ORIGIN_SHIFT_M = 0.0
Y_ORIGIN_SHIFT_M = 0.0

# ─── Visual-servoing parameters ───
SERVO_GAIN        = 0.7     # damped correction
CONVERGE_THRESH_M = 0.005   # 5 mm
MAX_SERVO_ITERS   = 8
FLUSH_FRAMES      = 5       # discard stale USB-camera buffer frames
DETECT_RETRIES    = 3       # retries for initial detection

# ─── Camera ───
CAM_INDEX = 0
WIN_NAME  = "Visual Servo"
HEADLESS  = False   # auto-set True in main() if OpenCV has no GUI backend

# ─── Calibration file (at repo root, two levels above this script) ───
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
    time.sleep(1.5)


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


# ─── Pixel -> robot coordinates ───

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
    """Discard stale frames sitting in the USB camera buffer."""
    for _ in range(n):
        cap.read()


def grab_fresh_frame(cap):
    """Flush buffer, then return a fresh frame."""
    flush_camera(cap)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera read failed")
    return frame


def draw_overlay(img, status_text="", brick_uv=None, robot_xy=None):
    """Draw calibration circle, brick marker, and status text on frame."""
    h, w = img.shape[:2]
    cx, cy, r = CAL["cx"], CAL["cy"], CAL["r_px"]

    # Workspace circle
    if None not in (cx, cy, r):
        cv2.circle(img, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
        cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 255),
                       cv2.MARKER_CROSS, 18, 2)

    # Brick centroid
    if brick_uv is not None:
        u, v = brick_uv
        cv2.circle(img, (int(u), int(v)), 10, (0, 165, 255), -1)
        cv2.circle(img, (int(u), int(v)), 12, (0, 0, 255), 2)
        if robot_xy is not None:
            rx, ry = robot_xy
            cv2.putText(img, f"({rx:+.3f},{ry:+.3f})m",
                        (int(u) + 15, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Status bar at top
    if status_text:
        cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(img, status_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calibration info at bottom
    if CAL["m_per_px"] is not None:
        cv2.putText(img, f"m/px={CAL['m_per_px']:.6f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def show_frame(img, status_text="", brick_uv=None, robot_xy=None, wait_ms=1):
    """Draw overlay and display frame. Returns key pressed (or -1)."""
    if HEADLESS:
        return -1
    disp = img.copy()
    draw_overlay(disp, status_text, brick_uv, robot_xy)
    cv2.imshow(WIN_NAME, disp)
    return cv2.waitKey(wait_ms) & 0xFF


def detect_brick_world(cap, show=True, status="Detecting..."):
    """
    Capture a fresh frame and detect the red brick.
    Returns (robot_X, robot_Y, pixel_u, pixel_v, frame) or None.
    Shows the frame in the preview window when show=True.
    """
    frame = grab_fresh_frame(cap)

    # Ensure calibration is available
    if CAL["m_per_px"] is None:
        circ = detect_circle(frame)
        if circ is None:
            if show:
                show_frame(frame, "WARN: no workspace circle")
            print("[WARN] Cannot detect workspace circle")
            return None
        CAL["cx"], CAL["cy"], CAL["r_px"] = circ
        CAL["m_per_px"] = REACH_RADIUS_M / circ[2]
        print(f"[CAL] Auto-detected circle: center=({circ[0]:.1f},{circ[1]:.1f}), "
              f"r={circ[2]:.1f}, m/px={CAL['m_per_px']:.6f}")

    uv = detect_red_centroid(frame)
    if uv is None:
        if show:
            show_frame(frame, f"{status} - no brick found")
        return None

    u, v = uv
    xy = img_to_robot(u, v)
    if xy is None:
        if show:
            show_frame(frame, f"{status} - mapping failed", brick_uv=(u, v))
        return None

    if show:
        show_frame(frame, status, brick_uv=(u, v), robot_xy=xy)

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


# ═══════════════════════  VISUAL SERVOING  ═══════════════════════

def visual_servo_pick(cap):
    """Full autonomous visual-servoing pick sequence."""

    pick_q = yaw_about_z(SAFE_Q, math.radians(GRIPPER_WORLD_YAW_DEG))

    # ── Phase 1: prepare ──
    print("\n=== Phase 1: Prepare ===")
    try:
        gripper_open()
    except Exception as e:
        print(f"[WARN] gripper open: {e}")
    go_hide()

    # ── Phase 2: initial detection (with retries) ──
    print("\n=== Phase 2: Initial detection ===")
    detection = None
    for attempt in range(1, DETECT_RETRIES + 1):
        detection = detect_brick_world(
            cap, status=f"Detecting brick ({attempt}/{DETECT_RETRIES})")
        if detection is not None:
            break
        print(f"[DETECT] Attempt {attempt}/{DETECT_RETRIES}: no brick found, retrying...")
        time.sleep(0.5)

    if detection is None:
        print("[ABORT] Could not detect red brick after retries.")
        return False

    target_x, target_y, u, v, _ = detection
    print(f"[DETECT] Brick at pixel ({u:.1f},{v:.1f}) -> robot ({target_x:+.4f},{target_y:+.4f})")

    if not is_safe(target_x, target_y):
        print("[ABORT] Target outside safe workspace.")
        return False

    # ── Phase 3: servoing loop ──
    print("\n=== Phase 3: Visual servoing loop ===")
    converged = False

    for iteration in range(1, MAX_SERVO_ITERS + 1):
        print(f"\n--- Servo iteration {iteration}/{MAX_SERVO_ITERS} ---")
        print(f"[TARGET] ({target_x:+.4f}, {target_y:+.4f})")

        # Move to current target at approach height
        if not is_safe(target_x, target_y):
            print("[ABORT] Corrected target outside safe workspace.")
            return False

        call_move(target_x, target_y, Z_APPROACH, pick_q)
        time.sleep(1.5)

        # Hide arm to get a clear camera view
        go_hide()

        # Re-detect brick
        detection = detect_brick_world(
            cap, status=f"Servo iter {iteration}/{MAX_SERVO_ITERS}")
        if detection is None:
            print("[WARN] Lost sight of brick during servoing. "
                  "Trying once more...")
            time.sleep(0.5)
            detection = detect_brick_world(
                cap, status=f"Servo iter {iteration} - retry")
            if detection is None:
                print("[WARN] Brick still not visible. "
                      "Proceeding with last known target.")
                converged = True
                break

        new_x, new_y, u, v, _ = detection
        print(f"[DETECT] Brick now at pixel ({u:.1f},{v:.1f}) "
              f"-> robot ({new_x:+.4f},{new_y:+.4f})")

        # Compute correction
        dx = new_x - target_x
        dy = new_y - target_y
        error = math.hypot(dx, dy)
        print(f"[SERVO] Correction needed: dx={dx:+.4f} dy={dy:+.4f} "
              f"|err|={error:.4f} m")

        if error < CONVERGE_THRESH_M:
            print(f"[SERVO] Converged! Error {error*1000:.1f} mm < "
                  f"{CONVERGE_THRESH_M*1000:.1f} mm threshold")
            # Use the freshly detected position as final target
            target_x = new_x
            target_y = new_y
            converged = True
            break

        # Apply damped correction
        target_x += SERVO_GAIN * dx
        target_y += SERVO_GAIN * dy
        print(f"[SERVO] Updated target: ({target_x:+.4f}, {target_y:+.4f}) "
              f"(gain={SERVO_GAIN})")

    if not converged:
        print(f"[WARN] Did not converge after {MAX_SERVO_ITERS} iterations. "
              "Attempting pick with best estimate.")

    # ── Phase 4: pick sequence ──
    print("\n=== Phase 4: Pick ===")
    if not is_safe(target_x, target_y):
        print("[ABORT] Final target outside safe workspace.")
        return False

    print(f"[PICK] Final target: ({target_x:+.4f}, {target_y:+.4f})")

    # Approach height
    print("[PICK] Moving to approach height...")
    call_move(target_x, target_y, Z_APPROACH, pick_q)
    time.sleep(1.5)

    # Descend to pick height
    print("[PICK] Descending to pick height...")
    call_move(target_x, target_y, Z_PICK, pick_q)
    time.sleep(2.0)

    # Close gripper (send multiple times for reliability over serial)
    print("[PICK] Closing gripper...")
    for attempt in range(1, 4):
        try:
            gripper_close()
            print(f"[PICK] Gripper close sent ({attempt}/3)")
        except Exception as e:
            print(f"[WARN] gripper close attempt {attempt}: {e}")
        time.sleep(1.5)

    # Retract to approach height
    print("[PICK] Retracting...")
    call_move(target_x, target_y, Z_APPROACH, pick_q)
    time.sleep(1.0)

    # Move to hide/home
    print("[PICK] Returning to hide pose...")
    go_hide()

    # ── Phase 5: verify ──
    print("\n=== Phase 5: Verify ===")
    time.sleep(1.0)
    detection = detect_brick_world(cap, status="Verifying pick...")
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
    global HEADLESS
    load_cfg()

    try:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    except cv2.error:
        HEADLESS = True
        print("[GUI] OpenCV has no display backend - running HEADLESS (auto-start on brick detection).")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")
    print(f"[CAM] Opened camera {CAM_INDEX}")

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
            ok, frame = cap.read()
            if not ok:
                continue

            # Run detection on each preview frame for visual feedback
            uv = detect_red_centroid(frame)
            brick_uv = None
            robot_xy = None
            if uv is not None:
                brick_uv = uv
                xy = img_to_robot(uv[0], uv[1])
                if xy is not None:
                    robot_xy = xy

            key = show_frame(frame, "PREVIEW - 's'=start  'q'=quit",
                             brick_uv=brick_uv, robot_xy=robot_xy, wait_ms=30)

            if HEADLESS:
                # No keypress possible: auto-start once the brick is detected.
                if robot_xy is not None:
                    print(f"[START] Headless auto-start - brick detected at {robot_xy}.")
                    break
                continue
            if key == ord('q'):
                print("[QUIT] User cancelled.")
                return
            if key == ord('s'):
                print("[START] Beginning visual servo pick...")
                break

        success = visual_servo_pick(cap)

        if success:
            print("\n*** Pick completed successfully! ***")
        else:
            print("\n*** Pick did not succeed. ***")

        # Hold the window open until user presses a key (skip when headless)
        if not HEADLESS:
            print("[DONE] Press any key in the camera window to exit.")
            cv2.waitKey(0)

    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("[CAM] Camera released.")


if __name__ == "__main__":
    main()
