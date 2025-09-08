#!/usr/bin/env python3
import os, time, math, json, base64
from typing import Tuple, Dict, Optional
import cv2
import numpy as np
import requests

# ------------ CONFIG / IO ------------
CAM_INDEX = 0

# Robot move endpoints
MOVE_URL = "http://localhost:8080/move"
GRIPPER_OPEN_URL  = "http://localhost:8080/gripper/open"
GRIPPER_CLOSE_URL = "http://localhost:8080/gripper/close"

HOME_POSE = {
    "posX": 0.06, "posY": 0.079, "posZ": 0.411,
    "rotX": 0.03, "rotY": -0.382, "rotZ": 0.001, "rotW": 1.0
}

HIDE_POSE = {
    "posX": 0.06, "posY": 0.005, "posZ": 0.444,
    "rotX": 0.352, "rotY": -0.354, "rotZ": -0.143, "rotW": 0.854
}


# Heights (m)
Z_APPROACH = 0.12
Z_PICK     = 0.08

# Safe tilt quaternion (will be yaw-rotated)
SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

# Workspace geometry
REACH_RADIUS_M = 0.28           # 28 cm semicircle
KEEP_OUT_R_M   = 0.05           # 5 cm near the base
ALLOW_NEG_Y    = False

# Yaw control
FORCE_HORIZONTAL_YAW = False
GRIPPER_WORLD_YAW_DEG = -90.0
YAW_STEP = 5.0

# Small trims (meters)
X_BIAS = 0.000
Y_BIAS = 0.000
X_SCALE = 1.000
Y_SCALE = 1.000

# Origin micro shifts (meters) — persisted
X_ORIGIN_SHIFT_M = 0.000   # + shifts origin LEFT (X readings larger)
Y_ORIGIN_SHIFT_M = 0.000   # + shifts origin DOWN (Y readings larger)

# Saved camera calibration (derived from circle)
CFG_FILE = "vision_circle_calib.json"
CAL = {
    "cx": None,          # circle center u (px)
    "cy": None,          # circle center v (px)
    "r_px": None,        # circle radius (px)
    "m_per_px": None,    # meters per pixel, = REACH_RADIUS_M / r_px
    "x_origin_shift_m": 0.0,
    "y_origin_shift_m": 0.0,
    "x_bias": 0.0, "y_bias": 0.0, "x_scale": 1.0, "y_scale": 1.0
}

# ------------ UTILS ------------

def load_cfg():
    global CAL, X_ORIGIN_SHIFT_M, Y_ORIGIN_SHIFT_M, X_BIAS, Y_BIAS, X_SCALE, Y_SCALE
    try:
        with open(CFG_FILE, "r") as f:
            d = json.load(f)
        CAL.update(d)
        X_ORIGIN_SHIFT_M = float(CAL.get("x_origin_shift_m", 0.0))
        Y_ORIGIN_SHIFT_M = float(CAL.get("y_origin_shift_m", 0.0))
        X_BIAS = float(CAL.get("x_bias", 0.0))
        Y_BIAS = float(CAL.get("y_bias", 0.0))
        X_SCALE = float(CAL.get("x_scale", 1.0))
        Y_SCALE = float(CAL.get("y_scale", 1.0))
        print(f"[CFG] Loaded {CFG_FILE}: center=({CAL['cx']},{CAL['cy']}), r_px={CAL['r_px']}, m/px={CAL['m_per_px']:.6f} "
              f"shifts(X,Y)=({X_ORIGIN_SHIFT_M:+.3f},{Y_ORIGIN_SHIFT_M:+.3f}) biases=({X_BIAS:+.3f},{Y_BIAS:+.3f})")
    except Exception:
        print("[CFG] No saved calibration; will auto-detect circle.")

def save_cfg():
    CAL["x_origin_shift_m"] = float(X_ORIGIN_SHIFT_M)
    CAL["y_origin_shift_m"] = float(Y_ORIGIN_SHIFT_M)
    CAL["x_bias"] = float(X_BIAS)
    CAL["y_bias"] = float(Y_BIAS)
    CAL["x_scale"] = float(X_SCALE)
    CAL["y_scale"] = float(Y_SCALE)
    try:
        with open(CFG_FILE, "w") as f:
            json.dump(CAL, f, indent=2)
        print(f"[CFG] Saved -> {CFG_FILE}")
    except Exception as e:
        print("[CFG] Save failed:", e)

def b64_jpeg(img_bgr, q=90):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok: raise RuntimeError("jpeg encode failed")
    import base64
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def yaw_about_z(q, yaw_rad):
    x1,y1,z1,w1 = q
    cz, sz = math.cos(yaw_rad/2.0), math.sin(yaw_rad/2.0)
    x2,y2,z2,w2 = 0.0,0.0,sz,cz
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1
    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    return (x,y,z,w)

def call_move_pose(pose, timeout=15.0):
    body = dict(pose)
    for k in ("posX","posY","posZ"):
        body[k] = round(float(body[k]), 3)
    print("[MOVE] ->", body)
    r = requests.post(MOVE_URL, json=body, timeout=timeout)
    if r.status_code >= 400: print("[MOVE][HTTP]", r.status_code, r.text[:300])
    r.raise_for_status()
    return r.json() if r.text else {"ok": True}

def call_move(x,y,z,q, timeout=15.0):
    return call_move_pose({"posX":float(x),"posY":float(y),"posZ":float(z),
                           "rotX":q[0],"rotY":q[1],"rotZ":q[2],"rotW":q[3]}, timeout=timeout)

def gripper_open():  requests.post(GRIPPER_OPEN_URL, timeout=8).raise_for_status()
def gripper_close(): requests.post(GRIPPER_CLOSE_URL, timeout=8).raise_for_status()

# ------------ DETECTION ------------

def detect_circle(img_bgr) -> Optional[Tuple[float,float,float]]:
    """
    Detect the black semicircle as a circle (cx,cy,r_px) in the *camera image*.
    Returns None if not found.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 5)
    # emphasize dark ring
    edges = cv2.Canny(g, 50, 120)
    # Hough circle
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                               param1=120, param2=40, minRadius=150, maxRadius=900)
    if circles is not None:
        c = circles[0][0]
        return float(c[0]), float(c[1]), float(c[2])
    # fallback: contour fit
    _,th = cv2.threshold(g, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    (x,y), r = cv2.minEnclosingCircle(c)
    if r>150:
        return float(x), float(y), float(r)
    return None

def detect_red_centroid(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 80), (10,255,255))
    m2 = cv2.inRange(hsv, (170,90, 80), (180,255,255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < 60: return None
    M = cv2.moments(c);  m00 = M["m00"]
    if m00 == 0: return None
    u = M["m10"]/m00; v = M["m01"]/m00
    return float(u), float(v)

def go_home():
    try: gripper_open()
    except Exception as e: print("[WARN] gripper open:", e)
    try: call_move_pose(HIDE_POSE)
    except Exception as e: print("[WARN] home move:", e)

# ------------ MAPPING (camera opposite side) ------------
def img_to_robot(u,v) -> Optional[Tuple[float,float]]:
    """
    Use current CAL (cx,cy,r_px) to convert a camera pixel (u,v) to robot (X,Y) in meters.
    Polarity: +X is image-LEFT; +Y is image-DOWN (camera opposite).
    """
    cx,cy,r = CAL["cx"], CAL["cy"], CAL["r_px"]
    m_per_px = CAL["m_per_px"]
    if None in (cx,cy,r,m_per_px): return None
    X = ((cx - u) * m_per_px) * X_SCALE + X_BIAS + X_ORIGIN_SHIFT_M
    Y = ((v  - cy) * m_per_px) * Y_SCALE + Y_BIAS + Y_ORIGIN_SHIFT_M
    return X, Y

# ------------ OVERLAY ------------
def draw_overlay(img, u, v, X, Y):
    h,w = img.shape[:2]
    cx,cy,r = CAL["cx"], CAL["cy"], CAL["r_px"]
    # circle + center
    if None not in (cx,cy,r):
        cv2.circle(img, (int(cx),int(cy)), int(r), (0,255,255), 2)
        cv2.drawMarker(img, (int(cx),int(cy)), (255,255,255), cv2.MARKER_CROSS, 18, 2)
    # brick
    if u is not None and v is not None:
        cv2.circle(img, (int(u),int(v)), 8, (0,165,255), -1)
    # text
    cv2.putText(img, f"(X,Y)=({X:+.3f},{Y:+.3f}) m", (18,h-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    cv2.putText(img, f"m/px={CAL['m_per_px']:.6f}  shifts=({X_ORIGIN_SHIFT_M:+.3f},{Y_ORIGIN_SHIFT_M:+.3f})",
                (18,h-52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

# ------------ MAIN ------------
def main():
    global X_ORIGIN_SHIFT_M, Y_ORIGIN_SHIFT_M, X_BIAS, Y_BIAS, X_SCALE, Y_SCALE
    load_cfg()

    cv2.namedWindow("cam", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue

            # 1) circle detect (update CAL if found)
            found = detect_circle(frame)
            if found is not None:
                cx,cy,r = found
                CAL["cx"], CAL["cy"], CAL["r_px"] = cx,cy,r
                CAL["m_per_px"] = REACH_RADIUS_M / r
            elif CAL["m_per_px"] is None:
                cv2.imshow("cam", frame)
                cv2.waitKey(10)
                print("[WARN] Circle not found yet; no scale/origin. Move/lighting?")
                continue  # we need at least one calibration

            # 2) brick centroid
            uv = detect_red_centroid(frame)
            X = Y = None
            if uv is not None:
                u,v = uv
                xy = img_to_robot(u,v)
                if xy is not None:
                    X,Y = xy

            # 3) overlay
            disp = frame.copy()
            if uv is not None:
                draw_overlay(disp, u, v, X, Y)
            else:
                draw_overlay(disp, None, None, 0.0, 0.0)
            cv2.putText(disp, "s=save  [/] Xshift +-1mm  ;/' Yshift +-1mm  1..8 trims  g yaw  9/0 yaw step  c=pick  q=quit",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.imshow("cam", disp)

            # 4) keys
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'): break
            if k == ord('s'): save_cfg(); continue
            if k == ord('['): X_ORIGIN_SHIFT_M -= 0.001; print("X_SHIFT",X_ORIGIN_SHIFT_M); continue
            if k == ord(']'): X_ORIGIN_SHIFT_M += 0.001; print("X_SHIFT",X_ORIGIN_SHIFT_M); continue
            if k == ord(';'): Y_ORIGIN_SHIFT_M -= 0.001; print("Y_SHIFT",Y_ORIGIN_SHIFT_M); continue
            if k == ord('\''): Y_ORIGIN_SHIFT_M += 0.001; print("Y_SHIFT",Y_ORIGIN_SHIFT_M); continue
            if k == ord('1'): X_BIAS -= 0.005; print("X_BIAS",X_BIAS); continue
            if k == ord('2'): X_BIAS += 0.005; print("X_BIAS",X_BIAS); continue
            if k == ord('3'): Y_BIAS -= 0.005; print("Y_BIAS",Y_BIAS); continue
            if k == ord('4'): Y_BIAS += 0.005; print("Y_BIAS",Y_BIAS); continue
            if k == ord('5'): X_SCALE *= 0.99;  print("X_SCALE",X_SCALE); continue
            if k == ord('6'): X_SCALE *= 1.01;  print("X_SCALE",X_SCALE); continue
            if k == ord('7'): Y_SCALE *= 0.99;  print("Y_SCALE",Y_SCALE); continue
            if k == ord('8'): Y_SCALE *= 1.01;  print("Y_SCALE",Y_SCALE); continue

            if k == ord('h'):
                print("[RESET] Home…"); 
                go_home()
                continue


            # 5) pick
            if k == ord('c') and uv is not None and X is not None and Y is not None:
                r = math.hypot(X,Y)
                if r < KEEP_OUT_R_M: 
                    print(f"[SKIP] inside keep-out (r={r:.3f})"); 
                    continue
                if (not ALLOW_NEG_Y) and Y < 0:
                    print(f"[SKIP] Y={Y:.3f} < 0"); 
                    continue
                if r > REACH_RADIUS_M:
                    print(f"[SKIP] outside reach (r={r:.3f})");
                    continue

                # Yaw (simple: keep horizontal, or 45deg if you like)
                yaw_deg = GRIPPER_WORLD_YAW_DEG if FORCE_HORIZONTAL_YAW else 45.0
                qx,qy,qz,qw = yaw_about_z(SAFE_Q, math.radians(yaw_deg))
                print(f"[MOVE] Approach X={X:.3f} Y={Y:.3f} Z={Z_APPROACH:.3f} yaw={yaw_deg:.1f}")
                try:
                    gripper_open()
                except Exception as e:
                    print("[WARN] gripper", e)
                time.sleep(1)
                call_move(X,Y,Z_APPROACH,(qx,qy,qz,qw))
                time.sleep(2)
                call_move(X,Y,Z_PICK,(qx,qy,qz,qw))
                time.sleep(2)
                try:
                    gripper_close()
                except Exception as e:
                    print("[WARN] gripper", e)
                time.sleep(2)
                call_move_pose(HIDE_POSE)

    finally:
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
