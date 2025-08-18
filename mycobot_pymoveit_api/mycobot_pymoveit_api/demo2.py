#!/usr/bin/env python3
import os, time, math, json, base64
from typing import Tuple, Dict
from collections import deque
import cv2
import numpy as np
import requests

# ----------------- CONFIG -----------------

CAM_INDEX = 0

# Heights (m)
Z_APPROACH = 0.12
Z_PICK     = 0.08

# Safe tilt quaternion
SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

# Rotate gripper to align with USB long axis
GRIPPER_YAW_OFFSET_DEG = -90.0

# Rounding when sending
XY_DECIMALS = 3

# Workspace guards (m)
MAX_RADIUS = 0.28
BASE_BUFFER_RADIUS = 0.05
ALLOW_NEGATIVE_Y = False

MOVE_URL = "http://localhost:8080/move"
GRIPPER_OPEN_URL  = "http://localhost:8080/gripper/open"
GRIPPER_CLOSE_URL = "http://localhost:8080/gripper/close"

HOME_POSE = {
    "posX": 0.06, "posY": 0.079, "posZ": 0.411,
    "rotX": 0.03, "rotY": -0.382, "rotZ": 0.001, "rotW": 1.0
}

# Azure OpenAI env
AZURE_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VER    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Image-right ↔ robot X polarity
IMAGE_RIGHT_IS_POSITIVE_X = -1   # +1:right=+X,  -1:right=-X (your setup)

# Optional flips (generally leave as-is)
INVERT_X = False
INVERT_Y = False
YAW_SIGN = +1

# Adjustable X-origin column (pixels) & trims
CFG_FILE = "vision_origin.json"
ORIGIN_U_PX = None
NUDGE_STEP = 5  # px

# Per-axis calibration (meters)
X_BIAS = 0.000   # add/subtract a few mm
Y_BIAS = 0.000
X_SCALE = 1.000  # 1% ≈ 2–3 mm at 0.25 m
Y_SCALE = 1.000

# Averaging (to smooth 1-frame noise)
AVG_WINDOW = 3
avg_buf = deque(maxlen=AVG_WINDOW)

# ----------------- UTILS -----------------

def load_cfg():
    global ORIGIN_U_PX, IMAGE_RIGHT_IS_POSITIVE_X, X_BIAS, Y_BIAS, X_SCALE, Y_SCALE
    try:
        with open(CFG_FILE, "r") as f:
            cfg = json.load(f)
        ORIGIN_U_PX = int(cfg.get("origin_u_px", ORIGIN_U_PX or 0))
        IMAGE_RIGHT_IS_POSITIVE_X = int(cfg.get("image_right_pos_x", IMAGE_RIGHT_IS_POSITIVE_X))
        X_BIAS = float(cfg.get("x_bias", X_BIAS))
        Y_BIAS = float(cfg.get("y_bias", Y_BIAS))
        X_SCALE = float(cfg.get("x_scale", X_SCALE))
        Y_SCALE = float(cfg.get("y_scale", Y_SCALE))
        print(f"[CFG] Loaded {CFG_FILE}: origin_u={ORIGIN_U_PX}, right→X={IMAGE_RIGHT_IS_POSITIVE_X}, "
              f"Xbias={X_BIAS:.3f}, Ybias={Y_BIAS:.3f}, Xscale={X_SCALE:.3f}, Yscale={Y_SCALE:.3f}")
    except Exception:
        print("[CFG] No saved config; using defaults.")

def save_cfg():
    cfg = {
        "origin_u_px": int(ORIGIN_U_PX),
        "image_right_pos_x": int(IMAGE_RIGHT_IS_POSITIVE_X),
        "x_bias": float(X_BIAS),
        "y_bias": float(Y_BIAS),
        "x_scale": float(X_SCALE),
        "y_scale": float(Y_SCALE),
    }
    try:
        with open(CFG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[CFG] Saved -> {CFG_FILE}")
    except Exception as e:
        print("[CFG] Save failed:", e)

def b64_jpeg(img_bgr, quality=92) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok: raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def yaw_about_z(q: Tuple[float,float,float,float], yaw_rad: float):
    x1, y1, z1, w1 = q
    cz, sz = math.cos(yaw_rad/2.0), math.sin(yaw_rad/2.0)
    x2, y2, z2, w2 = 0.0, 0.0, sz, cz
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1
    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    return (x, y, z, w)

# def axis_adjust(x_m: float, y_m: float, u_px: float, origin_u_px: float) -> Tuple[float,float]:
#     """Get X sign from which side of origin_u_px the target lies, then apply scale & bias.
#        Y is never mirrored unless you set INVERT_Y; only Y_SCALE/Y_BIAS are applied."""
#     X = -x_m if INVERT_X else x_m
#     Y = -y_m if INVERT_Y else y_m

#     side = 1.0 if (u_px - origin_u_px) >= 0.0 else -1.0
#     sign = IMAGE_RIGHT_IS_POSITIVE_X * side
#     X = abs(X) * sign

#     # Per-axis trims
#     X = (X * X_SCALE) + X_BIAS
#     Y = (Y * Y_SCALE) + Y_BIAS
#     return X, Y


def axis_adjust(x_m: float, y_m: float) -> Tuple[float,float]:
    """
    Trust Azure's posX,posY as meters in robot frame.
    Only apply bias/scale trims. No image-based sign correction.
    """
    X = (x_m * X_SCALE) + X_BIAS
    Y = (y_m * Y_SCALE) + Y_BIAS
    return X, Y

def call_move_pose(pose: dict, timeout=15.0):
    body = dict(pose)
    for k in ("posX","posY","posZ"):
        body[k] = round(float(body[k]), XY_DECIMALS)
    print("[MOVE] ->", body)
    r = requests.post(MOVE_URL, json=body, timeout=timeout)
    if r.status_code >= 400:
        print("[MOVE][HTTP]", r.status_code, r.text[:300])
    r.raise_for_status()
    return r.json() if r.text else {"ok": True}

def call_move(x, y, z, q, timeout=15.0):
    pose = {
        "posX": float(x), "posY": float(y), "posZ": float(z),
        "rotX": float(q[0]), "rotY": float(q[1]),
        "rotZ": float(q[2]), "rotW": float(q[3]),
    }
    return call_move_pose(pose, timeout=timeout)

def gripper_open(timeout=10.0):
    print("[GRIPPER] open")
    r = requests.post(GRIPPER_OPEN_URL, timeout=timeout)
    if r.status_code >= 400: print("[GRIPPER][HTTP]", r.status_code, r.text[:300])
    r.raise_for_status()

def gripper_close(timeout=10.0):
    print("[GRIPPER] close")
    r = requests.post(GRIPPER_CLOSE_URL, timeout=timeout)
    if r.status_code >= 400: print("[GRIPPER][HTTP]", r.status_code, r.text[:300])
    r.raise_for_status()

def go_home():
    try: gripper_open()
    except Exception as e: print("[WARN] gripper open:", e)
    try: call_move_pose(HOME_POSE)
    except Exception as e: print("[WARN] home move:", e)

def call_azure_for_robot_xy(img_bgr) -> Dict[str, float]:
    img64 = b64_jpeg(img_bgr)
    system_prompt = (
        "You are a vision assistant for a myCobot_280 tabletop scene. "
        "Camera is opposite the robot. Robot frame:\n"
        " • Origin (0,0) at BACK-CENTER of the robot base (near TOP-CENTER of image).\n"
        " • Y+ is forward toward the camera (downward in the image).\n"
        " • X+ is LEFT OF THE IMAGE (Important).\n"
        " • X- is RIGHT OF THE IMAGE!(Important).\n"
        "Target: a USB stick ~50mm x 18mm on the table, within 0.28 m radius. "
        "Ignore the robot; find the USB. "
        "Return STRICT JSON: {\"posX\":<float>,\"posY\":<float>,\"yaw_deg\":<float>,\"u\":<float>,\"v\":<float>} "
        "with posX/posY in METERS, yaw_deg clockwise relative to +X, and (u,v) pixel center."
    )
    user_text = "ONLY JSON. Example: {\"posX\":0.105,\"posY\":0.240,\"yaw_deg\":-90.0,\"u\":512.0,\"v\":360.0}"
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VER}"
    headers = {"Content-Type":"application/json", "api-key":AZURE_API_KEY}
    payload = {
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":[
                {"type":"text","text":user_text},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}}
            ]}
        ],
        "temperature": 0.0,
        "response_format": {"type":"json_object"}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    det = json.loads(resp.json()["choices"][0]["message"]["content"])
    for k in ("posX","posY","yaw_deg","u","v"):
        if k not in det: raise ValueError(f"Azure response missing key: {k}")
    return det

def overlay_info(img, posX, posY, yaw_deg, u, v, origin_u_px):
    h, w = img.shape[:2]
    origin_u_vis, origin_v = w//2, int(0.11*h)
    cv2.circle(img,(origin_u_vis,origin_v),6,(255,0,0),-1)
    cv2.arrowedLine(img,(origin_u_vis,origin_v),(origin_u_vis+80,origin_v),(255,0,0),2,cv2.LINE_AA,0,0.25)
    cv2.arrowedLine(img,(origin_u_vis,origin_v),(origin_u_vis,origin_v+80),(0,255,0),2,cv2.LINE_AA,0,0.25)

    x0 = int(round(origin_u_px))
    cv2.line(img, (x0, 0), (x0, h), (255,255,0), 1)
    cv2.putText(img, f"origin_u={x0}px", (x0+6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    u_i, v_i = int(round(u)), int(round(v))
    cv2.circle(img,(u_i,v_i),8,(0,0,255),-1)
    ang_img = -math.radians(yaw_deg)
    u2 = int(round(u_i + 60*math.cos(ang_img)))
    v2 = int(round(v_i + 60*math.sin(ang_img)))
    cv2.arrowedLine(img,(u_i,v_i),(u2,v2),(0,255,0),2,cv2.LINE_AA,0,0.30)

    txt = f"(X,Y)=({posX:.3f},{posY:.3f}) m, yaw={yaw_deg:.1f}°"
    cv2.putText(img, txt, (20, h-24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    trims = f"Xbias={X_BIAS:+.3f}  Ybias={Y_BIAS:+.3f}  Xscale={X_SCALE:.3f}  Yscale={Y_SCALE:.3f}"
    cv2.putText(img, trims, (20, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

# ----------------- MAIN LOOP -----------------

def main():
    global ORIGIN_U_PX, IMAGE_RIGHT_IS_POSITIVE_X, X_BIAS, Y_BIAS, X_SCALE, Y_SCALE

    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT):
        raise RuntimeError("Missing Azure env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

    load_cfg()

    print("=== Vision pick (no homography) ===")
    print("Keys:")
    print("  c = capture & pick | v = preview only | r = home")
    print("  [ / ] = nudge X-origin left/right (5 px) | p = flip image-right↔X sign")
    print("  1/2 = X bias ±5mm | 3/4 = Y bias ±5mm | 5/6 = X scale ±1% | 7/8 = Y scale ±1%")
    print("  s = save trims+origin | q = quit")

    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            h, w = frame.shape[:2]
            if ORIGIN_U_PX is None: ORIGIN_U_PX = w // 2

            view = frame.copy()
            cv2.putText(view, "c=pick  v=preview  r=home  [ ]=nudge  p=flipX  1-8=trims  s=save  q=quit",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.line(view, (int(ORIGIN_U_PX), 0), (int(ORIGIN_U_PX), h), (255,255,0), 1)
            cv2.imshow("webcam", view)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): print("[RESET] Home…"); go_home(); continue
            if key == ord('['): ORIGIN_U_PX -= NUDGE_STEP; print(f"[ORIGIN] {ORIGIN_U_PX}"); continue
            if key == ord(']'): ORIGIN_U_PX += NUDGE_STEP; print(f"[ORIGIN] {ORIGIN_U_PX}"); continue
            if key == ord('p'): IMAGE_RIGHT_IS_POSITIVE_X *= -1; print(f"[POLARITY] {IMAGE_RIGHT_IS_POSITIVE_X}"); continue
            if key == ord('1'): X_BIAS -= 0.005; print(f"[TRIM] X_BIAS={X_BIAS:+.3f}"); continue
            if key == ord('2'): X_BIAS += 0.005; print(f"[TRIM] X_BIAS={X_BIAS:+.3f}"); continue
            if key == ord('3'): Y_BIAS -= 0.005; print(f"[TRIM] Y_BIAS={Y_BIAS:+.3f}"); continue
            if key == ord('4'): Y_BIAS += 0.005; print(f"[TRIM] Y_BIAS={Y_BIAS:+.3f}"); continue
            if key == ord('5'): X_SCALE *= 0.99;  print(f"[TRIM] X_SCALE={X_SCALE:.3f}"); continue
            if key == ord('6'): X_SCALE *= 1.01;  print(f"[TRIM] X_SCALE={X_SCALE:.3f}"); continue
            if key == ord('7'): Y_SCALE *= 0.99;  print(f"[TRIM] Y_SCALE={Y_SCALE:.3f}"); continue
            if key == ord('8'): Y_SCALE *= 1.01;  print(f"[TRIM] Y_SCALE={Y_SCALE:.3f}"); continue
            if key == ord('s'): save_cfg(); continue
            if key not in (ord('c'), ord('v')): continue

            img = frame.copy()
            try:
                det = call_azure_for_robot_xy(img)
                x_raw = float(det["posX"]); y_raw = float(det["posY"])
                yaw_deg = float(det["yaw_deg"]); u = float(det["u"]); v = float(det["v"])
                
                X, Y = axis_adjust(x_raw, y_raw)

                # Average a few frames to reduce flicker
                avg_buf.append((X, Y, yaw_deg, u, v))
                X, Y, yaw_deg, u, v = map(lambda a: sum(a)/len(a), zip(*avg_buf))

                print(f"[Azure] raw=({x_raw:.3f},{y_raw:.3f}) -> adj_avg=({X:.3f},{Y:.3f}), yaw={yaw_deg:.1f}°")

                overlay = img.copy()
                overlay_info(overlay, X, Y, yaw_deg, u, v, ORIGIN_U_PX)
                cv2.imshow("detection", overlay)
                cv2.imwrite("last_detection_noH.jpg", overlay)

                r = math.hypot(X, Y)
                if r < BASE_BUFFER_RADIUS: print(f"[SKIP] Inside 5 cm keep-out (r={r:.3f})."); continue
                if not ALLOW_NEGATIVE_Y and Y < 0.0: print(f"[SKIP] Y={Y:.3f} < 0."); continue
                if r > MAX_RADIUS: print(f"[SKIP] Outside radius {MAX_RADIUS} (r={r:.3f})."); continue

                if key == ord('c'):
                    gripper_open()
                    yaw_total_deg = (YAW_SIGN * yaw_deg) + GRIPPER_YAW_OFFSET_DEG
                    qx,qy,qz,qw = yaw_about_z(SAFE_Q, math.radians(yaw_total_deg))
                    print(f"[MOVE] yaw_total_deg={yaw_total_deg:.1f}°")

                    print("[MOVE] Approach…"); 
                    call_move(X, Y, Z_APPROACH, (qx,qy,qz,qw))
                    time.sleep(2.0)
                    print("[MOVE] Pick height…"); 
                    call_move(X, Y, Z_PICK, (qx,qy,qz,qw))
                    print("[WAIT] 2.0s…"); 
                    time.sleep(2.0)
                    gripper_close()
                    print("[WAIT] 5s…"); 
                    time.sleep(5.0)
                    print("[MOVE] lift"); 
                    call_move(X, Y, Z_APPROACH, (qx,qy,qz,qw))
                    time.sleep(2.0)
                    print("[HOME] …"); call_move_pose(HOME_POSE)
                    print("[DONE]")

            except Exception as e:
                print("[ERROR]", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
