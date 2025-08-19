#!/usr/bin/env python3
import os, time, math, json, base64
from typing import Tuple, Dict
from collections import deque
import cv2
import numpy as np
import requests

# ----------------- AZURE OPENAI ENV -----------------
AZURE_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://ffhg-ai-service.cognitiveservices.azure.com").rstrip("/")
AZURE_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY", "A3LrQXp6s3Jrg7oevQkNy4i4wyC25imawvsqg1Hp3j69UODsNYZ2JQQJ99BAACmepeSXJ3w3AAAAACOGuuil")
AZURE_API_VER    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ----------------- CONFIG -----------------

CAM_INDEX = 0

# Heights (m)
Z_APPROACH = 0.12
Z_PICK     = 0.08

# Safe tilt quaternion
SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

# Original offset (used only if we follow model yaw)
GRIPPER_YAW_OFFSET_DEG = -90.0

# Gripper yaw control
FORCE_HORIZONTAL_YAW = True            # keep fingers horizontal to camera
GRIPPER_WORLD_YAW_DEG = -90.0          # commanded world yaw when forcing horizontal
GRIPPER_YAW_STEP_DEG  = 5.0            # step for live tuning with keys 9/0

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

HIDE_POSE = {
    "posX": 0.06, "posY": 0.005, "posZ": 0.444,
    "rotX": 0.352, "rotY": -0.354, "rotZ": -0.143, "rotW": 0.854
}

# Image-right ↔ robot X polarity (kept for compatibility)
IMAGE_RIGHT_IS_POSITIVE_X = -1   # +1:right=+X,  -1:right=-X

# Optional flips (generally leave as-is)
INVERT_X = False
INVERT_Y = False
YAW_SIGN = +1

# Adjustable X-origin column (pixels) & trims
CFG_FILE = "vision_origin.json"
ORIGIN_U_PX = None
NUDGE_STEP = 5  # px

# Per-axis calibration (meters)
X_BIAS = 0.000
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

def axis_adjust(x_m: float, y_m: float) -> Tuple[float,float]:
    """Trust Azure's posX,posY; only apply bias/scale trims."""
    X = (x_m * X_SCALE) + X_BIAS
    Y = (y_m * Y_SCALE) + Y_BIAS
    return X, Y

def wrap_deg180(a):
    """Wraps angle to (-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0

def prefer_horizontal_unwound(yaw_deg):
    """
    Keep the finger opening axis horizontal but prefer |yaw| <= 90.
    0° and ±180° are equivalent for a parallel gripper; choose the gentler wrist twist.
    """
    y = wrap_deg180(yaw_deg)
    if y > 90.0:  y -= 180.0
    if y < -90.0: y += 180.0
    return y

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
    try: call_move_pose(HIDE_POSE)
    except Exception as e: print("[WARN] home move:", e)

# ----------------- AZURE VISION -----------------

def call_azure_for_robot_xy(img_bgr) -> Dict[str, float]:
    img64 = b64_jpeg(img_bgr)

    system_prompt = (
        "You are a vision assistant for a myCobot_280 tabletop scene. "
        "The camera faces the robot from the front, so the robot base is near the TOP of the image and the table extends downward.\n\n"
        "ROBOT FRAME (do NOT confuse with image pixels):\n"
        " • Origin (0,0): back-center of the robot base (near TOP-CENTER of image).\n"
        " • +Y: forward toward the camera  → DOWN in the image.\n"
        " • -Y: away from camera          → UP in the image.\n"
        " • +X: robot's LEFT              → LEFT side of the image.\n"
        " • -X: robot's RIGHT             → RIGHT side of the image.\n\n"
        "WORKSPACE BOUNDS (physical 280 mm black box):\n"
        " • X ∈ [-0.28, +0.28] meters (560 mm total width).\n"
        " • Y ∈ [ 0.00, +0.28] meters (280 mm toward the camera; no negative Y).\n\n"
        "TASK: Detect ONLY a red 2×4 LEGO brick (~16×32 mm) lying flat on the table, within the bounds above. Ignore the robot and tools.\n\n"
        "OUTPUT (STRICT JSON only): {\"posX\":<float>, \"posY\":<float>, \"yaw_deg\":<float>, \"u\":<float>, \"v\":<float>}.\n"
        " • posX,posY: meters in the ROBOT FRAME.  • yaw_deg: CLOCKWISE from +X.  • (u,v): image pixel center (u rightward, v downward).\n\n"
        "MANDATORY SELF-CHECKS BEFORE ANSWERING:\n"
        " 1) Pixel-side rule: if the brick is LEFT of the image center, posX MUST be POSITIVE; if RIGHT of center, posX MUST be NEGATIVE.\n"
        " 2) Y MUST be ≥ 0 because +Y is toward the camera.\n"
        " 3) Clamp posX to [-0.28,+0.28] and posY to [0.00,+0.28].\n"
        " 4) Ensure (u,v) are inside the image frame.\n"
        " 5) Return ONLY the JSON object with those five keys—no extra text.\n"
    )

    user_text = "ONLY JSON. Example: {\"posX\":0.105,\"posY\":0.240,\"yaw_deg\":-90.0,\"u\":512.0,\"v\":360.0}"
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT):
        raise RuntimeError("Missing Azure env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")
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

# ----------------- OVERLAY / VISION QA -----------------

def hsv_red_centroid(img_bgr):
    """
    Find the largest red blob (HSV) and return its centroid (u,v) and area.
    Returns (None, None, 0) if not found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Two ranges for red hue
    m1 = cv2.inRange(hsv, (0,   90, 80), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 90, 80), (180,255,255))
    mask = cv2.bitwise_or(m1, m2)

    # Clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, 0

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 60:
        return None, None, 0

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None, 0

    u = int(M["m10"]/M["m00"])
    v = int(M["m01"]/M["m00"])
    return u, v, int(area)

def clamp_uv(u, v, w, h):
    """Keep pixel coordinates inside the image."""
    u = max(0, min(w-1, float(u)))
    v = max(0, min(h-1, float(v)))
    return int(round(u)), int(round(v))

def image_vec_from_world_yaw(yaw_deg, length_px=60):
    """
    Map world yaw (clockwise from +X) to an image-space vector.
    World +X → image left (−u), World +Y → image down (+v).
    => Δu = −cosθ * L,  Δv = +sinθ * L
    """
    th = math.radians(float(yaw_deg))
    du = -math.cos(th) * length_px
    dv =  math.sin(th) * length_px
    return int(round(du)), int(round(dv))

def overlay_info(img, posX, posY, yaw_used_deg, u_model, v_model, origin_u_px, u_hsv=None, v_hsv=None):
    h, w = img.shape[:2]

    # Visual axes (image u right, v down)
    origin_u_vis, origin_v = w // 2, int(0.11 * h)
    cv2.circle(img, (origin_u_vis, origin_v), 6, (255, 0, 0), -1)
    cv2.arrowedLine(img, (origin_u_vis, origin_v), (origin_u_vis + 80, origin_v), (255, 0, 0), 2, cv2.LINE_AA, 0, 0.25)
    cv2.arrowedLine(img, (origin_u_vis, origin_v), (origin_u_vis, origin_v + 80), (0, 255, 0), 2, cv2.LINE_AA, 0, 0.25)

    # Chosen image column for robot X-origin
    x0 = int(round(origin_u_px))
    cv2.line(img, (x0, 0), (x0, h), (255, 255, 0), 1)
    cv2.putText(img, f"origin_u={x0}px  (Image-left=+X, right=-X)", (x0 + 6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # Azure (u,v) in ORANGE
    um = vm = None
    if u_model is not None and v_model is not None:
        um, vm = clamp_uv(u_model, v_model, w, h)
        cv2.circle(img, (um, vm), 8, (0,165,255), -1)  # orange

    # HSV centroid in RED (truth for overlay)
    uh = vh = None
    if u_hsv is not None and v_hsv is not None:
        uh, vh = clamp_uv(u_hsv, v_hsv, w, h)
        cv2.circle(img, (uh, vh), 8, (0, 0, 255), -1)  # red

        # Green arrow from HSV dot using commanded yaw
        du, dv = image_vec_from_world_yaw(yaw_used_deg, length_px=60)
        cv2.arrowedLine(img, (uh, vh), (uh + du, vh + dv), (0, 255, 0), 2, cv2.LINE_AA, 0, 0.30)

    # Readouts
    txt1 = f"(X,Y)=({posX:+.3f},{posY:+.3f}) m   yaw_used={yaw_used_deg:+.1f}°"
    txt2 = "HSV u,v=(" + ("none" if uh is None else f"{uh},{vh}") + ")"
    txt3 = "Azure u,v=(" + ("none" if um is None else f"{um},{vm}") + ")"
    trims = f"Xbias={X_BIAS:+.3f}  Ybias={Y_BIAS:+.3f}  Xscale={X_SCALE:.3f}  Yscale={Y_SCALE:.3f}"

    cv2.putText(img, txt1, (18, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f"{txt2}   {txt3}   {trims}", (18, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

# ----------------- MAIN LOOP -----------------

def main():
    global ORIGIN_U_PX, IMAGE_RIGHT_IS_POSITIVE_X, X_BIAS, Y_BIAS, X_SCALE, Y_SCALE
    global FORCE_HORIZONTAL_YAW, GRIPPER_WORLD_YAW_DEG

    load_cfg()

    print("=== Vision pick (no homography) ===")
    print("Keys:")
    print("  c = capture & pick | v = preview only | r = home")
    print("  [ / ] = nudge X-origin  | p = flip image-right↔X sign")
    print("  1/2=X bias  3/4=Y bias  5/6=X scale  7/8=Y scale  s=save")
    print("  g = toggle force-horizontal-yaw  |  9/0 = yaw -/+ 5°")
    print("  q = quit")

    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if ORIGIN_U_PX is None:
                ORIGIN_U_PX = w // 2

            view = frame.copy()
            cv2.putText(view, "c=pick  v=preview  r=home  [ ]=nudge  p=flipX  1-8=trims  g=fixYaw  9/0=yaw  s=save  q=quit",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
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
            if key == ord('g'):
                FORCE_HORIZONTAL_YAW = not FORCE_HORIZONTAL_YAW
                print(f"[YAW] FORCE_HORIZONTAL_YAW={FORCE_HORIZONTAL_YAW}")
                continue
            if key == ord('9'):
                GRIPPER_WORLD_YAW_DEG -= GRIPPER_YAW_STEP_DEG
                print(f"[YAW] GRIPPER_WORLD_YAW_DEG={GRIPPER_WORLD_YAW_DEG:+.1f}")
                continue
            if key == ord('0'):
                GRIPPER_WORLD_YAW_DEG += GRIPPER_YAW_STEP_DEG
                print(f"[YAW] GRIPPER_WORLD_YAW_DEG={GRIPPER_WORLD_YAW_DEG:+.1f}")
                continue

            if key not in (ord('c'), ord('v')): continue

            img = frame.copy()
            try:
                # 1) Azure model detection
                det = call_azure_for_robot_xy(img)
                x_raw = float(det["posX"]); y_raw = float(det["posY"])
                yaw_deg_model = float(det["yaw_deg"]); u_model = float(det["u"]); v_model = float(det["v"])

                # 2) Optional: compute HSV centroid for overlay truth
                u_hsv, v_hsv, area = hsv_red_centroid(img)

                # 3) Axis trims
                X, Y = axis_adjust(x_raw, y_raw)

                # 4) Decide yaw to command
                if FORCE_HORIZONTAL_YAW:
                    yaw_cmd_deg = prefer_horizontal_unwound(GRIPPER_WORLD_YAW_DEG)
                else:
                    yaw_cmd_deg = (YAW_SIGN * yaw_deg_model) + GRIPPER_YAW_OFFSET_DEG
                    yaw_cmd_deg = prefer_horizontal_unwound(yaw_cmd_deg)

                print(f"[Azure] raw=({x_raw:.3f},{y_raw:.3f}) -> adj=({X:.3f},{Y:.3f}), "
                      f"model_yaw={yaw_deg_model:.1f}°, cmd_yaw={yaw_cmd_deg:.1f}°, "
                      f"u_model={u_model:.1f}, v_model={v_model:.1f}, hsv={'none' if u_hsv is None else (u_hsv, v_hsv)}")

                # 5) Overlay (HSV dot + Azure dot + correct arrow)
                overlay = img.copy()
                overlay_info(overlay, X, Y, yaw_cmd_deg, u_model, v_model, ORIGIN_U_PX, u_hsv=u_hsv, v_hsv=v_hsv)
                cv2.imshow("detection", overlay)
                cv2.imwrite("last_detection_noH.jpg", overlay)

                # 6) Guards
                r = math.hypot(X, Y)
                if r < BASE_BUFFER_RADIUS: print(f"[SKIP] Inside 5 cm keep-out (r={r:.3f})."); continue
                if not ALLOW_NEGATIVE_Y and Y < 0.0: print(f"[SKIP] Y={Y:.3f} < 0."); continue
                if r > MAX_RADIUS: print(f"[SKIP] Outside radius {MAX_RADIUS} (r={r:.3f})."); continue

                time.sleep(1.0)

                # 7) Execute pick
                if key == ord('c'):
                    # call_move_pose(HOME_POSE)
                    time.sleep(2.0)
                    gripper_open()
                    qx,qy,qz,qw = yaw_about_z(SAFE_Q, math.radians(yaw_cmd_deg))
                    print(f"[MOVE] yaw_cmd_deg={yaw_cmd_deg:.1f}°")

                    print("[MOVE] Approach…")
                    call_move(X, Y, Z_APPROACH, (qx,qy,qz,qw))
                    time.sleep(2.0)
                    print("[MOVE] Pick height…")
                    call_move(X, Y, Z_PICK, (qx,qy,qz,qw))
                    print("[WAIT] 2.0s…")
                    time.sleep(2.0)
                    gripper_close()
                    print("[WAIT] 2s…")
                    time.sleep(2.0)
                    print("[HOME] …")
                    call_move_pose(HOME_POSE)
                    time.sleep(3.0)
                    call_move_pose(HIDE_POSE)
                    print("[DONE]")

            except Exception as e:
                print("[ERROR]", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
