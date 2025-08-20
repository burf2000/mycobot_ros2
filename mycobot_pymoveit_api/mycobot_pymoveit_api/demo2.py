#!/usr/bin/env python3
import os, time, math, json, base64
from typing import Tuple, Dict, Optional
from collections import deque
import cv2
import numpy as np
import requests

# ----------------- AZURE OPENAI ENV -----------------
AZURE_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_API_KEY    = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VER    = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ----------------- CONFIG -----------------

CAM_INDEX = 0

# Heights (m)
Z_APPROACH = 0.12
Z_PICK     = 0.08

# Safe tilt quaternion
SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

# Gripper yaw control
FORCE_HORIZONTAL_YAW = False            # keep fingers horizontal to camera
GRIPPER_WORLD_YAW_DEG = -90.0          # yaw when forcing horizontal
GRIPPER_YAW_STEP_DEG  = 5.0            # step with 9/0 keys

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

# Per-axis fine trims (meters) AFTER metric ROI mapping
X_BIAS = 0.000
Y_BIAS = 0.000
X_SCALE = 1.000
Y_SCALE = 1.000

# Averaging (to smooth 1-frame noise)
AVG_WINDOW = 3
avg_buf = deque(maxlen=AVG_WINDOW)

# --------- METRIC ROI (the 560mm x 280mm box) ----------
ROI_W_MM = 560.0
ROI_H_MM = 280.0
MM_PER_PX = 0.5                 # 0.5 mm/px → 1120x560 px
M_PER_PX  = MM_PER_PX / 1000.0
ROI_W_PX  = int(round(ROI_W_MM / MM_PER_PX))  # 1120
ROI_H_PX  = int(round(ROI_H_MM / MM_PER_PX))  # 560

# --- NEW: inner margins (mm) to remove tape thickness (use your real values) ---
MARGIN_LEFT_MM   = 10.0
MARGIN_RIGHT_MM  = 10.0
MARGIN_TOP_MM    = 10.0
MARGIN_BOTTOM_MM = 10.0

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

def wrap_deg180(a): return ((a + 180.0) % 360.0) - 180.0

def prefer_horizontal_unwound(yaw_deg):
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

# ----------------- AZURE VISION (full image) -----------------

def call_azure_for_robot_xy(img_bgr) -> Dict[str, float]:
    img64 = b64_jpeg(img_bgr)
    system_prompt = (
        "You are a vision assistant for a myCobot_280 tabletop scene. "
        "The camera faces the robot from the front; the robot base is near the TOP of the image and the table extends downward.\n\n"
        "ROBOT FRAME (not image pixels):\n"
        " • Origin (0,0): back-center of the robot base (near TOP-CENTER of image).\n"
        " • +Y: toward the camera → DOWN in the image.\n"
        " • +X: robot's LEFT      → LEFT side of the image.\n"
        " • -X: robot's RIGHT     → RIGHT side of the image.\n\n"
        "WORKSPACE: X ∈ [-0.28,+0.28] m, Y ∈ [0.00,+0.28] m.\n"
        "TASK: Detect ONLY a red 2×4 LEGO brick. OUTPUT STRICT JSON {\"posX\":f,\"posY\":f,\"yaw_deg\":f,\"u\":f,\"v\":f}."
    )
    user_text = "Return only the JSON object."
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT):
        raise RuntimeError("Missing Azure env")
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
    resp = requests.post(url, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    det = json.loads(resp.json()["choices"][0]["message"]["content"])
    for k in ("posX","posY","yaw_deg","u","v"):
        if k not in det: raise ValueError(f"Azure response missing key: {k}")
    return det

# ----------------- ROI DETECTION & WARP -----------------

def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def find_quad_by_contour(img_bgr) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best, best_area = None, 0.0
    h, w = gray.shape[:2]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (w*h)*0.05:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4: 
            continue
        rect = cv2.minAreaRect(approx)
        cw, ch = rect[1]
        if cw < 1 or ch < 1: 
            continue
        ratio = max(cw,ch)/max(1.0,min(cw,ch))
        if 1.5 <= ratio <= 2.7 and area > best_area:
            best, best_area = approx, area
    return order_quad(best) if best is not None else None

def find_quad_by_hough(img_bgr) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=edges.shape[1]//3, maxLineGap=20)
    if lines is None: return None
    vertical = []; horizontal = []
    for x1,y1,x2,y2 in lines[:,0]:
        dx, dy = x2-x1, y2-y1
        if abs(dx) < abs(dy)*0.5:      vertical.append((x1,y1,x2,y2))
        elif abs(dy) < abs(dx)*0.5:    horizontal.append((x1,y1,x2,y2))
    if len(vertical) < 2 or len(horizontal) < 2: return None
    v_sorted = sorted(vertical, key=lambda L: (L[0]+L[2])/2)
    h_sorted = sorted(horizontal, key=lambda L: (L[1]+L[3])/2)
    left  = v_sorted[0];  right = v_sorted[-1]
    top   = h_sorted[0];  bot   = h_sorted[-1]

    def line_from_pts(p):
        x1,y1,x2,y2=p
        A = np.array([[x1,1],[x2,1]], dtype=float)
        b = np.array([y1,y2], dtype=float)
        m,c = np.linalg.lstsq(A,b,rcond=None)[0]
        return m,c  # y = m x + c

    mL,cL = line_from_pts(left);  mR,cR = line_from_pts(right)
    mT,cT = line_from_pts(top);   mB,cB = line_from_pts(bot)

    def inter(m1,c1,m2,c2):
        x = (c2-c1)/(m1-m2+1e-9); y = m1*x + c1
        return [x,y]

    TL = inter(mL,cL,mT,cT); TR = inter(mR,cR,mT,cT)
    BR = inter(mR,cR,mB,cB); BL = inter(mL,cL,mB,cB)
    quad = np.array([TL,TR,BR,BL], dtype=np.float32)
    return quad

def find_quad_by_band_scan(img_bgr) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    top_band    = gray[0:int(0.35*h), :]
    bottom_band = gray[int(0.55*h):, :]
    left_band   = gray[:, 0:int(0.35*w)]
    right_band  = gray[:, int(0.65*w):]

    top_idx = np.argmin(top_band.mean(axis=1))
    bot_idx = np.argmin(bottom_band.mean(axis=1)) + int(0.55*h)
    left_idx = np.argmin(left_band.mean(axis=0))
    right_idx = np.argmin(right_band.mean(axis=0)) + int(0.65*w)

    if bot_idx - top_idx < h*0.3 or right_idx - left_idx < w*0.3: return None

    TL = [left_idx,  top_idx];  TR = [right_idx, top_idx]
    BR = [right_idx, bot_idx];  BL = [left_idx,  bot_idx]
    return np.array([TL,TR,BR,BL], dtype=np.float32)

def find_black_box_quad(img_bgr) -> Optional[np.ndarray]:
    quad = find_quad_by_contour(img_bgr)
    if quad is not None: return quad
    quad = find_quad_by_hough(img_bgr)
    if quad is not None: return quad
    quad = find_quad_by_band_scan(img_bgr)
    return quad

def warp_to_metric_roi(img_bgr, quad_img_px: np.ndarray):
    dst = np.array([
        [0, 0],
        [ROI_W_PX-1, 0],
        [ROI_W_PX-1, ROI_H_PX-1],
        [0, ROI_H_PX-1]
    ], dtype=np.float32)
    H  = cv2.getPerspectiveTransform(quad_img_px.astype(np.float32), dst)
    Hinv = np.linalg.inv(H)
    roi = cv2.warpPerspective(img_bgr, H, (ROI_W_PX, ROI_H_PX))
    return roi, H, Hinv

# ----------------- VISION HELPERS -----------------

def hsv_red_centroid(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,   90, 80), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 90, 80), (180,255,255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None, 0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 60: return None, None, 0
    M = cv2.moments(c)
    if M["m00"] == 0: return None, None, 0
    u = int(M["m10"]/M["m00"])
    v = int(M["m01"]/M["m00"])
    return u, v, int(area)

def clamp_uv(u, v, w, h):
    u = max(0, min(w-1, float(u)))
    v = max(0, min(h-1, float(v)))
    return int(round(u)), int(round(v))

def image_vec_from_world_yaw(yaw_deg, length_px=60):
    th = math.radians(float(yaw_deg))
    du = -math.cos(th) * length_px   # +X world -> left
    dv =  math.sin(th) * length_px   # +Y world -> down
    return int(round(du)), int(round(dv))

# ----------------- AZURE (YAW-ONLY MODE) -----------------

def azure_yaw_for_roi(roi_bgr, u_roi_px: float, v_roi_px: float, x_m: float, y_m: float) -> float:
    print("Calling Azure")

    try:
        img64 = b64_jpeg(roi_bgr)
        hint = {
            "u_px": float(u_roi_px),
            "v_px": float(v_roi_px),
            "posX_m": float(x_m),
            "posY_m": float(y_m),
            "roi_width_px": ROI_W_PX,
            "roi_height_px": ROI_H_PX,
            "mm_per_px": MM_PER_PX
        }
        system_prompt = (
            "You receive a rectified top-down ROI of a 560mm x 280mm workspace. "
            "The ROI is metric: +X leftwards, +Y downwards. "
            "Origin is top-center; X∈[-0.28,+0.28]m, Y∈[0.00,+0.28]m. "
            "Determine the brick yaw (clockwise from +X) near the provided centroid. "
            "Return STRICT JSON {\"yaw_deg\": <float>} only."
        )
        user_text = "Use the centroid & meters. Output only {\"yaw_deg\": number}."
        url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VER}"
        headers = {"Content-Type":"application/json", "api-key":AZURE_API_KEY}
        payload = {
            "messages": [
                {"role":"system","content":system_prompt},
                {"role":"user","content":[
                    {"type":"text","text":json.dumps(hint)},
                    {"type":"text","text":user_text},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}}
                ]}
            ],
            "temperature": 0.0,
            "response_format": {"type":"json_object"}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=35)
        resp.raise_for_status()
        obj = json.loads(resp.json()["choices"][0]["message"]["content"])
        return float(obj.get("yaw_deg", -90.0))
    except Exception as e:
        print("[WARN] Azure yaw fallback:", e)
        return -90.0

# ----------------- OVERLAY -----------------

def overlay_info(img, u_img_overlay, v_img_overlay, yaw_used_deg, origin_u_px,
                 x_m: float, y_m: float, u_model=None, v_model=None):
    h, w = img.shape[:2]

    # Visual axes
    origin_u_vis, origin_v = w // 2, int(0.11 * h)
    cv2.circle(img, (origin_u_vis, origin_v), 6, (255, 0, 0), -1)
    cv2.arrowedLine(img, (origin_u_vis, origin_v), (origin_u_vis + 80, origin_v), (255, 0, 0), 2, cv2.LINE_AA, 0, 0.25)
    cv2.arrowedLine(img, (origin_u_vis, origin_v), (origin_u_vis, origin_v + 80), (0, 255, 0), 2, cv2.LINE_AA, 0, 0.25)

    # Robot X-origin column
    x0 = int(round(origin_u_px))
    cv2.line(img, (x0, 0), (x0, h), (255, 255, 0), 1)
    cv2.putText(img, f"origin_u={x0}px  (Image-left=+X, right=-X)", (x0 + 6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # Azure (u,v) in ORANGE
    if u_model is not None and v_model is not None:
        um, vm = clamp_uv(u_model, v_model, w, h)
        cv2.circle(img, (um, vm), 8, (0,165,255), -1)  # orange

    # ROI/HSV backprojected point in YELLOW
    if u_img_overlay is not None and v_img_overlay is not None:
        uh, vh = clamp_uv(u_img_overlay, v_img_overlay, w, h)
        cv2.circle(img, (uh, vh), 8, (0, 255, 255), -1)  # yellow
        du, dv = image_vec_from_world_yaw(yaw_used_deg, length_px=60)
        cv2.arrowedLine(img, (uh, vh), (uh + du, vh + dv), (0, 255, 0), 2, cv2.LINE_AA, 0, 0.30)

    txt1 = f"(X,Y)=({x_m:+.3f},{y_m:+.3f}) m   yaw_used={yaw_used_deg:+.1f}°"
    cv2.putText(img, txt1, (18, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# ----------------- MAIN LOOP -----------------

def main():
    global ORIGIN_U_PX, IMAGE_RIGHT_IS_POSITIVE_X, X_BIAS, Y_BIAS, X_SCALE, Y_SCALE
    global FORCE_HORIZONTAL_YAW, GRIPPER_WORLD_YAW_DEG

    load_cfg()

    print("=== Vision pick (metric ROI; inner margins) ===")
    print("Keys:")
    print("  c = capture & pick | v = preview only | r = home")
    print("  [ / ] = nudge X-origin  | p = flip image-right↔X sign")
    print("  1/2=X bias  3/4=Y bias  5/6=X scale  7/8=Y scale  s=save")
    print("  g = toggle force-horizontal-yaw  |  9/0 = yaw -/+ 5°")
    print("  q = quit")


    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cap = cv2.VideoCapture(CAM_INDEX)
    ok, tmp = cap.read()
    if ok:
        h, w = tmp.shape[:2]
        scale = 1.8  # <- tweak (e.g., 1.5, 2.0)
        cv2.resizeWindow("webcam", int(w * scale), int(h * scale))

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
            cv2.putText(view, "c=pick  v=preview  r=home s=save  q=quit",
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
                # ---- A) Find and warp the 560x280 box to metric ROI
                quad = find_black_box_quad(img)
                if quad is None:
                    print("[WARN] Box not found; using Azure raw (u,v) only for overlay.")
                    roi = H = Hinv = None
                else:
                    roi, H, Hinv = warp_to_metric_roi(img, quad)

                # ---- B) Use INNER margins (remove tape) for centroid & meters
                u_roi = v_roi = None
                X = Y = None
                u_img_overlay = v_img_overlay = None

                if roi is not None:
                    ml = int(round(MARGIN_LEFT_MM   / MM_PER_PX))
                    mr = int(round(MARGIN_RIGHT_MM  / MM_PER_PX))
                    mt = int(round(MARGIN_TOP_MM    / MM_PER_PX))
                    mb = int(round(MARGIN_BOTTOM_MM / MM_PER_PX))

                    x0i = ml
                    x1i = ROI_W_PX - mr
                    y0i = mt
                    y1i = ROI_H_PX - mb
                    x0i = max(0, min(ROI_W_PX-2, x0i))
                    x1i = max(x0i+2, min(ROI_W_PX, x1i))
                    y0i = max(0, min(ROI_H_PX-2, y0i))
                    y1i = max(y0i+2, min(ROI_H_PX, y1i))

                    roi_inner = roi[y0i:y1i, x0i:x1i].copy()
                    u_in, v_in, _ = hsv_red_centroid(roi_inner)
                    if u_in is not None:
                        # Convert inner (u,v) to full ROI coords
                        u_roi = x0i + u_in
                        v_roi = y0i + v_in

                        # Origin for X is INNER center on the top edge
                        inner_center_u = (x0i + x1i) / 2.0
                        # X positive LEFT  → X = (center - u)*M_PER_PX
                        X = ((inner_center_u - u_roi) * M_PER_PX) * X_SCALE + X_BIAS
                        # Y=0 at inner top edge (y0i)
                        Y = ((v_roi - y0i) * M_PER_PX) * Y_SCALE + Y_BIAS

                        # Back-project to original image for overlay
                        if Hinv is not None:
                            pt = np.array([[u_roi, v_roi, 1.0]], dtype=np.float32).T
                            uvw = Hinv @ pt
                            u_img_overlay = float(uvw[0]/uvw[2])
                            v_img_overlay = float(uvw[1]/uvw[2])

                        # Draw inner rectangle (debug)
                        cv2.rectangle(roi, (x0i,y0i), (x1i,y1i), (0,255,255), 2)

                # ---- C) Get yaw
                yaw_cmd_deg = prefer_horizontal_unwound(GRIPPER_WORLD_YAW_DEG)  # default
                if roi is not None and u_roi is not None and not FORCE_HORIZONTAL_YAW:
                    yaw_deg_model = azure_yaw_for_roi(roi, u_roi, v_roi, X, Y)
                    yaw_cmd_deg = prefer_horizontal_unwound(YAW_SIGN * yaw_deg_model)

                # Also fetch Azure full-image (u,v) for debugging overlay
                u_model = v_model = None
                try:
                    det_full = call_azure_for_robot_xy(img)
                    u_model = float(det_full["u"]); v_model = float(det_full["v"])
                    if X is None or Y is None:
                        # LAST resort for position (less reliable)
                        X = float(det_full["posX"]) * X_SCALE + X_BIAS
                        Y = float(det_full["posY"]) * Y_SCALE + Y_BIAS
                except Exception as e:
                    print("[INFO] Azure full-image (u,v) skipped:", e)

                if X is None or Y is None:
                    raise RuntimeError("No position available.")

                print(f"[ROI] inner-margins(mm) L{MARGIN_LEFT_MM}/R{MARGIN_RIGHT_MM}/T{MARGIN_TOP_MM}/B{MARGIN_BOTTOM_MM}  "
                      f"u_roi,v_roi={u_roi},{v_roi} -> (X,Y)=({X:.3f},{Y:.3f}) m  yaw_cmd={yaw_cmd_deg:.1f}°")

                # ---- D) Overlay
                overlay = img.copy()
                overlay_info(overlay, u_img_overlay, v_img_overlay, yaw_cmd_deg, ORIGIN_U_PX,
                             X, Y, u_model=u_model, v_model=v_model)
                cv2.imshow("detection", overlay)
                cv2.imwrite("last_detection_metricROI.jpg", overlay)

                # ---- E) Guards then move
                r = math.hypot(X, Y)
                if r < BASE_BUFFER_RADIUS: print(f"[SKIP] Inside 5 cm keep-out (r={r:.3f})."); continue
                if not ALLOW_NEGATIVE_Y and Y < 0.0: print(f"[SKIP] Y={Y:.3f} < 0."); continue
                if r > MAX_RADIUS: print(f"[SKIP] Outside radius {MAX_RADIUS} (r={r:.3f})."); continue

                time.sleep(0.8)

                if key == ord('c'):
                    call_move_pose(HOME_POSE)
                    time.sleep(1.5)
                    gripper_open()
                    qx,qy,qz,qw = yaw_about_z(SAFE_Q, math.radians(yaw_cmd_deg))
                    print(f"[MOVE] yaw_cmd_deg={yaw_cmd_deg:.1f}°")

                    print("[MOVE] Approach…")
                    call_move(X, Y, Z_APPROACH, (qx,qy,qz,qw))
                    time.sleep(1.8)
                    print("[MOVE] Pick height…")
                    call_move(X, Y, Z_PICK, (qx,qy,qz,qw))
                    print("[WAIT] 2.0s…")
                    time.sleep(3.0)
                    gripper_close()
                    time.sleep(3.0)
                    print("[HOME] …")
                    call_move_pose(HOME_POSE)
                    time.sleep(2.0)
                    call_move_pose(HIDE_POSE)
                    print("[DONE]")

            except Exception as e:
                print("[ERROR]", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
