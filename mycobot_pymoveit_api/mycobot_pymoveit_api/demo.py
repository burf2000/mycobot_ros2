#!/usr/bin/env python3
import os, time, math, json, base64
from typing import Tuple, Dict
import numpy as np
import cv2
import requests

# ----------------- CONFIG -----------------

CAM_INDEX = 0

H_FILE = os.path.expanduser(
    "~/ros2_ws/src/mycobot_ros2/mycobot_pymoveit_api/mycobot_pymoveit_api/H_table.npy"
)

Z_APPROACH = 0.10
Z_PICK     = 0.08

SAFE_Q = (-0.491, -0.503, 0.520, 0.483)

MOVE_URL = "http://localhost:8080/move"

# Axis debug toggles
SWAP_XY  = False     # we’re swapping because your frame showed X using the long edge
INVERT_X = True
INVERT_Y = False
YAW_SIGN = +1

# NEW: fixed extra gripper rotation (clockwise = negative degrees here)
GRIPPER_YAW_OFFSET_DEG = -90.0

# Round XY we send to the move API
XY_DECIMALS = 3

# Workspace guard
MAX_RADIUS = 0.28
ALLOW_NEGATIVE_Y = False

# ------------------------------------------

if not os.path.exists(H_FILE):
    raise FileNotFoundError(f"H file not found: {H_FILE}")
H = np.load(H_FILE)
Hinv = np.linalg.inv(H)
print("[Info] Loaded H from:", H_FILE)

def b64_jpeg(img_bgr, quality=92) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
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

def pixel_to_xy(u: float, v: float, H: np.ndarray) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """Return (X,Y) after toggles AND (X_raw,Y_raw) direct from H."""
    pix = np.array([u, v, 1.0], dtype=np.float64)
    w = H @ pix
    X_raw = w[0] / w[2]
    Y_raw = w[1] / w[2]

    X, Y = X_raw, Y_raw
    if SWAP_XY: X, Y = Y, X
    if INVERT_X: X = -X
    if INVERT_Y: Y = -Y
    return (X, Y), (X_raw, Y_raw)

def xy_to_pixel(X: float, Y: float) -> Tuple[int,int]:
    Xi, Yi = X, Y
    if INVERT_Y: Yi = -Yi
    if INVERT_X: Xi = -Xi
    if SWAP_XY:  Xi, Yi = Yi, Xi
    P = np.array([Xi, Yi, 1.0], dtype=np.float64)
    q = Hinv @ P
    return int(round(q[0]/q[2])), int(round(q[1]/q[2]))

def call_move(x, y, z, q, timeout=10.0):
    # Round X/Y to 3 decimals before sending
    xr = round(float(x), XY_DECIMALS)
    yr = round(float(y), XY_DECIMALS)
    zr = round(float(z), XY_DECIMALS)
    body = {"posX": xr, "posY": yr, "posZ": zr,
            "rotX": float(q[0]), "rotY": float(q[1]), "rotZ": float(q[2]), "rotW": float(q[3])}
    print("[MOVE] ->", body)
    r = requests.post(MOVE_URL, json=body, timeout=timeout)
    if r.status_code >= 400:
        print("[MOVE][HTTP]", r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json() if r.text else {"ok": True}

def call_azure_vision(img_bgr) -> Dict[str, float]:
    img64 = b64_jpeg(img_bgr)
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VER}"
    headers = {"Content-Type":"application/json", "api-key":AZURE_API_KEY}
    system_prompt = (
        "You are a vision assistant for a myCobot_280 tabletop scene. "
        "The robot and its gripper may be visible—IGNORE the robot entirely. "
        "Focus ONLY on the A4 white paper area on the table. "
        "Find the most salient small, non-white object ON THE PAPER (e.g., a red USB stick). "
        "Return STRICT JSON: {\"u\":<float>,\"v\":<float>,\"yaw_deg\":<float>} "
        "where (u,v) is the object's CENTER pixel (image origin top-left) and yaw_deg is the object's in-plane rotation "
        "clockwise about the camera Z axis. No extra keys, no text."
    )
    user_text = "Return ONLY JSON. Example: {\"u\":512.3,\"v\":321.8,\"yaw_deg\":0.0}"
    payload = {
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content": [
                {"type":"text","text":user_text},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img64}"}}
            ]}
        ],
        "temperature": 0.0,
        "response_format": {"type":"json_object"}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    det = json.loads(content)
    for k in ("u","v","yaw_deg"):
        if k not in det: raise ValueError(f"Azure response missing {k}")
    return det

def draw_axes(img):
    u0,v0 = xy_to_pixel(0.0, 0.0)
    ux,vx = xy_to_pixel(0.10, 0.00)
    uy,vy = xy_to_pixel(0.00, 0.10)
    cv2.circle(img,(u0,v0),6,(255,0,0),-1)
    cv2.arrowedLine(img,(u0,v0),(ux,vx),(255,0,0),2,cv2.LINE_AA,0,0.25)  # +X
    cv2.putText(img,"+X",(ux+6,vx-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    cv2.arrowedLine(img,(u0,v0),(uy,vy),(0,255,0),2,cv2.LINE_AA,0,0.25)  # +Y
    cv2.putText(img,"+Y",(uy+6,vy-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

def main():
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT):
        raise RuntimeError("Missing Azure env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")

    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAM_INDEX}")

    try:
        print("Press 'c' to capture & move; 'v' capture only; 'q' quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            view = frame.copy()
            draw_axes(view)
            cv2.putText(view, "c=capture+move, v=capture, q=quit",
                        (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("webcam", view)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key not in (ord('c'), ord('v')):
                continue

            img = frame.copy()
            try:
                det = call_azure_vision(img)
                u = float(det["u"]); v = float(det["v"]); yaw_deg = float(det["yaw_deg"])
                print(f"[Azure] u={u:.1f}, v={v:.1f}, yaw={yaw_deg:.1f} deg")

                # Raw detection overlay
                dbg = img.copy()
                cv2.circle(dbg,(int(u),int(v)),8,(0,0,255),-1)
                ang_img = math.radians(-yaw_deg)  # image v grows downward
                x2 = int(u + 60*math.cos(ang_img))
                y2 = int(v + 60*math.sin(ang_img))
                cv2.arrowedLine(dbg,(int(u),int(v)),(x2,y2),(0,255,0),2,cv2.LINE_AA,0,0.30)
                cv2.putText(dbg, f"(u,v)=({int(round(u))},{int(round(v))})",
                            (int(u)+10, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Map to robot XY (final + raw)
                (X, Y), (X_raw, Y_raw) = pixel_to_xy(u, v, H)

                # Reproject final (X,Y) for error check
                uu, vv = xy_to_pixel(X, Y)
                cv2.circle(dbg,(uu,vv),6,(255,0,0),-1)
                pix_err = math.hypot(uu - u, vv - v)

                # Labels: round to 3 decimals in overlay
                Xr, Yr = round(X, XY_DECIMALS), round(Y, XY_DECIMALS)
                cv2.putText(dbg, f"(X,Y)=({Xr:.3f},{Yr:.3f}) m  err={pix_err:.1f}px",
                            (uu+10, vv+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                draw_axes(dbg)
                cv2.imshow("detection", dbg)
                cv2.imwrite("last_detection_debug.jpg", dbg)
                print(f"[Map] X={Xr:.3f}, Y={Yr:.3f} | err={pix_err:.1f}px")

                # Guards
                if not ALLOW_NEGATIVE_Y and Y < 0.0:
                    #print(f("[SKIP] Y={:.3f} < 0 (disallowed).").format(Y))
                    continue
                if math.hypot(X, Y) > MAX_RADIUS:
                    print(f"[SKIP] Outside radius {MAX_RADIUS} m."); continue

                # Compose orientation: Azure yaw + fixed gripper offset
                if key == ord('c'):
                    yaw_total_deg = (YAW_SIGN * yaw_deg) + GRIPPER_YAW_OFFSET_DEG
                    qx,qy,qz,qw = yaw_about_z(SAFE_Q, math.radians(yaw_total_deg))
                    print(f"[MOVE] Using yaw_total_deg={yaw_total_deg:.1f}")

                    print("[MOVE] Approach high…")
                    call_move(Xr, Yr, Z_APPROACH, (qx,qy,qz,qw))
                    time.sleep(0.2)
                    print("[MOVE] Drop to pick…")
                    call_move(Xr, Yr, Z_PICK, (qx,qy,qz,qw))
                    print("[OK] Sent.")

            except Exception as e:
                print("[ERROR]", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
