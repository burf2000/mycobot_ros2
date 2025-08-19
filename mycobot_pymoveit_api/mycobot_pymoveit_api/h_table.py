#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ======================= CONFIG =======================

# Your setup: camera is on the opposite side of the table along +Y.
# In the camera image the ROBOT BASE is at the TOP of the frame.
CAMERA_OPPOSITE_Y = True

# A4 paper (meters)
PAPER_W = 0.210  # X axis (robot left↔right, short edge)
PAPER_H = 0.297  # Y axis (forward from robot, toward camera, long edge)

# Robot-frame target points (meters).
# (0,0) is the center of the BACK edge (robot side).
WORLD_POINTS = np.array([
    [-PAPER_W/2, 0.0],        # 1) Back-Left (BL)
    [ PAPER_W/2, 0.0],        # 2) Back-Right (BR)
    [ PAPER_W/2, PAPER_H],    # 3) Front-Right (FR)
    [-PAPER_W/2, PAPER_H],    # 4) Front-Left (FL)
], dtype=np.float32)

# ======================================================

def load_image(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # RGB for matplotlib

def corner_labels_for_camera_layout():
    # Click order with explicit side hints
    if CAMERA_OPPOSITE_Y:
        return [
            ("top-right",    "1) BL (Back-Left)",   "robot side"),
            ("top-left",     "2) BR (Back-Right)",  "robot side"),
            ("bottom-left",  "3) FR (Front-Right)", "camera side"),
            ("bottom-right", "4) FL (Front-Left)",  "camera side"),
        ]
    else:
        return [
            ("top-left",     "1) BL (Back-Left)",   "robot side"),
            ("top-right",    "2) BR (Back-Right)",  "robot side"),
            ("bottom-right", "3) FR (Front-Right)", "camera side"),
            ("bottom-left",  "4) FL (Front-Left)",  "camera side"),
        ]

def draw_click_guides(img_rgb, step_idx):
    h, w, _ = img_rgb.shape
    overlay = img_rgb.copy()
    corner_text = corner_labels_for_camera_layout()

    pos = {
        "top-left":     (20, 40),
        "top-right":    (w - 320, 40),
        "bottom-left":  (20,  h - 20),
        "bottom-right": (w - 350, h - 20),
    }

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (corner_name, label, side) in enumerate(corner_text):
        x, y = pos[corner_name]
        color = (0, 200, 0) if i == step_idx else (160, 160, 160)
        cv2.putText(overlay, f"{label}  [{corner_name}, {side}]",
                    (x, y), font, 0.8, color, 2, cv2.LINE_AA)

    banner_h = 70
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), thickness=-1)
    label, side = corner_text[step_idx][1], corner_text[step_idx][2]
    cv2.putText(overlay,
                f"CLICK {label} — ({corner_text[step_idx][0]}, {side}).  Robot base = TOP edge.",
                (20, 48), font, 0.9, (0,255,0), 2, cv2.LINE_AA)
    return overlay

def click_points_with_guides(img_rgb):
    pts = []
    fig, ax = plt.subplots(num="H calibration — guided (robot side vs camera side)")
    plt.imshow(img_rgb); plt.axis('image')

    for step in range(4):
        guided = draw_click_guides(img_rgb, step)
        ax.clear(); ax.imshow(guided); ax.set_axis_off()
        plt.draw()
        click = plt.ginput(1, timeout=0)
        if not click:
            plt.close(fig)
            raise RuntimeError("Calibration cancelled.")
        pts.append(click[0])

        ax.plot(pts[-1][0], pts[-1][1], 'ro', markersize=8)
        ax.text(pts[-1][0] + 8, pts[-1][1] - 8, f"#{step+1}", color='r')
        plt.draw()

    plt.close(fig)
    return np.array(pts, dtype=np.float32)

def compute_and_save_h(img_path: Path, save_path: Path, points_override=None):
    img_rgb = load_image(img_path)

    if points_override is None:
        img_pts = click_points_with_guides(img_rgb)
    else:
        arr = []
        for p in points_override:
            u, v = map(float, p.split(","))
            arr.append((u, v))
        if len(arr) != 4:
            raise ValueError("Need exactly 4 points for --points (BL BR FR FL)")
        img_pts = np.array(arr, dtype=np.float32)

    # Homography: pixels -> meters (table plane)
    H, mask = cv2.findHomography(img_pts, WORLD_POINTS, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("Homography computation failed")

    np.save(str(save_path), H)
    print(f"Saved {save_path}")
    print("H =\n", H)

    # -------------- Diagnostics / Verification --------------
    Hinv = np.linalg.inv(H)

    def world_to_pixel(X, Y):
        p = np.array([X, Y, 1.0], dtype=np.float64)
        q = Hinv @ p
        return np.array([q[0]/q[2], q[1]/q[2]], dtype=np.float64)

    # Project the IDEAL world rectangle corners back to image:
    ideal_world = WORLD_POINTS
    proj_pts = np.array([world_to_pixel(X, Y) for (X, Y) in ideal_world], dtype=np.float32)

    # Per-corner pixel errors (clicked vs reprojected)
    # (order matches BL, BR, FR, FL)
    diffs = img_pts - proj_pts
    per_corner_err = np.linalg.norm(diffs, axis=1)  # pixels
    rms_err = np.sqrt(np.mean(per_corner_err**2))

    print("\nCalibration error (pixels):")
    names = ["BL", "BR", "FR", "FL"]
    for i, n in enumerate(names):
        print(f"  {n}: {per_corner_err[i]:.2f}px")
    print(f"  RMS: {rms_err:.2f}px\n")

    # Build a visualization
    vis = img_rgb.copy()

    # Draw clicked points (red) and projected ideal rectangle (cyan)
    for (u,v) in img_pts:
        cv2.circle(vis, (int(round(u)), int(round(v))), 8, (255,0,0), -1)  # red
    for (u,v) in proj_pts:
        cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0,255,255), 2)  # cyan ring

    # Connect projected rectangle edges (cyan)
    poly = proj_pts.astype(int).tolist()
    poly.append(poly[0])
    for i in range(4):
        cv2.line(vis, tuple(poly[i]), tuple(poly[i+1]), (0,255,255), 2)

    # Annotate errors
    for i, (c, p) in enumerate(zip(img_pts, proj_pts)):
        cu, cv, pu, pv = int(c[0]), int(c[1]), int(p[0]), int(p[1])
        cv2.line(vis, (cu,cv), (pu,pv), (0,165,255), 2)  # orange error vector
        cv2.putText(vis, f"{names[i]} err={per_corner_err[i]:.1f}px",
                    (pu+8, pv-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

    # Also draw origin and axes for a final sanity check
    u0, v0 = world_to_pixel(0.0, 0.0)
    cv2.circle(vis, (int(u0), int(v0)), 8, (255, 0, 0), -1)
    cv2.putText(vis, "origin (0,0) — back center (robot side)",
                (int(u0)+10, int(v0)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    ux, vx = world_to_pixel(0.10, 0.00)
    uy, vy = world_to_pixel(0.00, 0.10)
    cv2.arrowedLine(vis, (int(u0), int(v0)), (int(ux), int(vx)), (255,0,0), 2, tipLength=0.25)
    cv2.putText(vis, "+X (robot-left)", (int(ux)+6, int(vx)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.arrowedLine(vis, (int(u0), int(v0)), (int(uy), int(vy)), (0,255,0), 2, tipLength=0.25)
    cv2.putText(vis, "+Y (forward, toward camera side)", (int(uy)+6, int(vy)-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    title = (f"Verification — clicked (red) vs projected ideal (cyan)\n"
             f"Per-corner error px: BL={per_corner_err[0]:.1f}, BR={per_corner_err[1]:.1f}, "
             f"FR={per_corner_err[2]:.1f}, FL={per_corner_err[3]:.1f} | RMS={rms_err:.1f}px")
    fig2, ax2 = plt.subplots(num="Verification: clicked vs ideal")
    ax2.imshow(vis); ax2.set_title(title); ax2.axis('image')
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    home = Path.home()
    default_img = home / "mycobot_ws/src/mycobot_ros2/mycobot_pymoveit_api/mycobot_pymoveit_api/calib2.jpg"

    ap = argparse.ArgumentParser(description="Create H_table.npy with guided clicks + projected A4 diagnostics.")
    ap.add_argument("image", nargs="?", default=str(default_img), help="Calibration image path")
    ap.add_argument("--out", default="H_table.npy", help="Output .npy file")
    ap.add_argument("--points", nargs=4, metavar="u,v",
                    help="Optional: provide 4 pixel points (BL BR FR FL) instead of clicking")
    args = ap.parse_args()

    compute_and_save_h(Path(args.image), Path(args.out), points_override=args.points)
