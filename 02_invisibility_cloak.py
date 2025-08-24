import cv2
import numpy as np
import json
from collections import deque
import argparse
from pathlib import Path

def load_ranges(config_path):
    if config_path and Path(config_path).exists():
        cfg = json.loads(Path(config_path).read_text())
        ranges = cfg.get("ranges", [])
        dual_red = cfg.get("use_dual_range_for_red", False)
        return ranges, dual_red
    # defaults for RED cloak with dual range
    ranges = [
        {"lower": [0, 120, 70],   "upper": [10, 255, 255]},
        {"lower": [170, 120, 70], "upper": [180, 255, 255]},
    ]
    return ranges, True

def capture_background(cap, frames=60):
    buf = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        buf.append(frame)
    if not buf:
        raise RuntimeError("Failed to capture background frames.")
    # median is robust to small flickers
    return np.median(np.array(buf), axis=0).astype(np.uint8)

def build_mask(hsv, ranges):
    masks = []
    for r in ranges:
        lower = np.array(r["lower"], dtype=np.uint8)
        upper = np.array(r["upper"], dtype=np.uint8)
        masks.append(cv2.inRange(hsv, lower, upper))
    if not masks: 
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    return mask

def clean_mask(mask):
    kernel_open = np.ones((3,3), np.uint8)
    kernel_dil  = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    mask = cv2.dilate(mask, kernel_dil, iterations=1)
    return mask

def main():
    ap = argparse.ArgumentParser(description="Invisibility Cloak (OpenCV)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--config", type=str, default="", help="path to config.json with HSV ranges")
    ap.add_argument("--bg-frames", type=int, default=60, help="frames to build background")
    ap.add_argument("--width", type=int, default=640, help="capture width")
    ap.add_argument("--height", type=int, default=480, help="capture height")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1 or check permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ranges, _ = load_ranges(args.config)

    print("Step 1/1: Capturing clean background. Please stay out of frame...")
    background = capture_background(cap, frames=args.bg_frames)
    bg_buffer = deque([background], maxlen=15)

    print("Ready! Hold the colored cloak to vanish. Keys: q=quit, r=re-capture bg, s=save frame")
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = build_mask(hsv, ranges)
        mask     = clean_mask(raw_mask)
        inv_mask = cv2.bitwise_not(mask)

        # non-cloak area from live frame
        live_part = cv2.bitwise_and(frame, frame, mask=inv_mask)

        # cloak area from stable background
        stable_bg = np.median(np.array(bg_buffer), axis=0).astype(np.uint8)
        cloak_part = cv2.bitwise_and(stable_bg, stable_bg, mask=mask)

        out = cv2.add(live_part, cloak_part)
        cv2.imshow("Invisibility Cloak", out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite("invisibility_frame.jpg", out)
            print("Saved -> invisibility_frame.jpg")
        elif k == ord('r'):
            print("Re-capturing background... stay out of frame.")
            background = capture_background(cap, frames=args.bg_frames)
            bg_buffer.clear()
            bg_buffer.append(background)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
