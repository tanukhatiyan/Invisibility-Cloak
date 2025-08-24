import cv2
import numpy as np

WIN = 'Calibrate HSV'
def nothing(x): pass

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Try camera index 1 or check permissions.")

cv2.namedWindow(WIN)
# sensible starting values for a red cloth; tweak as needed
cv2.createTrackbar('LH', WIN, 0,   180, nothing)   # Low Hue
cv2.createTrackbar('LS', WIN, 120, 255, nothing)   # Low Sat
cv2.createTrackbar('LV', WIN, 70,  255, nothing)   # Low Val
cv2.createTrackbar('UH', WIN, 10,  180, nothing)   # Up Hue
cv2.createTrackbar('US', WIN, 255, 255, nothing)   # Up Sat
cv2.createTrackbar('UV', WIN, 255, 255, nothing)   # Up Val

print("Tip: ensure only the CLOAK shows up as white in 'Mask'. Press 'q' to finish.")

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)             # mirror for natural feel
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos('LH', WIN)
    ls = cv2.getTrackbarPos('LS', WIN)
    lv = cv2.getTrackbarPos('LV', WIN)
    uh = cv2.getTrackbarPos('UH', WIN)
    us = cv2.getTrackbarPos('US', WIN)
    uv = cv2.getTrackbarPos('UV', WIN)

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)         # smooth edges

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow(WIN, res)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        print("\nCopy these HSV bounds into config.json or the main script:")
        print("lower:", lower.tolist(), " upper:", upper.tolist())
        break

cap.release()
cv2.destroyAllWindows()
