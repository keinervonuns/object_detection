import sys

import cv2

# V4L2 exists on Linux only; elsewhere let OpenCV pick the default backend.
_CAM_BACKEND = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY

cap = cv2.VideoCapture(0, _CAM_BACKEND)
if not cap.isOpened():
    sys.exit("ERROR: Cannot open camera (index 0).")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Frame grab failed -- retrying ...")
        continue
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{w}x{h}  [q] quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
