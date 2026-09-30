"""
capture_images.py — Capture training images for a selected class.

Controls (while camera window is open)
───────────────────────────────────────
  SPACE  — save current frame
  c      — toggle continuous mode (auto-save every N ms)
  n      — switch to next class
  p      — switch to previous class
  q      — quit
"""

import sys
import time

import cv2
import config

# ── Settings ──────────────────────────────────────────────────────────────────
CONTINUOUS_INTERVAL_MS = 500   # ms between auto-saves in continuous mode

# V4L2 exists on Linux only; elsewhere let OpenCV pick the default backend.
_CAM_BACKEND = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, _CAM_BACKEND)
if not cap.isOpened():
    raise SystemExit("ERROR: Cannot open camera (index 0).")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ── Class selection ───────────────────────────────────────────────────────────
# config.CLASSES is a {class_name: led_number} dict — only the names are needed
# here, and they are kept in the dict's insertion order so n / p step through the
# classes in ascending LED-number order.
classes = list(config.CLASSES.keys())
class_idx = 0

def current_class():
    return classes[class_idx]

def save_dir():
    d = config.SOURCE_DIR / current_class()
    d.mkdir(parents=True, exist_ok=True)
    return d

def next_index(d):
    """Return the next free image index in the save directory (max + 1).

    Non-numeric stems (e.g. timestamp names) are ignored, so a new file can
    never land on top of an existing one.
    """
    nums = []
    for p in d.glob("*.jpg"):
        try:
            nums.append(int(p.stem))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0

def save_frame(frame):
    d = save_dir()
    idx = next_index(d)
    path = d / f"{idx:05d}.jpg"
    cv2.imwrite(str(path), frame)
    return path

# ── UI ────────────────────────────────────────────────────────────────────────
def draw_overlay(frame, class_name, count, continuous):
    h, w = frame.shape[:2]
    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.putText(frame, f"Class: {class_name}", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2, cv2.LINE_AA)
    # Image count
    cv2.putText(frame, f"Saved: {count}", (w - 180, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
    # Bottom bar
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    mode = "  [AUTO]" if continuous else ""
    cv2.putText(
        frame,
        f"SPACE: save  c: auto{mode}  n/p: class  q: quit",
        (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA,
    )
    # Flash red border in continuous mode
    if continuous:
        cv2.rectangle(frame, (3, 3), (w - 3, h - 3), (0, 0, 220), 3)
    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────
cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

continuous     = False
last_save_time = 0.0
saved_count    = next_index(save_dir())

print(f"\n  Starting capture — class: {current_class()}")
print(f"  Save folder: {save_dir()}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.monotonic()

    # Auto-save in continuous mode
    if continuous and (now - last_save_time) >= CONTINUOUS_INTERVAL_MS / 1000:
        path = save_frame(frame)
        saved_count += 1
        last_save_time = now
        print(f"  AUTO  → {path}")

    display = draw_overlay(frame.copy(), current_class(), saved_count, continuous)
    cv2.imshow("Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord(" "):
        path = save_frame(frame)
        saved_count += 1
        last_save_time = now
        print(f"  SAVED → {path}")

    elif key == ord("c"):
        continuous = not continuous
        last_save_time = 0.0
        print(f"  Continuous mode {'ON' if continuous else 'OFF'}")

    elif key in (ord("n"), ord("p")):
        continuous = False
        class_idx = (class_idx + (1 if key == ord("n") else -1)) % len(classes)
        saved_count = next_index(save_dir())
        print(f"  Class → {current_class()}  (saved so far: {saved_count})")
        print(f"  Save folder: {save_dir()}")

cap.release()
cv2.destroyAllWindows()
print("\n  Done.")
