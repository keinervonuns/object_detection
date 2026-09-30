"""
run.py — Real-time camera inference with OmniXAI explainability.

Usage
─────
    python run.py                # no GPIO LED output
    sudo python3 run.py          # on the Pi — drives the GPIO LEDs

The predicted class is broadcast on four GPIO pins. GPIO output is on by
default; pass --no-gpio to disable it (e.g. on the training PC, or when
RPi.GPIO is unavailable). On the Pi, run with the system python3 and sudo so
RPi.GPIO is importable — the project venv (Torch/venv) is added to sys.path so
the heavy packages still load from it. GPIO is initialised below, *before* those
imports, which can otherwise interfere with GPIO setup.

Controls
────────
  q  — quit
  g  — toggle GradCAM heatmap overlay (real-time)
  s  — save current frame + full OmniXAI explanation report to captures/
  e  — enter / exit region edit mode (saves the rectangle on exit)
  r  — reset FPS counter
  h  — print this help in the terminal

In region edit mode: arrows move the active border, s toggles expand/shrink,
z undoes, c resets to the full frame, e (or q) confirms and saves.

Requirements
────────────
  Trained model must exist at models/best_model.pth
  (run train.py first)
"""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Early bootstrap (must run before the heavy imports below) ─────────────────
# GPIO LED output is on unless --no-gpio is passed.
GPIO_ENABLED = "--no-gpio" not in sys.argv

# 1. Initialise GPIO first (system Python) — importing torch/cv2 beforehand can
#    interfere with GPIO setup.
if GPIO_ENABLED:
    try:
        import RPi.GPIO as _GPIO
        _GPIO.setmode(_GPIO.BCM)
        _GPIO.setwarnings(False)
        _GPIO.cleanup()   # left clean — BinaryLEDController reinitialises in main()
    except Exception as _exc:
        print(f"  ⚠ GPIO unavailable ({_exc}) — LED output disabled.")
        print("    On the Pi run with sudo; pass --no-gpio to silence this.")
        GPIO_ENABLED = False

# 2. Make the project venv importable (torch/cv2 load from it on the Pi).
_site_packages = glob.glob(os.path.join(HERE, "venv", "lib", "python*", "site-packages"))
if _site_packages:
    sys.path.insert(0, _site_packages[0])

# ── Heavy imports ─────────────────────────────────────────────────────────────
import configparser
import queue
import threading
import time
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config

# ──────────────────────────────────────────────────────────────────────────────
#  Region rectangle  (inline editor — saves to INI)
# ──────────────────────────────────────────────────────────────────────────────

_REGION_INI = config.BASE_DIR / "regions" / "region_config.ini"

_REGION_STEP = 8   # pixels moved per key-press

# Extended key codes (cv2.waitKeyEx on Linux/X11)
_KEY_UP    = 65362
_KEY_DOWN  = 65364
_KEY_LEFT  = 65361
_KEY_RIGHT = 65363


def load_region_rect(w: int, h: int) -> dict:
    """Load saved rectangle from INI; default to full frame."""
    cfg = configparser.ConfigParser()
    if _REGION_INI.exists():
        cfg.read(_REGION_INI)
        try:
            return {
                "left":   int(cfg["region"]["left"]),
                "top":    int(cfg["region"]["top"]),
                "right":  int(cfg["region"]["right"]),
                "bottom": int(cfg["region"]["bottom"]),
            }
        except (KeyError, ValueError):
            pass
    return {"left": 0, "top": 0, "right": w, "bottom": h}


def save_region_rect(rect: dict) -> None:
    """Persist rectangle to INI."""
    _REGION_INI.parent.mkdir(exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg["region"] = {k: str(v) for k, v in rect.items()}
    with _REGION_INI.open("w") as f:
        cfg.write(f)
    print(f"  ✓ Region saved → {_REGION_INI}")


def _clamp_rect(rect: dict, w: int, h: int) -> dict:
    """Keep rectangle edges inside the frame with a minimum size of 40 px."""
    min_size = 40
    rect["left"]   = max(0, min(rect["left"],   rect["right"]  - min_size))
    rect["top"]    = max(0, min(rect["top"],    rect["bottom"] - min_size))
    rect["right"]  = min(w, max(rect["right"],  rect["left"]   + min_size))
    rect["bottom"] = min(h, max(rect["bottom"], rect["top"]    + min_size))
    return rect


def rect_to_mask(rect: dict, h: int, w: int) -> np.ndarray:
    """Convert a rect dict to a uint8 mask (255 inside, 0 outside)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rect["top"]:rect["bottom"], rect["left"]:rect["right"]] = 255
    return mask

# ──────────────────────────────────────────────────────────────────────────────
#  GPIO LED control (non-blocking, with fallback)
# ──────────────────────────────────────────────────────────────────────────────

class BinaryLEDController:
    """Binary LED controller (standalone reference: tests/test_led.py --mode binary)."""
    
    def __init__(self, pins=(17, 27, 22, 23), enabled=True):
        self.pins = list(pins)
        self.GPIO = None
        if enabled:
            self._init_leds()
    
    def _init_leds(self):
        """Initialize the four GPIO output pins."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            
            for pin in self.pins:
                GPIO.setup(pin, GPIO.OUT)
            
            self.GPIO = GPIO
            print(f"  ✓ LEDs initialized with RPi.GPIO on GPIO pins {self.pins}")
            return True
        except Exception as e:
            print(f"  ✗ LED initialization failed: {type(e).__name__}: {e}")
            return False
    
    def set_binary(self, number):
        """Set LEDs to the 4-bit binary representation of `number` (active-LOW)."""
        if self.GPIO is None:
            return
        
        try: #
            binary = format(number, '04b')
            for i, bit in enumerate(reversed(binary)):  # reversed to match pin order
                self.GPIO.output(self.pins[i], self.GPIO.LOW if bit == '1' else self.GPIO.HIGH)
        except Exception as e:
            pass
    
    def off(self): #
        """Turn all LEDs off"""
        if self.GPIO is None:
            return
        try:
            for pin in self.pins:
                self.GPIO.output(pin, self.GPIO.HIGH)
        except Exception:
            pass
    
    def cleanup(self): #
        """Clean up GPIO resources"""
        if self.GPIO is None:
            return
        try:
            for pin in self.pins:
                self.GPIO.output(pin, self.GPIO.HIGH)
            self.GPIO.cleanup()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────
#  Model loader
# ──────────────────────────────────────────────────────────────────────────────

def load_model() -> Tuple[nn.Module, list, nn.Module, torch.device]:
    if not config.MODEL_PATH.exists():
        sys.exit(
            f"\n  ERROR: No trained model found at {config.MODEL_PATH}\n"
            "  Run `python train.py` first.\n"
        )

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config.MODEL_PATH, map_location=device)
    classes    = checkpoint["classes"]
    backbone   = checkpoint.get("backbone", config.BACKBONE)
    n_cls      = len(classes)

    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
        target_layer = m.features[-1]
    elif backbone == "efficientnet_b2":
        m = models.efficientnet_b2()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
        target_layer = m.features[-1]
    elif backbone == "resnet50":
        m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, n_cls)
        target_layer = m.layer4[-1]
    else:
        sys.exit(f"\n  ERROR: Unknown backbone in checkpoint: {backbone}\n")

    m.load_state_dict(checkpoint["model_state"])
    m = m.to(device).eval()

    print(
        f"  Loaded model — {backbone}  |  "
        f"epoch {checkpoint['epoch']}  |  "
        f"val acc {checkpoint['val_acc']*100:.2f}%"
    )
    print(f"  Device  : {device}")
    print(f"  Classes : {classes}")

    return m, classes, target_layer, device


# ──────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# Pre-computed constants for fast OpenCV-based preprocessing (no PIL overhead)
_RESIZE_SIZE = int(config.IMG_SIZE * 1.15)   # 257 for IMG_SIZE=224
_CROP_START  = (_RESIZE_SIZE - config.IMG_SIZE) // 2
_mean_t      = torch.tensor(_MEAN).view(3, 1, 1)
_std_t       = torch.tensor(_STD).view(3, 1, 1)


def preprocess(bgr_frame: np.ndarray) -> torch.Tensor:
    """Convert a BGR OpenCV frame to a normalised (1,C,H,W) tensor (OpenCV, no PIL)."""
    rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_RESIZE_SIZE, _RESIZE_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    s       = _CROP_START
    cropped = resized[s : s + config.IMG_SIZE, s : s + config.IMG_SIZE]
    # uint8 HWC  →  float32 CHW tensor, normalised
    t = torch.from_numpy(cropped).permute(2, 0, 1).float().div_(255.0)
    t.sub_(_mean_t).div_(_std_t)
    return t.unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
#  GradCAM  (lightweight PyTorch-hook implementation for real-time use)
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Map.
    Registers forward + backward hooks on `target_layer`.
    """

    def __init__(
        self, model: nn.Module, target_layer: nn.Module, device: torch.device
    ) -> None:
        self.model  = model
        self.device = device
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _mod, _inp, output: torch.Tensor) -> None:
        self._acts = output.detach()

    def _bwd_hook(self, _mod, _grad_in, grad_out: Tuple) -> None:
        self._grads = grad_out[0].detach()

    def __call__(
        self, tensor: torch.Tensor, class_idx: int
    ) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1]."""
        with torch.enable_grad():
            self.model.zero_grad()
            t = tensor.clone().to(self.device).requires_grad_(True)
            output = self.model(t)
            output[0, class_idx].backward()

        # Weight activations by global-average-pooled gradients
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (weights * self._acts).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam


# ──────────────────────────────────────────────────────────────────────────────
#  OmniXAI explainer  (built once, used on demand with 's' key)
# ──────────────────────────────────────────────────────────────────────────────

def build_omnixai_explainer(
    model: nn.Module,
    target_layer: nn.Module,
    device: torch.device,
) -> Tuple[Optional[object], Optional[type]]:
    """
    Wrap the trained model in an OmniXAI VisionExplainer.
    Returns (explainer, OmniImage class) or (None, None) on failure.
    """
    try:
        from omnixai.data.image import Image as OmniImage
        from omnixai.explainers.vision import VisionExplainer

        mean = np.array(_MEAN, dtype=np.float32)
        std  = np.array(_STD,  dtype=np.float32)

        def preprocess_fn(omni_imgs):
            """OmniImage (N,H,W,C uint8) → normalised GPU tensor."""
            data = omni_imgs.to_numpy()  # (N, H, W, 3)
            batch = []
            for img in data:
                arr = img.astype(np.float32) / 255.0
                arr = (arr - mean) / std
                batch.append(torch.from_numpy(arr).permute(2, 0, 1))
            return torch.stack(batch).to(device)

        def postprocess_fn(logits: torch.Tensor) -> torch.Tensor:
            return torch.softmax(logits, dim=1)

        explainer = VisionExplainer(
            explainers=["gradcam", "lime"],
            mode="classification",
            model=model,
            preprocess=preprocess_fn,
            postprocess=postprocess_fn,
            params={
                "gradcam": {"target_layer": target_layer},
                "lime":    {"num_samples": 64},
            },
        )
        print("  OmniXAI VisionExplainer ready  (GradCAM + LIME on demand)")
        return explainer, OmniImage

    except Exception as exc:
        print(f"  OmniXAI explainer unavailable: {exc}")
        print("  (the `s` key will save the frame without an HTML report)")
        traceback.print_exc()
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
#  Drawing / UI helpers
# ──────────────────────────────────────────────────────────────────────────────

# One distinct colour per class (BGR)
_PALETTE = [
    (71,  99, 255),   # tomato-red  -> BGR
    (0,  165, 255),   # orange
    (50, 205,  50),   # lime green
    (255, 191,  0),   # deep sky blue
    (226,  43, 138),  # med violet
    (147,  20, 255),  # deep pink
    (127, 255,   0),  # spring green
    (0,  215, 255),   # gold
]


def overlay_heatmap(
    frame: np.ndarray, cam: np.ndarray, alpha: float = 0.42
) -> np.ndarray:
    h, w = frame.shape[:2]
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat   = cv2.applyColorMap(
        (cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(frame, 1.0 - alpha, heat, alpha, 0)


def apply_region_display(
    frame: np.ndarray,
    rect: dict,
    edit_mode: bool,
    shrink_mode: bool = False,
) -> np.ndarray:
    """
    Grey out pixels outside the detection rectangle.
    In edit mode also draws resize handles and an info banner.
    """
    l, t, r, b = rect["left"], rect["top"], rect["right"], rect["bottom"]
    h, w = frame.shape[:2]

    # Darken the area outside the region (35% brightness)
    dimmed = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    out = frame.copy()
    # Top strip
    if t > 0:
        out[:t, :] = dimmed[:t, :]
    # Bottom strip
    if b < h:
        out[b:, :] = dimmed[b:, :]
    # Left strip (between t and b)
    if l > 0:
        out[t:b, :l] = dimmed[t:b, :l]
    # Right strip (between t and b)
    if r < w:
        out[t:b, r:] = dimmed[t:b, r:]

    # Border colour: yellow in edit mode, soft green normally
    color     = (0, 220, 255) if edit_mode else (80, 200, 80)
    thickness = 2 if edit_mode else 1
    cv2.rectangle(out, (l, t), (r, b), color, thickness)

    if edit_mode:
        # Corner handle squares
        hs = 8
        for cx, cy in [(l, t), (r, t), (l, b), (r, b)]:
            cv2.rectangle(out, (cx - hs, cy - hs), (cx + hs, cy + hs), color, -1)

        # Edit mode banner at bottom of frame
        mode_label = "SHRINK" if shrink_mode else "EXPAND"
        mode_color = (0, 120, 255) if shrink_mode else (0, 220, 255)
        bh = 48
        cv2.rectangle(out, (0, h - bh), (w, h), (20, 20, 20), -1)
        cv2.putText(
            out, f"REGION EDIT  │  Mode: {mode_label}  │  Arrows: move border   s: expand/shrink   z: undo   c: reset   e: confirm & save",
            (10, h - bh + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_color, 1, cv2.LINE_AA,
        )
        coverage = ((r - l) * (b - t)) / (w * h) * 100
        cv2.putText(
            out,
            f"L:{l}  T:{t}  R:{r}  B:{b}    {r-l}x{b-t} px   {coverage:.1f}% coverage",
            (10, h - bh + 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA,
        )

    return out


def draw_ui(
    frame: np.ndarray,
    class_name: str,
    conf: float,
    probs: np.ndarray,
    class_idx: int,
    fps: float,
    gradcam_on: bool,
    classes: list,
    region_edit: bool = False,
) -> np.ndarray:
    h, w = frame.shape[:2]
    color = _PALETTE[class_idx % len(_PALETTE)]

    # ── Top banner ────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 64), (18, 18, 18), -1)

    label = class_name.upper()
    cv2.putText(
        frame, label, (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 1.35, color, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"{conf * 100:.1f}%", (w - 120, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 220), 2, cv2.LINE_AA
    )

    # Confidence bar
    bar_w = int((w - 24) * conf)
    cv2.rectangle(frame, (12, 54), (w - 12, 62), (55, 55, 55), -1)
    cv2.rectangle(frame, (12, 54), (12 + bar_w, 62), color, -1)

    # ── Side panel: top-3 probabilities ───────────────────────────────────────
    top3_idx  = np.argsort(probs)[::-1][:3]
    panel_x   = w - 210
    panel_y   = 80
    cv2.rectangle(frame, (panel_x - 6, panel_y - 4),
                  (w - 6, panel_y + 68), (25, 25, 25), -1)

    for rank, idx in enumerate(top3_idx):
        c      = _PALETTE[idx % len(_PALETTE)]
        bar_len = int(190 * probs[idx])
        y0     = panel_y + rank * 22
        cv2.rectangle(frame, (panel_x, y0), (panel_x + bar_len, y0 + 14), c, -1)
        cv2.putText(
            frame, f"{classes[idx]}  {probs[idx]*100:.0f}%",
            (panel_x + 2, y0 + 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA
        )

    # ── Bottom status bar (hidden during region edit — edit mode has its own) ─
    if not region_edit:
        cv2.rectangle(frame, (0, h - 26), (w, h), (18, 18, 18), -1)
        status_str = (
            f"FPS {fps:5.1f}  |  "
            f"{'GradCAM ON' if gradcam_on else 'GradCAM OFF'}  |  "
            "[g] toggle  [s] save  [e] region  [r] fps  [q] quit"
        )
        cv2.putText(
            frame, status_str, (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA
        )

    return frame


# ──────────────────────────────────────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────────────────────────────────────

HELP = """
  ┌──────────────────────────────────────────┐
  │  OmniXAI Live Feed — Keyboard Controls   │
  │  q   quit                                │
  │  g   toggle real-time GradCAM overlay    │
  │  s   save frame + OmniXAI HTML report    │
  │  e   enter / exit region edit mode       │
  │       Arrows  move the active border     │
  │       s       toggle expand / shrink     │
  │       z       undo last move             │
  │       c       reset to full frame        │
  │  r   reset FPS counter                   │
  │  h   show this help                      │
  └──────────────────────────────────────────┘
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Background inference worker
# ──────────────────────────────────────────────────────────────────────────────

def _inference_worker(
    model: nn.Module,
    classes: list,
    device: torch.device,
    gradcam_fn: "GradCAM",
    frame_q: "queue.Queue",
    result: dict,
    result_lock: threading.Lock,
    stop_evt: threading.Event,
    led_ctrl: "BinaryLEDController",
    numbers: dict,
) -> None:
    """Background thread: preprocess + infer (+ optional GradCAM); updates shared result."""
    while not stop_evt.is_set():
        try:
            frame, gradcam_on = frame_q.get(timeout=0.05)
        except queue.Empty:
            continue

        tensor     = preprocess(frame)
        tensor_dev = tensor.to(device)

        with torch.inference_mode():
            logits = model(tensor_dev)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        class_name = classes[pred_idx]
        number = numbers.get(class_name, 0)
        led_ctrl.set_binary(number)
        cam      = gradcam_fn(tensor, pred_idx) if gradcam_on else None

        with result_lock:
            result["probs"]      = probs
            result["pred_idx"]   = pred_idx
            result["conf"]       = float(probs[pred_idx])
            result["class_name"] = classes[pred_idx]
            result["cam"]        = cam


def main() -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║            OmniXAI  —  Live Detection Feed               ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    model, classes, target_layer, device = load_model()
    gradcam_fn = GradCAM(model, target_layer, device)
    omnixai, OmniImage = build_omnixai_explainer(model, target_layer, device)

    # ── LED initialization ─────────────────────────────────────────────────────
    print()
    led_ctrl = BinaryLEDController(enabled=GPIO_ENABLED)
    print()

    # ── Class to number mapping ─────────────────────────────────────────────────
    numbers = config.CLASSES
    print(f"  Class to number mapping: {numbers}")
    print()

    save_dir = config.BASE_DIR / "captures"
    save_dir.mkdir(exist_ok=True)

    # ── Detection region ───────────────────────────────────────────────────────
    # Rectangle loaded from regions/region_config.ini and editable at runtime
    # with the `e` key. With no INI present the region is the full frame.
    _tmp_cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    _tmp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    _tmp_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    _tmp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    _fw = int(_tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1920)
    _fh = int(_tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    _tmp_cap.release()

    region_rect = load_region_rect(_fw, _fh)

    if _REGION_INI.exists():
        print(f"  ✓ Region rectangle loaded from {_REGION_INI}")
    else:
        print("  ⓘ No detection region — press e to set one.")
    print()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit("\n  ERROR: Cannot open camera (index 0).\n")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # minimize latency

    cv2.namedWindow("OmniXAI Live Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("OmniXAI Live Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(HELP)

    # ── Background inference thread ───────────────────────────────────────────
    _result = {
        "probs":      np.ones(len(classes), dtype=np.float32) / len(classes),
        "pred_idx":   0,
        "conf":       0.0,
        "class_name": classes[0],
        "cam":        None,
    }
    _result_lock = threading.Lock()
    _frame_q     = queue.Queue(maxsize=1)
    _stop_evt    = threading.Event()
    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(model, classes, device, gradcam_fn,
              _frame_q, _result, _result_lock, _stop_evt, led_ctrl, numbers),
        daemon=True,
    )
    infer_thread.start()

    gradcam_on        = False
    region_edit_mode  = False
    region_shrink_mode = False
    region_rect_history: list = []   # undo stack for rect edits
    fps         = 0.0
    frame_count = 0
    fps_t0      = time.perf_counter()
    save_count  = 0
    
    # ── Flicker prevention: 1-second delay before switching to new class ──────
    display_class_name = classes[0]
    display_pred_idx   = 0
    display_conf       = 0.0
    display_probs      = np.ones(len(classes), dtype=np.float32) / len(classes)
    last_class_change_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  WARNING: Frame grab failed — retrying …")
            continue

        # Apply detection region mask for inference
        frame_for_inference = frame.copy()
        h_f, w_f = frame.shape[:2]
        if _REGION_INI.exists():
            _inf_mask = rect_to_mask(region_rect, h_f, w_f)
            frame_for_inference[_inf_mask < 128] = 0

        # ── Enqueue frame for background inference (drop if busy) ───────────
        try:
            _frame_q.put_nowait((frame_for_inference, gradcam_on))
        except queue.Full:
            pass  # inference still running — display prior result

        # ── Fetch latest inference result ─────────────────────────────────────
        with _result_lock:
            probs       = _result["probs"].copy()
            pred_idx    = _result["pred_idx"]
            conf        = _result["conf"]
            class_name  = _result["class_name"]
            current_cam = _result["cam"]
        
        # ── Apply 1-second delay before switching to new class ────────────────
        current_time = time.perf_counter()
        if class_name != display_class_name:
            # Class changed - check if configured delay has passed
            if current_time - last_class_change_time >= config.DISPLAY_SECONDS:
                display_class_name = class_name
                display_pred_idx   = pred_idx
                display_conf       = conf
                display_probs      = probs.copy()
                last_class_change_time = current_time
            # else: keep displaying the old class until delay expires
        else:
            # Same class - always update (confidence might have changed)
            display_class_name = class_name
            display_pred_idx   = pred_idx
            display_conf       = conf
            display_probs      = probs.copy()

        if gradcam_on and current_cam is not None:
            frame = overlay_heatmap(frame, current_cam)

        # Grey-out / border for detection region
        frame = apply_region_display(frame, region_rect, region_edit_mode, region_shrink_mode)

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.perf_counter() - fps_t0
        if elapsed >= 0.5:
            fps         = frame_count / elapsed
            frame_count = 0
            fps_t0      = time.perf_counter()

        # ── Draw UI ───────────────────────────────────────────────────────────
        frame = draw_ui(
            frame, display_class_name, display_conf, display_probs, display_pred_idx,
            fps, gradcam_on, classes,
            region_edit=region_edit_mode,
        )

        cv2.imshow("OmniXAI Live Detection", frame)

        # ── Key handling ──────────────────────────────────────────────────────
        raw_key = cv2.waitKeyEx(1)
        key     = raw_key & 0xFF if raw_key != -1 else 0xFF

        # --- Region edit mode: arrow keys move borders ---
        if region_edit_mode and raw_key != -1:
            h_f, w_f = frame.shape[:2]
            base = raw_key & 0xFFFF
            step = _REGION_STEP
            if base in (_KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT):
                region_rect_history.append(region_rect.copy())  # save for undo
            # In shrink mode arrows move the border inward; expand moves outward
            if base == _KEY_UP:
                if region_shrink_mode:  region_rect["top"]    += step
                else:                   region_rect["top"]    -= step
            elif base == _KEY_DOWN:
                if region_shrink_mode:  region_rect["bottom"] -= step
                else:                   region_rect["bottom"] += step
            elif base == _KEY_LEFT:
                if region_shrink_mode:  region_rect["left"]   += step
                else:                   region_rect["left"]   -= step
            elif base == _KEY_RIGHT:
                if region_shrink_mode:  region_rect["right"]  -= step
                else:                   region_rect["right"]  += step
            region_rect = _clamp_rect(region_rect, w_f, h_f)

        if key == ord("q"):
            if region_edit_mode:
                save_region_rect(region_rect)
                region_edit_mode    = False
                region_shrink_mode  = False
                region_rect_history.clear()
                print("  Region edit mode OFF")
            else:
                break

        elif key == ord("e"):
            region_edit_mode = not region_edit_mode
            if not region_edit_mode:
                region_shrink_mode = False
                region_rect_history.clear()
                save_region_rect(region_rect)
                print("  Region edit mode OFF")
            else:
                print("  Region edit mode ON  —  Arrows: move   s: expand/shrink   z: undo   c: reset   e: confirm")

        elif key == ord("z") and region_edit_mode:
            if region_rect_history:
                region_rect = region_rect_history.pop()
                print("  Undo.")
            else:
                print("  Nothing to undo.")

        elif key == ord("c") and region_edit_mode:
            h_f, w_f = frame.shape[:2]
            region_rect = {"left": 0, "top": 0, "right": w_f, "bottom": h_f}
            region_rect_history.clear()
            print("  Region reset to full frame.")

        elif key == ord("s") and region_edit_mode:
            region_shrink_mode = not region_shrink_mode
            label = "SHRINK" if region_shrink_mode else "EXPAND"
            print(f"  Region mode: {label}")

        elif key == ord("g") and not region_edit_mode:
            gradcam_on = not gradcam_on
            with _result_lock:
                _result["cam"] = None   # clear stale heatmap immediately
            print(f"  GradCAM {'ON' if gradcam_on else 'OFF'}")

        elif key == ord("r") and not region_edit_mode:
            fps = 0.0
            frame_count = 0
            fps_t0 = time.perf_counter()

        elif key == ord("h") and not region_edit_mode:
            print(HELP)

        elif key == ord("s") and not region_edit_mode:
            stem = f"{save_count:04d}_{display_class_name}"

            # Save annotated frame
            img_path = save_dir / f"capture_{stem}.jpg"
            cv2.imwrite(str(img_path), frame)
            print(f"  Saved frame → {img_path}")

            # OmniXAI full explanation
            if omnixai is not None and OmniImage is not None:
                try:
                    rgb_resized = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        (config.IMG_SIZE, config.IMG_SIZE),
                    )
                    omni_img = OmniImage(
                        data=np.array([rgb_resized]),
                        batched=True,
                        channel_last=True,
                    )
                    local_exp  = omnixai.explain(omni_img)
                    html_path  = save_dir / f"explanation_{stem}.html"
                    local_exp.plotly_plot().write_html(str(html_path))
                    print(f"  Saved explanation → {html_path}")
                except Exception as exc:
                    print(f"  OmniXAI explanation failed: {exc}")
                    traceback.print_exc()
            else:
                print("  ⓘ OmniXAI not available — frame saved without an HTML report. "
                      "Install it with: pip install \"omnixai[vision,plot]\"")

            save_count += 1

    _stop_evt.set()
    infer_thread.join(timeout=2.0)

    # ── LED cleanup ────────────────────────────────────────────────────────────
    led_ctrl.cleanup()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Feed closed.")


if __name__ == "__main__":
    main()
