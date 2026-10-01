"""
pipeline_ui.py — GUI control panel for the PI Detection pipeline.

A Tkinter wizard that walks through every step of the pipeline
(environment → classes → collect → split → train → transfer → detect),
runs pre-flight checks so you can see whether a step is ready, and launches
the existing scripts as subprocesses with their live output streamed into the
window.

It is purely a front-end: it never edits config.py, train.py, run.py or any
other project file. Every action is exactly the command the README documents,
run from the project root with the project's virtual-environment interpreter.

Usage
─────
    # Windows:  venv\\Scripts\\Activate.ps1      Linux/Pi: source venv/bin/activate
    python pipeline_ui.py

Cross-platform: runs on Windows and Linux (the same machines the pipeline
targets). Only the Python standard library is required — no new dependencies.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# ──────────────────────────────────────────────────────────────────────────────
#  Project paths & config
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import config  # noqa: E402  — constants only, no side effects
    CONFIG_ERROR = None
except Exception as exc:  # pragma: no cover - only on a broken checkout
    config = None
    CONFIG_ERROR = f"{type(exc).__name__}: {exc}"


def find_python() -> str:
    """Prefer the project venv interpreter, else the one running this UI."""
    for cand in (
        BASE_DIR / "venv" / "Scripts" / "python.exe",   # Windows
        BASE_DIR / "venv" / "bin" / "python",           # Linux / Pi
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


PYTHON = find_python()


# ──────────────────────────────────────────────────────────────────────────────
#  Check results
# ──────────────────────────────────────────────────────────────────────────────

# Severity of a single check. BLOCK stops a step from running; WARN is advisory
# ("runnable, but not optimal") and INFO is a neutral note. The step badge only
# reacts to BLOCK (and PEND while a probe is still running).
OK, WARN, BLOCK, INFO, PEND = "ok", "warn", "block", "info", "pending"
_SYMBOL = {OK: "✓", WARN: "⚠", BLOCK: "✖", INFO: "•", PEND: "○"}
_COLOR = {OK: "#1a7f37", WARN: "#9a6700", BLOCK: "#b42318",
          INFO: "#57606a", PEND: "#0969da"}


class Chk:
    """One pre-flight check result."""

    __slots__ = ("status", "text")

    def __init__(self, status: str, text: str) -> None:
        self.status = status
        self.text = text


def badge(checks: list[Chk]) -> str:
    """Left-hand step status — is this step runnable?

    Only a BLOCK stops a step: BLOCK > PEND (still checking) > OK (ready).
    WARN and INFO are advisory and deliberately do NOT change the badge, so a
    step that is merely "not optimal" still shows a check and can be run.
    """
    present = {c.status for c in checks}
    if BLOCK in present:
        return BLOCK
    if PEND in present:
        return PEND
    return OK


def blockers(checks: list[Chk]) -> list[Chk]:
    """The checks that actually prevent a step from running."""
    return [c for c in checks if c.status == BLOCK]


def _truncate(text: str, n: int = 58) -> str:
    """One-line, length-capped version of a check message for the step list."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def summary(checks: list[Chk]) -> tuple[str, str]:
    """Most important message to show next to a step's name, with its severity.

    Block > warning > pending; returns ("", OK) when there is nothing to show.
    """
    for status in (BLOCK, WARN, PEND):
        for c in checks:
            if c.status == status:
                return c.text, status
    return "", OK


# ──────────────────────────────────────────────────────────────────────────────
#  Filesystem helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def source_counts(cfg) -> dict[str, int]:
    """{class_name: image_count} for every sub-folder of source/."""
    if cfg is None or not cfg.SOURCE_DIR.exists():
        return {}
    return {
        p.name: count_images(p)
        for p in sorted(cfg.SOURCE_DIR.iterdir())
        if p.is_dir()
    }


def split_state(cfg) -> dict:
    """Per-class counts + broken-symlink counts for data/train and data/val."""
    st = {"train": {}, "val": {}, "broken_train": 0, "broken_val": 0}
    if cfg is None:
        return st
    for name, folder in (("train", cfg.TRAIN_DIR), ("val", cfg.VAL_DIR)):
        if not folder.exists():
            continue
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            files = list(sub.iterdir())
            st[name][sub.name] = sum(
                1 for f in files if f.is_file() and f.suffix.lower() in IMG_EXTS
            )
            st["broken_" + name] += sum(
                1 for f in files if f.is_symlink() and not f.exists()
            )
    return st


# ──────────────────────────────────────────────────────────────────────────────
#  Pre-flight checks  (pure functions — no Tk, easy to test)
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_PKGS = ["torch", "torchvision", "cv2", "numpy", "matplotlib",
                  "sklearn", "tqdm", "PIL"]
_OPTIONAL_PKGS = ["omnixai"]
_VALID_BACKBONES = {"efficientnet_b0", "efficientnet_b2", "resnet50"}


def checks_environment(probe) -> list[Chk]:
    out: list[Chk] = []
    out.append(Chk(INFO, f"Interpreter : {PYTHON}"))
    if probe is None:
        out.append(Chk(PEND, "Probing interpreter & packages …"))
        return out
    if probe.get("error"):
        out.append(Chk(WARN, f"Environment probe failed ({probe['error']}) — could not "
                             "confirm packages. See the Output log; use “Re-probe”."))
        return out

    out.append(Chk(OK, f"Python {probe['version']}  ({platform.system()})"))

    missing = [p for p in _REQUIRED_PKGS if not probe["packages"].get(p)]
    for pkg in _REQUIRED_PKGS:
        ver = probe["packages"].get(pkg)
        if ver:
            out.append(Chk(OK, f"{pkg} {ver}"))
        else:
            out.append(Chk(WARN, f"{pkg} is NOT installed — run “Install requirements”"))
    for pkg in _OPTIONAL_PKGS:
        ver = probe["packages"].get(pkg)
        if ver:
            out.append(Chk(OK, f"{pkg} {ver}  (optional)"))
        else:
            out.append(Chk(INFO, f"{pkg} not installed  (optional — HTML report only)"))

    if probe.get("cuda"):
        out.append(Chk(OK, f"CUDA available — {probe.get('gpu','GPU')} "
                           f"({probe.get('vram_gb','?')} GB VRAM)"))
    elif probe.get("cuda") is False:
        out.append(Chk(WARN, "CUDA not available — training will use the CPU (slow). "
                             "Use BACKBONE = efficientnet_b0 for CPU training."))
    if missing:
        out.append(Chk(WARN, "Missing required packages: " + ", ".join(missing) +
                             " — installable from this step (not a blocker)."))
    return out


def checks_classes(cfg) -> list[Chk]:
    out: list[Chk] = []
    if cfg is None:
        return [Chk(BLOCK, f"config.py could not be imported ({CONFIG_ERROR})")]
    classes: dict = getattr(cfg, "CLASSES", {})
    if not classes:
        return [Chk(BLOCK, "config.py: CLASSES is empty — define your classes.")]

    out.append(Chk(OK, f"{len(classes)} classes (order matters): "
                       f"{', '.join(classes)}"))

    values = list(classes.values())
    bad_type = [k for k, v in classes.items() if not isinstance(v, int)]
    if bad_type:
        out.append(Chk(BLOCK, "Non-integer LED value for: " + ", ".join(bad_type)))
    else:
        oob = [f"{k}={v}" for k, v in classes.items() if not 0 <= v <= 15]
        if oob:
            out.append(Chk(BLOCK, "LED value outside 0–15 (breaks the 4-bit output): "
                                + ", ".join(oob)))
        else:
            out.append(Chk(OK, "All LED values are within 0–15."))

        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            out.append(Chk(WARN, "Duplicate LED values (those classes look identical "
                                 f"on the LEDs): {dupes}"))
        else:
            out.append(Chk(OK, "All LED values are unique."))

    bad_name = [k for k in classes if not k or any(c in k for c in "/\\")]
    if bad_name:
        out.append(Chk(BLOCK, f"Class names are not valid folder names: {bad_name}"))

    # Do the source folders line up with CLASSES?
    on_disk = set(source_counts(cfg))
    if on_disk:
        missing = [k for k in classes if k not in on_disk]
        extra = sorted(on_disk - set(classes))
        if missing:
            out.append(Chk(WARN, "In CLASSES but no source/ folder yet (run “Create "
                                 f"source folders”): {missing}"))
        if extra:
            out.append(Chk(WARN, "In source/ but not in CLASSES (they WILL be split "
                                 f"and trained on): {extra}"))
    return out


def checks_source(cfg) -> list[Chk]:
    out: list[Chk] = []
    if cfg is None:
        return [Chk(BLOCK, f"config.py could not be imported ({CONFIG_ERROR})")]

    if not cfg.SOURCE_DIR.exists():
        return [Chk(WARN, "source/ does not exist yet — capture images or run “Create "
                          "source folders” (this step still runs).")]

    counts = source_counts(cfg)
    if not counts:
        return [Chk(WARN, "source/ has no class sub-folders yet — capture images or run "
                          "“Create source folders” (this step still runs).")]

    out.append(Chk(INFO, f"source/ : {cfg.SOURCE_DIR}"))
    total = sum(counts.values())
    empties, few = [], []
    for name, n in counts.items():
        tag = ""
        if n == 0:
            empties.append(name)
            tag = "  ← empty"
        elif n < 50:
            few.append(name)
            tag = "  ← aim for 50–200"
        out.append(Chk(OK if n >= 50 else WARN, f"{name:<22} {n:>5} images{tag}"))

    out.append(Chk(OK, f"Total : {total} images across {len(counts)} classes"))
    if empties:
        out.append(Chk(WARN, "Classes with no images are skipped by the split: "
                             + ", ".join(empties)))
    if few:
        out.append(Chk(WARN, "Fewer than 50 images in: " + ", ".join(few)))
    present = [n for n in counts.values() if n > 0]
    if len(present) > 1 and max(present) > 3 * min(present):
        out.append(Chk(INFO, f"Class imbalance — max {max(present)} vs min {min(present)} "
                             "images. More balanced classes usually train better."))
    return out


def data_ready(cfg) -> tuple[bool, str]:
    """Is data/train & data/val present and usable for training?"""
    if not (cfg.TRAIN_DIR.exists() and cfg.VAL_DIR.exists()):
        return False, "data/ not built — run “Split into train / val” first"
    st = split_state(cfg)
    if sum(st["train"].values()) == 0 or sum(st["val"].values()) == 0:
        return False, "data/train or data/val is empty"
    if st["broken_train"] or st["broken_val"]:
        return False, "data/ has broken symlinks — re-run the split"
    return True, ""


def checks_split(cfg) -> list[Chk]:
    """State of the split step.

    Only “there is nothing to split” blocks this step — the current contents of
    data/ are reported as information, since running the split is the fix.
    """
    out: list[Chk] = []
    if cfg is None:
        return [Chk(BLOCK, f"config.py could not be imported ({CONFIG_ERROR})")]

    counts = source_counts(cfg)
    total_src = sum(counts.values())
    if total_src == 0:
        out.append(Chk(BLOCK, "No images in source/ — nothing to split. Collect "
                              "images first."))
    else:
        out.append(Chk(OK, f"source/ has {total_src} images in {len(counts)} classes"))

    st = split_state(cfg)
    if not (cfg.TRAIN_DIR.exists() and cfg.VAL_DIR.exists()):
        out.append(Chk(INFO, "data/ not built yet — run this step to create it."))
        return out

    n_tr, n_va = sum(st["train"].values()), sum(st["val"].values())
    out.append(Chk(OK, f"data/train : {n_tr:>5} images in {len(st['train'])} classes"))
    out.append(Chk(OK, f"data/val   : {n_va:>5} images in {len(st['val'])} classes"))

    for name in sorted(set(st["train"]) | set(st["val"])):
        tr, va = st["train"].get(name, 0), st["val"].get(name, 0)
        out.append(Chk(OK if (tr and va) else WARN,
                       f"{name:<22} train {tr:>5}   val {va:>5}"))

    if set(st["train"]) != set(st["val"]):
        out.append(Chk(WARN, "Train and val contain different class sets — re-run the split."))
    if counts and set(st["train"]) != set(counts):
        out.append(Chk(WARN, "Split classes differ from source/ — re-run the split."))
    if cfg.COPY_FILES:
        out.append(Chk(INFO, "COPY_FILES = True — real copies (no symlink support needed)."))
    else:
        note = "symlinks; needs Developer Mode on Windows" if os.name == "nt" else "symlinks"
        out.append(Chk(INFO, f"COPY_FILES = False — {note} (falls back to copying)."))

    broken = st["broken_train"] + st["broken_val"]
    if broken:
        out.append(Chk(WARN, f"{broken} broken symlink(s) in data/ — re-run the split to fix."))
    if n_tr == 0 or n_va == 0:
        out.append(Chk(WARN, "One of the splits is empty — re-run the split."))
    return out


def checks_train(cfg, probe) -> list[Chk]:
    out: list[Chk] = []
    if cfg is None:
        return [Chk(BLOCK, f"config.py could not be imported ({CONFIG_ERROR})")]

    # Data is a hard prerequisite: train.py exits if data/ is missing/empty.
    data_ok, data_reason = data_ready(cfg)
    if data_ok:
        out.append(Chk(OK, "Training data present (data/train & data/val)."))
    else:
        out.append(Chk(BLOCK, f"Cannot train — {data_reason}."))

    backbone = str(getattr(cfg, "BACKBONE", ""))
    if backbone.lower() in _VALID_BACKBONES:
        out.append(Chk(OK, f"BACKBONE = {backbone}"))
    else:
        out.append(Chk(BLOCK, f"BACKBONE = {backbone!r} is not one of "
                            f"{sorted(_VALID_BACKBONES)}"))

    if probe is not None:
        if probe["packages"].get("torch"):
            out.append(Chk(OK, f"torch {probe['packages']['torch']}"))
        else:
            out.append(Chk(BLOCK, "torch is not installed — run “Install requirements”."))
        if probe.get("cuda"):
            out.append(Chk(OK, f"GPU training on {probe.get('gpu','GPU')}"))
        elif probe.get("cuda") is False:
            out.append(Chk(WARN, "No CUDA — CPU training is many minutes per epoch. "
                                 "Consider BACKBONE = efficientnet_b0, BATCH_SIZE = 8."))

    out.append(Chk(INFO, f"Epochs {cfg.NUM_EPOCHS} · batch {cfg.BATCH_SIZE} · "
                         f"{cfg.SCHEDULER} LR · early stop {cfg.EARLY_STOP}"))

    if getattr(cfg, "USE_AMP", False) and probe is not None and probe.get("cuda") is False:
        out.append(Chk(INFO, "USE_AMP = True is ignored on CPU (CUDA-only)."))
    if getattr(cfg, "PIN_MEMORY", False) and probe is not None and probe.get("cuda") is False:
        out.append(Chk(INFO, "PIN_MEMORY = True has no effect without CUDA."))
    if os.name == "nt" and getattr(cfg, "NUM_WORKERS", 0) > 0:
        out.append(Chk(INFO, f"NUM_WORKERS = {cfg.NUM_WORKERS} — set 0 in config.py only "
                             "if the DataLoader errors on Windows."))
    return out


def checks_transfer(cfg, meta) -> list[Chk]:
    out: list[Chk] = []
    if cfg is None:
        return [Chk(BLOCK, f"config.py could not be imported ({CONFIG_ERROR})")]

    model = cfg.MODEL_PATH
    if not model.exists():
        return [Chk(BLOCK, "No trained model yet — run “Start training” first.")]

    size_mb = model.stat().st_size / 1e6
    out.append(Chk(OK, f"{model.name} — {size_mb:.1f} MB"))
    if meta:
        if meta.get("backbone"):
            out.append(Chk(OK, f"Backbone : {meta['backbone']}"))
        if meta.get("epoch") is not None:
            acc = meta.get("val_acc")
            acc_txt = f"  ·  val acc {acc*100:.2f}%" if isinstance(acc, (int, float)) else ""
            out.append(Chk(OK, f"Epoch {meta['epoch']}{acc_txt}"))
        ck_classes = meta.get("classes")
        if ck_classes:
            out.append(Chk(OK, f"Classes : {', '.join(map(str, ck_classes))}"))
            cfg_classes = list(getattr(cfg, "CLASSES", {}))
            if list(map(str, ck_classes)) != cfg_classes:
                out.append(Chk(WARN, "Checkpoint classes differ from config.CLASSES — "
                                     "the model decides names; LEDs use config.CLASSES."))
    else:
        out.append(Chk(INFO, "Reading checkpoint metadata …"))

    out.append(Chk(INFO, "Inference needs only this one file — copy it to the Pi."))
    return out


def checks_run(cfg, meta) -> list[Chk]:
    out: list[Chk] = []
    if cfg is None:
        return [Chk(BLOCK, f"config.py could not be imported ({CONFIG_ERROR})")]

    if not cfg.MODEL_PATH.exists():
        out.append(Chk(BLOCK, "No trained model at models/best_model.pth — train first."))
    else:
        out.append(Chk(OK, "Trained model found (models/best_model.pth)."))
        if meta and meta.get("backbone"):
            out.append(Chk(OK, f"Backbone : {meta['backbone']}"))
        if meta and meta.get("classes"):
            cfg_classes = list(getattr(cfg, "CLASSES", {}))
            if list(map(str, meta["classes"])) != cfg_classes:
                out.append(Chk(WARN, "Checkpoint classes differ from config.CLASSES — a "
                                     "class missing from CLASSES outputs 0 on the LEDs."))

    ini = BASE_DIR / "regions" / "region_config.ini"
    if ini.exists():
        try:
            vals = {}
            for line in ini.read_text().splitlines():
                if "=" in line and not line.strip().startswith("["):
                    k, v = line.split("=", 1)
                    vals[k.strip()] = int(v.strip())
            ordered = vals.get("left", 0) < vals.get("right", 0) and \
                vals.get("top", 0) < vals.get("bottom", 0)
            ok_size = all(vals.get(k, 0) >= 40 for k in ("left", "top", "right", "bottom"))
            if ordered and ok_size:
                out.append(Chk(OK, f"Region saved: L{vals.get('left')} T{vals.get('top')} "
                                   f"R{vals.get('right')} B{vals.get('bottom')}"))
            else:
                out.append(Chk(WARN, "regions/region_config.ini looks off — press ‘e’ in "
                                     "the feed and re-save the rectangle."))
        except Exception:
            out.append(Chk(WARN, "regions/region_config.ini could not be parsed."))
    else:
        out.append(Chk(INFO, "No saved region — the whole frame is used until you press ‘e’."))

    if platform.system() == "Linux":
        out.append(Chk(INFO, "On the Pi run detection with: sudo python3 run.py "
                             "(GPIO LEDs need sudo + the system python)."))
    else:
        out.append(Chk(INFO, "On this machine run with --no-gpio (no GPIO hardware)."))
    return out


def checks_tests(cfg, probe) -> list[Chk]:
    out: list[Chk] = []
    if probe is None:
        out.append(Chk(PEND, "Probing packages …"))
        return out
    if probe["packages"].get("cv2"):
        out.append(Chk(OK, f"OpenCV {probe['packages']['cv2']} — camera test can run."))
    else:
        out.append(Chk(WARN, "opencv-python is not installed — the camera test cannot "
                             "run (the LED test still can)."))
    out.append(Chk(INFO, "Camera test needs a display; press ‘q’ in its window to quit."))
    if platform.system() == "Linux":
        out.append(Chk(INFO, "LED test needs GPIO access — run it with sudo on the Pi."))
    else:
        out.append(Chk(INFO, "LED test is Pi-only; it will report missing GPIO here."))
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Subprocess probes (keep heavy imports out of the GUI process)
# ──────────────────────────────────────────────────────────────────────────────

_PROBE_SRC = r"""
import importlib, json, sys
info = {"executable": sys.executable, "version": sys.version.split()[0], "packages": {}}
for name in %r:
    try:
        mod = importlib.import_module(name)
        info["packages"][name] = getattr(mod, "__version__", "installed")
    except Exception:
        info["packages"][name] = None
try:
    import torch
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
except Exception:
    info["cuda"] = None
print("__PROBE__" + json.dumps(info))
"""

_CKPT_SRC = r"""
import json, sys
import torch
path = sys.argv[1]
try:
    ck = torch.load(path, map_location="cpu", weights_only=True)
except Exception:
    ck = torch.load(path, map_location="cpu")
meta = {
    "epoch": ck.get("epoch"),
    "val_acc": ck.get("val_acc"),
    "classes": ck.get("classes"),
    "backbone": ck.get("backbone"),
}
print("__CKPT__" + json.dumps(meta))
"""


def _run_probe(src: str, *args: str) -> str | None:
    """Run a short probe script and return its sentinel-marked JSON line."""
    try:
        proc = subprocess.run(
            [PYTHON, "-c", src, *args],
            cwd=str(BASE_DIR), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
    except Exception:
        return None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__PROBE__") or line.startswith("__CKPT__"):
            return line
    return None


def _parse(tag: str, line: str | None):
    if not line or tag not in line:
        return None
    try:
        return json.loads(line.split(tag, 1)[1])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  Step model
# ──────────────────────────────────────────────────────────────────────────────

class Step:
    """One pipeline step: description, checks, actions and option chips."""

    def __init__(self, key, title, blurb, commands, check, actions, options=()):
        self.key = key
        self.title = title
        self.blurb = blurb
        self.commands = commands            # [(label, [argv])] shown as the exact CLI
        self.check = check                  # callable -> list[Chk]
        self.actions = actions              # [(label, callable)]
        self.options = list(options)        # [(chip label, argv fragment)]


def _cmd_str(argv: list[str]) -> str:
    py = Path(argv[0]).name
    parts = [py] + argv[1:]
    return " ".join(p if " " not in p else f'"{p}"' for p in parts)


def _split_args(text: str) -> list[str]:
    """Split a shell-ish argument string into tokens, honouring quotes.

    Deliberately simple and identical on Windows and POSIX (no shlex), so a
    typed argument list behaves the same everywhere.
    """
    args: list[str] = []
    cur = ""
    quote: str | None = None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            else:
                cur += ch
        elif ch in "\"'":
            quote = ch
        elif ch.isspace():
            if cur:
                args.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        args.append(cur)
    return args


# GUI text editors we know how to launch, most preferred first. Console editors
# (vi/nano) are intentionally excluded — they need a terminal the GUI can't give.
_EDITOR_CANDIDATES = [
    "code", "code-insiders", "codium", "subl", "notepad++", "npp",
    "gedit", "kate", "mousepad", "xed", "leafpad",
]


def _detect_editor() -> str:
    """Best guess at a GUI text-editor command, or '' if none is found."""
    for env in ("VISUAL", "EDITOR"):
        val = os.environ.get(env, "").strip()
        if val and shutil.which(val.split()[0]):
            return val
    for name in _EDITOR_CANDIDATES:
        if shutil.which(name):
            return name
    if os.name == "nt":
        notepad = (Path(os.environ.get("SystemRoot", r"C:\Windows"))
                   / "System32" / "notepad.exe")
        if notepad.exists():
            return str(notepad)
    return ""


def _windows_txt_command() -> list[str] | None:
    """argv prefix for the program Windows associates with .txt files.

    Reading the association (instead of using startfile on the .py) matters:
    the default handler for ``.py`` *executes* the file, whereas the ``.txt``
    handler opens a text editor.
    """
    try:
        import winreg
    except Exception:
        return None

    def query(root, sub, name):
        try:
            with winreg.OpenKey(root, sub) as k:
                return winreg.QueryValueEx(k, name)[0]
        except OSError:
            return None

    progid = (query(winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer"
                    r"\FileExts\.txt\UserChoice", "ProgId")
              or query(winreg.HKEY_CLASSES_ROOT, ".txt", ""))
    if not progid:
        return None
    cmd = query(winreg.HKEY_CLASSES_ROOT, rf"{progid}\shell\open\command", "")
    if not cmd:
        return None
    # Drop the "%1"/"%L" placeholder — the file path is appended by the caller.
    argv = [a for a in _split_args(cmd) if not a.startswith("%")]
    return argv or None


def _linux_txt_command() -> list[str] | None:
    """argv prefix for the desktop app associated with text/plain on Linux."""
    xdg_mime = shutil.which("xdg-mime")
    if not xdg_mime:
        return None
    try:
        desktop = subprocess.run([xdg_mime, "query", "default", "text/plain"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout.strip()
    except Exception:
        return None
    if not desktop:
        return None
    for folder in (Path.home() / ".local/share/applications",
                   Path("/usr/local/share/applications"),
                   Path("/usr/share/applications")):
        entry = folder / desktop
        if not entry.exists():
            continue
        for line in entry.read_text(errors="replace").splitlines():
            if line.startswith("Exec="):
                argv = [a for a in _split_args(line[5:]) if not a.startswith("%")]
                if argv:
                    return argv
    return None


def _system_default_txt_command() -> list[str] | None:
    """How to open a file with the OS default program for .txt files."""
    if os.name == "nt":
        return _windows_txt_command()
    if platform.system() == "Darwin":
        return ["open", "-t"]          # TextEdit / default text handler
    return _linux_txt_command()


# ──────────────────────────────────────────────────────────────────────────────
#  The application
# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PI Detection — Pipeline Wizard")
        self.geometry("1150x760")
        self.minsize(940, 620)

        self.proc: subprocess.Popen | None = None
        self.pending: list[list[str]] = []
        self.run_label = ""
        self.q: queue.Queue = queue.Queue()
        self.probe = None
        self.ckpt = None
        self.current = 0
        self.pi_host = tk.StringVar(value="pi@raspberrypi.local")
        self.arg_vars: dict[str, tk.StringVar] = {}
        self.editor_cmd = tk.StringVar(value="")   # blank = OS default for .txt
        self._overlay: tk.Frame | None = None

        self.steps = self._build_steps()
        self._build_style()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.select_step(0)
        self.refresh_all()
        self.update_idletasks()
        self.start_probe()          # modal overlay until the environment is confirmed
        self.after(100, self._poll)

    # ── steps ────────────────────────────────────────────────────────────────
    def _build_steps(self) -> list[Step]:
        py = PYTHON

        return [
            Step(
                "env", "Environment",
                "Check the interpreter, required packages and the compute device.\n"
                "The wizard uses the project venv interpreter when it exists.",
                [("Install dependencies", [py, "-m", "pip", "install", "-r", "requirements.txt"])],
                lambda: checks_environment(self.probe),
                [("Install requirements", lambda: self.start(
                    [[py, "-m", "pip", "install", "-r", "requirements.txt"]],
                    "pip install -r requirements.txt")),
                 ("Install optional omnixai (+ re-pin)", lambda: self.start(
                    [[py, "-m", "pip", "install", "omnixai[vision,plot]"],
                     [py, "-m", "pip", "install", "-r", "requirements.txt"]],
                    "pip install omnixai[vision,plot]")),
                 ("Re-probe", self.start_probe)],
            ),
            Step(
                "classes", "Configure classes",
                "CLASSES in config.py defines your classes and their 4-bit LED numbers.\n"
                "This panel only reports on it — edit config.py yourself.",
                [("Create source folders", [py, "setup_data.py", "--init"])],
                lambda: checks_classes(config),
                [("Create source folders", lambda: self.start(
                    [[py, "setup_data.py", "--init"]], "setup_data.py --init")),
                 ("Open config.py in editor", self._open_config),
                 ("Reload config.py", self._reload_config)],
            ),
            Step(
                "collect", "Collect images",
                "Put images in source/<class>/ — capture them live or drop them in.\n"
                "Aim for 50–200 per class, varied angles and lighting.",
                [("Capture (webcam)", [py, "capture_images.py"])],
                lambda: checks_source(config),
                [("Capture images (webcam)", lambda: self.start(
                    [[py, "capture_images.py"]], "capture_images.py")),
                 ("Open source/ folder", lambda: self._open_path(
                     config.SOURCE_DIR if config else BASE_DIR / "source"))],
            ),
            Step(
                "split", "Split into train / val",
                "Shuffles source/ into data/train and data/val (wipes data/ first).\n"
                f"Ratio {getattr(config, 'VAL_SPLIT', 0.2)} · seed "
                f"{getattr(config, 'SPLIT_SEED', 42)}.",
                [("Split", [py, "setup_data.py"])],
                lambda: checks_split(config),
                [("Split into train / val", lambda: self.start(
                    [[py, "setup_data.py"]], "setup_data.py")),
                 ("Open data/ folder", lambda: self._open_path(
                     config.DATA_DIR if config else BASE_DIR / "data"))],
            ),
            Step(
                "train", "Train",
                "Fine-tunes the ImageNet-pretrained backbone and saves the best\n"
                "checkpoint to models/best_model.pth. Long-running.",
                [("Train", [py, "train.py"])],
                lambda: checks_train(config, self.probe),
                [("Start training", lambda: self.start(
                    [[py, "train.py"]], "train.py")),
                 ("Open models/ folder", lambda: self._open_path(
                     config.MODEL_DIR if config else BASE_DIR / "models"))],
            ),
            Step(
                "transfer", "Transfer to Pi",
                "Copy models/best_model.pth to the Pi (inference needs only this file).",
                [("scp", ["scp", str(config.MODEL_PATH) if config else "models/best_model.pth",
                          "<pi-host>:/home/pi/pidetection/models/"])],
                lambda: checks_transfer(config, self.ckpt),
                [("Copy scp command", self._copy_scp),
                 ("Run scp now", self._run_scp)],
            ),
            Step(
                "tests", "Hardware tests",
                "Standalone sanity checks for the camera and the GPIO LEDs.",
                [("Camera", [py, "tests/test_camera.py"]),
                 ("LED (single)", [py, "tests/test_led.py"]),
                 ("LED (binary)", [py, "tests/test_led.py", "--mode", "binary"])],
                lambda: checks_tests(config, self.probe),
                [("Camera test", lambda: self.start(
                    [[py, "tests/test_camera.py"]], "tests/test_camera.py")),
                 ("LED test (single)", lambda: self.start(
                    [[py, "tests/test_led.py"]], "tests/test_led.py")),
                 ("LED test (binary — needs a terminal)", lambda: self.start(
                    [[py, "tests/test_led.py", "--mode", "binary"]],
                    "tests/test_led.py --mode binary"))],
                options=[("--mode single", "--mode single"),
                         ("--mode binary", "--mode binary")],
            ),
            Step(
                "run", "Run detection",
                "Live camera inference with GradCAM. GPIO LEDs are on by default;\n"
                "use --no-gpio on a machine without GPIO.",
                [("No GPIO (PC)", [py, "run.py", "--no-gpio"]),
                 ("GPIO (Pi)", [py, "run.py"])],
                lambda: checks_run(config, self.ckpt),
                [("Run (no GPIO)", lambda: self.start(
                    [[py, "run.py", "--no-gpio"]], "run.py --no-gpio")),
                 ("Run with GPIO", lambda: self.start(
                    [[py, "run.py"]], "run.py")),
                 ("Copy sudo command", lambda: self._copy(
                     "sudo python3 run.py", "sudo command copied"))],
                options=[("--no-gpio", "--no-gpio")],
            ),
        ]

    # ── styling & layout ─────────────────────────────────────────────────────
    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview", rowheight=26)
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold")
                        if os.name == "nt" else ("TkDefaultFont", 15, "bold"))
        style.configure("Step.TLabel", font=("Segoe UI", 13, "bold")
                        if os.name == "nt" else ("TkDefaultFont", 13, "bold"))
        style.configure("Muted.TLabel", foreground="#57606a")
        style.configure("Run.TButton", font=("TkDefaultFont", 10, "bold"))

    def _build_layout(self) -> None:
        # Header
        head = ttk.Frame(self, padding=(12, 10, 12, 4))
        head.pack(fill="x")
        ttk.Label(head, text="PI Detection — Pipeline Wizard", style="Title.TLabel")\
            .pack(side="left")
        ttk.Button(head, text="Re-check all", command=self.refresh_all)\
            .pack(side="right")
        ttk.Button(head, text="Next ▸", command=lambda: self._nudge(1))\
            .pack(side="right", padx=6)
        ttk.Button(head, text="◂ Prev", command=lambda: self._nudge(-1))\
            .pack(side="right")

        sub = ttk.Frame(self, padding=(14, 0, 12, 6))
        sub.pack(fill="x")
        ttk.Label(sub, text=f"Project: {BASE_DIR}    ·    Interpreter: {PYTHON}",
                  style="Muted.TLabel").pack(side="left")
        ttk.Button(sub, text="Open config.py in editor", command=self._open_config)\
            .pack(side="right")
        ttk.Label(sub, text="(blank = OS default for .txt)", style="Muted.TLabel")\
            .pack(side="right", padx=(0, 8))
        ttk.Entry(sub, textvariable=self.editor_cmd, width=16).pack(side="right", padx=6)
        ttk.Label(sub, text="Editor:", style="Muted.TLabel").pack(side="right")

        # Main split: steps list | detail
        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        left = ttk.Frame(main)
        main.add(left, weight=0)
        ttk.Label(left, text="Steps", style="Step.TLabel").pack(anchor="w", pady=(0, 4))
        self.tree = ttk.Treeview(left, columns=("info",), show="tree headings",
                                 selectmode="browse", height=16)
        self.tree.heading("#0", text="Step", anchor="w")
        self.tree.heading("info", text="Status", anchor="w")
        self.tree.column("#0", width=200, stretch=False, minwidth=150)
        self.tree.column("info", width=320, stretch=True, minwidth=160)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self._step_ids = {}
        for i, step in enumerate(self.steps):
            iid = self.tree.insert("", "end", text=f"  {step.title}")
            self._step_ids[iid] = i
        for status, color in _COLOR.items():
            self.tree.tag_configure(status, foreground=color)

        right = ttk.Frame(main)
        main.add(right, weight=1)
        self.detail = right
        self.detail_title = ttk.Label(right, text="", style="Step.TLabel")
        self.detail_title.pack(anchor="w")
        self.detail_blurb = ttk.Label(right, text="", justify="left",
                                      style="Muted.TLabel", wraplength=720)
        self.detail_blurb.pack(anchor="w", pady=(2, 8))

        ttk.Label(right, text="Pre-flight checks").pack(anchor="w")
        self.checks_text = tk.Text(right, height=12, wrap="word", relief="flat",
                                   background="#f6f8fa", padx=10, pady=8,
                                   font=("Consolas", 9) if os.name == "nt"
                                   else ("TkFixedFont", 9))
        self.checks_text.pack(fill="both", expand=True, pady=(2, 8))
        for status, color in _COLOR.items():
            self.checks_text.tag_configure(status, foreground=color)
        self.checks_text.configure(state="disabled")

        ttk.Label(right, text="Arguments").pack(anchor="w")
        self.args_frame = ttk.Frame(right)
        self.args_frame.pack(fill="x", pady=(2, 0))
        self.args_hint = ttk.Label(right, text="", style="Muted.TLabel")
        self.args_hint.pack(anchor="w", pady=(0, 8))

        ttk.Label(right, text="Commands").pack(anchor="w")
        self.cmd_text = tk.Text(right, height=3, wrap="none", relief="flat",
                                background="#f6f8fa", padx=10, pady=6,
                                font=("Consolas", 9) if os.name == "nt"
                                else ("TkFixedFont", 9))
        self.cmd_text.pack(fill="x", pady=(2, 8))
        self.cmd_text.tag_configure("cmd", foreground="#0a3069")
        self.cmd_text.configure(state="disabled")

        self.actions_frame = ttk.Frame(right)
        self.actions_frame.pack(anchor="w", pady=(0, 4))
        self.extra_frame = ttk.Frame(right)
        self.extra_frame.pack(anchor="w")
        self.gate_label = ttk.Label(right, text="", foreground=_COLOR[BLOCK])
        self.gate_label.pack(anchor="w", pady=(4, 0))

        # Output console
        out = ttk.LabelFrame(self, text="Output", padding=8)
        out.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        bar = ttk.Frame(out)
        bar.pack(fill="x", pady=(0, 6))
        self.status_label = ttk.Label(bar, text="Idle.", style="Muted.TLabel")
        self.status_label.pack(side="left")
        ttk.Button(bar, text="Clear log", command=self._clear_log).pack(side="right")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self._stop,
                                   state="disabled")
        self.stop_btn.pack(side="right", padx=6)
        self.console = ScrolledText(out, height=12, wrap="word", state="disabled",
                                    background="#0d1117", foreground="#c9d1d9",
                                    insertbackground="#c9d1d9",
                                    font=("Consolas", 9) if os.name == "nt"
                                    else ("TkFixedFont", 9))
        self.console.pack(fill="both", expand=True)
        self.console.tag_configure("cmd", foreground="#79c0ff")
        self.console.tag_configure("err", foreground="#ff7b72")
        self.console.tag_configure("ok", foreground="#7ee787")

    # ── selection / detail rendering ──────────────────────────────────────────
    def select_step(self, index: int) -> None:
        iid = list(self._step_ids)[index]
        self.tree.selection_set(iid)
        self.tree.focus(iid)

    def _nudge(self, delta: int) -> None:
        ids = list(self._step_ids)
        cur = self.tree.selection()
        idx = ids.index(cur[0]) if cur else 0
        self.select_step(max(0, min(len(ids) - 1, idx + delta)))

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self.current = self._step_ids[sel[0]]
        self._render_detail()

    def _render_detail(self) -> None:
        step = self.steps[self.current]
        self.detail_title.configure(text=step.title)
        self.detail_blurb.configure(text=step.blurb)

        # Arguments (extra CLI flags, appended to whatever action you run)
        for w in self.args_frame.winfo_children():
            w.destroy()
        ttk.Entry(self.args_frame, textvariable=self._arg_var(step), width=44)\
            .pack(side="left")
        for chip_label, fragment in step.options:
            ttk.Button(self.args_frame, text=chip_label,
                       command=lambda f=fragment: self._append_arg(f))\
                .pack(side="left", padx=(4, 0))
        self.args_hint.configure(
            text="Extra arguments appended to every command below — click an "
                 "option to add it." if step.options
            else "Extra arguments appended to every command below "
                 "(this script defines no options).")

        self._render_commands()

        # Actions
        for w in self.actions_frame.winfo_children():
            w.destroy()
        for w in self.extra_frame.winfo_children():
            w.destroy()
        if step.key == "transfer":
            ttk.Label(self.extra_frame, text="Pi host:").pack(side="left", padx=(0, 6))
            ttk.Entry(self.extra_frame, textvariable=self.pi_host, width=30)\
                .pack(side="left")
        for i, (label, fn) in enumerate(step.actions):
            style = "Run.TButton" if i == 0 else "TButton"
            ttk.Button(self.actions_frame, text=label, command=fn, style=style)\
                .pack(side="left", padx=(0, 6))

        checks = self._checks_for(step)
        self._render_checks(checks)
        self._render_gate(step, checks)

    def _checks_for(self, step: Step) -> list[Chk]:
        try:
            return step.check()
        except Exception as exc:  # never let a bad check crash the UI
            return [Chk(BLOCK, f"Check failed: {type(exc).__name__}: {exc}")]

    def _arg_var(self, step) -> tk.StringVar:
        """Per-step extra-arguments text, persisted across step switches."""
        var = self.arg_vars.get(step.key)
        if var is None:
            var = tk.StringVar(value="")
            var.trace_add("write", lambda *_: self._render_commands())
            self.arg_vars[step.key] = var
        return var

    def _append_arg(self, fragment: str) -> None:
        var = self._arg_var(self.steps[self.current])
        var.set((var.get().rstrip() + " " + fragment).strip())

    def _render_commands(self) -> None:
        """Show the effective command for the current step (base + extra args)."""
        step = self.steps[self.current]
        extra = _split_args(self._arg_var(step).get())
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        for label, argv in step.commands:
            shown = [str(a) for a in argv] + extra
            self.cmd_text.insert("end", f"# {label}\n", ())
            self.cmd_text.insert("end", f"{_cmd_str(shown)}\n", ("cmd",))
        self.cmd_text.configure(state="disabled")

    def _render_checks(self, checks: list[Chk]) -> None:
        self.checks_text.configure(state="normal")
        self.checks_text.delete("1.0", "end")
        for c in checks:
            self.checks_text.insert("end", f"{_SYMBOL[c.status]}  ", (c.status,))
            self.checks_text.insert("end", f"{c.text}\n", (c.status,))
        self.checks_text.configure(state="disabled")
        self._update_tree_badge(self.current, checks)

    def _render_gate(self, step: Step, checks: list[Chk]) -> None:
        """A step is blocked (and its primary action disabled) only by BLOCK
        checks — warnings never disable anything."""
        blocked = blockers(checks)
        if blocked:
            self.gate_label.configure(text="Blocked — " + blocked[0].text,
                                      foreground=_COLOR[BLOCK])
        else:
            self.gate_label.configure(text="", foreground=_COLOR[OK])
        children = self.actions_frame.winfo_children()
        if children and self.proc is None:
            children[0].configure(state="disabled" if blocked else "normal")

    def _update_tree_badge(self, index: int, checks: list[Chk]) -> None:
        """Left column: the run/block check. Right column: the top warning or
        block reason, so you see it without opening the step."""
        iid = list(self._step_ids)[index]
        step = self.steps[index]
        text, sev = summary(checks)
        info = f"{_SYMBOL[sev]} {_truncate(text)}" if text else ""
        self.tree.item(
            iid,
            text=f"  {_SYMBOL[badge(checks)]}  {step.title}",
            values=(info,),
            tags=(sev,),   # row colour follows the message (warn/block/pending)
        )

    # ── refreshing ────────────────────────────────────────────────────────────
    def refresh_all(self) -> None:
        for i, step in enumerate(self.steps):
            self._update_tree_badge(i, self._checks_for(step))
        if getattr(self, "current", None) is not None:
            self._render_detail()

    def start_probe(self) -> None:
        """Probe the interpreter/packages in the background.

        The modal overlay stays up — and every step stays unclickable — until
        the probe reports back, so nothing is started on an unverified env.
        """
        self.probe = None
        self._show_overlay("Checking the environment …")

        def work():
            try:
                line = _run_probe(_PROBE_SRC % (_REQUIRED_PKGS + _OPTIONAL_PKGS,))
                info = _parse("__PROBE__", line) or {"error": "probe produced no output"}
            except Exception as exc:      # never leave the overlay stuck
                info = {"error": f"{type(exc).__name__}: {exc}"}
            self.q.put(("probe", info))

        threading.Thread(target=work, daemon=True).start()

    # ── modal overlay ─────────────────────────────────────────────────────────
    def _show_overlay(self, message: str) -> None:
        """Grey, click-blocking cover shown while the environment is checked."""
        if self._overlay is not None:
            self._overlay_label.configure(text=message)
            return

        ov = tk.Frame(self, background="#c9ccd1")     # the grey field
        ov.place(x=0, y=0, relwidth=1, relheight=1)

        card = tk.Frame(ov, background="#ffffff", highlightthickness=1,
                        highlightbackground="#8c959f")
        card.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(card, text="PI Detection", background="#ffffff",
                 font=("Segoe UI", 14, "bold") if os.name == "nt"
                 else ("TkDefaultFont", 14, "bold")).pack(padx=48, pady=(26, 2))
        self._overlay_label = tk.Label(card, text=message, background="#ffffff",
                                       foreground="#57606a", wraplength=320,
                                       justify="center")
        self._overlay_label.pack(padx=48, pady=(0, 12))
        bar = ttk.Progressbar(card, mode="indeterminate", length=260)
        bar.pack(padx=48, pady=(0, 26))
        bar.start(12)

        ov.lift()
        try:
            ov.grab_set()        # swallow clicks/keys aimed at the widgets below
        except tk.TclError:
            pass                 # the covering frame still blocks the mouse
        self._overlay = ov

    def _hide_overlay(self) -> None:
        if self._overlay is None:
            return
        try:
            self._overlay.grab_release()
        except tk.TclError:
            pass
        self._overlay.destroy()
        self._overlay = None

    def start_ckpt_probe(self) -> None:
        if not (config and config.MODEL_PATH.exists()):
            self.ckpt = None
            return

        def work():
            line = _run_probe(_CKPT_SRC, str(config.MODEL_PATH))
            meta = _parse("__CKPT__", line)
            self.q.put(("ckpt", meta))

        threading.Thread(target=work, daemon=True).start()

    # ── subprocess runner ─────────────────────────────────────────────────────
    def start(self, commands: list[list[str]], label: str,
              use_args: bool = True) -> None:
        if self.proc is not None:
            messagebox.showinfo("Busy", "A step is already running — stop it first.")
            return
        # Append the current step's extra CLI arguments (if any) to every command.
        if use_args:
            extra = _split_args(self._arg_var(self.steps[self.current]).get())
            if extra:
                commands = [list(c) + extra for c in commands]
        self.pending = [list(c) for c in commands]
        self.run_label = label
        self._start_next()

    def _start_next(self) -> None:
        if not self.pending:
            return
        argv = [str(a) for a in self.pending.pop(0)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"        # stream output line-by-line
        env["PYTHONIOENCODING"] = "utf-8"    # the scripts print box-drawing chars
        env["PYTHONUTF8"] = "1"
        try:
            self.proc = subprocess.Popen(
                argv, cwd=str(BASE_DIR), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=None, text=True, bufsize=1,
                encoding="utf-8", errors="replace", env=env,
            )
        except Exception as exc:
            self._log(f"✖ Could not start: {exc}", "err")
            self.proc = None
            return

        self._log(f"\n$ {_cmd_str(argv)}", "cmd")
        self.status_label.configure(text=f"Running: {self.run_label}")
        self.stop_btn.configure(state="normal")
        self._set_actions_enabled(False)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.q.put(("log", line.rstrip("\r\n")))
        finally:
            proc.wait()
            self.q.put(("done", proc.returncode))

    def _stop(self) -> None:
        if self.proc is None:
            return
        self._log("… stopping …", "err")
        self.pending = []
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _on_done(self, code: int) -> None:
        self.proc = None
        if self.pending:
            self._start_next()
            return
        good = code == 0
        self._log(f"[{self.run_label}] exited with code {code}  "
                  f"{'✓' if good else '✖'}", "ok" if good else "err")
        self.status_label.configure(text=f"Idle. Last step exited with code {code}.")
        self.stop_btn.configure(state="disabled")
        self._set_actions_enabled(True)
        self.start_ckpt_probe()      # model may have been re-written
        self.refresh_all()

    def _set_actions_enabled(self, enabled: bool) -> None:
        for w in self.actions_frame.winfo_children():
            w.configure(state="normal" if enabled else "disabled")
        if enabled:
            step = self.steps[self.current]
            self._render_gate(step, self._checks_for(step))   # re-apply blocking

    # ── log helpers ───────────────────────────────────────────────────────────
    def _log(self, text: str, tag: str | None = None) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n", (tag,) if tag else ())
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_log(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    # ── misc actions ──────────────────────────────────────────────────────────
    def _copy(self, text: str, note: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_label.configure(text=note)

    def _scp_command(self) -> str:
        model = str(config.MODEL_PATH) if config else "models/best_model.pth"
        return f"scp \"{model}\" {self.pi_host.get()}:/home/pi/pidetection/models/"

    def _copy_scp(self) -> None:
        cmd = self._scp_command()
        self._copy(cmd, "scp command copied to clipboard")
        self._log(f"# copy this to the Pi:\n$ {cmd}", "cmd")

    def _run_scp(self) -> None:
        model = str(config.MODEL_PATH) if config else "models/best_model.pth"
        target = f"{self.pi_host.get()}:/home/pi/pidetection/models/"
        self.start([["scp", model, target]], f"scp → {self.pi_host.get()}")

    def _open_path(self, path: Path) -> None:
        try:
            if not path.exists():
                messagebox.showinfo("Not found", f"{path} does not exist yet.")
                return
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showwarning("Could not open", f"{path}\n{exc}")

    def _open_in_editor(self, path: Path) -> None:
        """Open a source file in a text editor — never execute it.

        Uses the “Editor:” box if it has a command; otherwise the OS default
        program for .txt (whatever Windows associates with .txt, the
        ``text/plain`` app on Linux, TextEdit on macOS). The .txt association is
        used on purpose: the default handler for .py *runs* the file.
        """
        editor = self.editor_cmd.get().strip()
        if editor:
            argv = _split_args(editor)
            where = f"“{Path(argv[0]).name}”"
        else:
            base = _system_default_txt_command()
            where = "the system default for .txt"
            if base is None:                      # no .txt association found
                detected = _detect_editor()
                if not detected:
                    messagebox.showinfo(
                        "No editor found",
                        f"Could not find a program to open:\n{path}\n\n"
                        "Type an editor command in the “Editor:” box "
                        "(e.g. code, notepad++) or set the EDITOR env var.")
                    return
                base = _split_args(detected)
                where = f"“{Path(base[0]).name}”"
            argv = base

        argv = argv + [str(path)]
        try:
            subprocess.Popen(argv, cwd=str(BASE_DIR))
        except Exception as exc:
            messagebox.showwarning("Could not open editor",
                                   f"Tried: {' '.join(argv)}\n{exc}")
            return
        self.status_label.configure(text=f"Opened {path.name} in {where}.")

    def _open_config(self) -> None:
        self._open_in_editor(BASE_DIR / "config.py")

    def _reload_config(self) -> None:
        """Re-import config.py so checks reflect edits made in the editor."""
        if config is None:
            messagebox.showerror("Reload failed",
                                 CONFIG_ERROR or "config.py is unavailable")
            return
        try:
            importlib.reload(config)
        except Exception as exc:
            messagebox.showerror(
                "Reload failed",
                f"config.py has an error:\n{type(exc).__name__}: {exc}")
            return
        self.steps = self._build_steps()      # re-read values quoted in blurbs
        self.refresh_all()
        self.status_label.configure(text="config.py reloaded.")

    # ── event pump ────────────────────────────────────────────────────────────
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "probe":
                    self.probe = payload
                    self._hide_overlay()
                    self.refresh_all()
                elif kind == "ckpt":
                    self.ckpt = payload
                    self.refresh_all()
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _on_close(self) -> None:
        if self.proc is not None:
            if not messagebox.askyesno("Quit", "A step is still running. Stop it and quit?"):
                return
            self._stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
