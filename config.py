"""
config.py — All settings for the OmniXAI Detection Pipeline.

Edit CLASSES to match your actual folder names, tweak training
hyper-parameters, then run:
    python setup_data.py   # creates folder skeleton
    python train.py        # trains the model
    python run.py         # real-time camera inference
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CLASS LABELS  — rename these to match your actual data folder names
# ══════════════════════════════════════════════════════════════════════════════
CLASSES = {
    "background": 0,
    "green_cube": 1,
    "blue_cube": 2,
    "pink_cylinder": 3,
    "silver_cylinder": 4,
    "silver_cube": 5,
    "gray_cuboid": 6
}

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).parent
SOURCE_DIR   = BASE_DIR / "source"   # ← drop ALL your raw images here (one sub-folder per class)
DATA_DIR     = BASE_DIR / "data"
TRAIN_DIR    = DATA_DIR / "train"
VAL_DIR      = DATA_DIR / "val"
MODEL_DIR    = BASE_DIR / "models"
MODEL_PATH   = MODEL_DIR / "best_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.json"

# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT
# ══════════════════════════════════════════════════════════════════════════════
VAL_SPLIT       = 0.2    # fraction of images that go to val/ (e.g. 0.2 = 80/20)
SPLIT_SEED      = 42     # fixed seed → reproducible split
COPY_FILES      = False  # False = symlink (saves disk); True = copy files

# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE = 224   # native size for EfficientNet-B0 / ResNet-50

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE    = 32      # lower to 16 if VRAM is tight
NUM_EPOCHS    = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4

# Learning-rate scheduler: "cosine" (recommended) or "step"
SCHEDULER      = "cosine"
STEP_LR_DECAY  = 0.5     # gamma for step scheduler
STEP_LR_STEP   = 10      # step size (epochs) for step scheduler

EARLY_STOP     = 8       # patience: epochs without val improvement → stop

# Label smoothing (helps with small datasets)
LABEL_SMOOTHING = 0.1

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
# Options: "efficientnet_b0"  "efficientnet_b2"  "resnet50"
# efficientnet_b0 is the fastest; b2 / resnet50 give higher accuracy
BACKBONE   = "resnet50"
PRETRAINED = True   # start from ImageNet weights

# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE  (RTX A2000)
# ══════════════════════════════════════════════════════════════════════════════
NUM_WORKERS = 4      # parallel data loading; set to 0 on Windows if errors
PIN_MEMORY  = True   # faster host → GPU transfers
USE_AMP     = True   # mixed-precision fp16/bf16 — faster & less VRAM on A2000

# ══════════════════════════════════════════════════════════════════════════════
#  DATA AUGMENTATION  (training split only)
# ══════════════════════════════════════════════════════════════════════════════
AUG_HFLIP        = True
AUG_VFLIP        = False
AUG_ROTATION     = 20     # ± degrees
AUG_COLOR_JITTER = True   # brightness / contrast / saturation / hue
AUG_RANDOM_ERASE = True   # randomly erase patches — reduces over-fitting

MIXUP_ALPHA = 0.2          # MixUp strength; 0 = disabled

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE FEED  
# ══════════════════════════════════════════════════════════════════════════════
DISPLAY_SECONDS = 1.0 # how long to wait before displaying new class after a switch (seconds)
