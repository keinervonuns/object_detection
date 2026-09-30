"""
setup_data.py — Create source folder skeleton and split images into train / val.

Workflow
────────
1. Run once to generate the source folder tree:
       python setup_data.py --init

2. Drop ALL your images for each class into:
       source/<class_name>/

3. Run again (no flag) to shuffle-split them into data/train & data/val:
       python setup_data.py

   The split ratio is controlled by VAL_SPLIT in config.py (default 0.2).
   Re-running regenerates the split cleanly (old data/ folders are wiped).

Image extensions recognised: .jpg  .jpeg  .png  .bmp  .webp  .tiff
"""

import random
import shutil
import sys

import config

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────

def init_source() -> None:
    """Create source/<class>/ folders so the user knows where to put images."""
    for cls in config.CLASSES:
        (config.SOURCE_DIR / cls).mkdir(parents=True, exist_ok=True)

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║         OmniXAI — Source Folders Initialised            ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    for cls in config.CLASSES:
        rel = (config.SOURCE_DIR / cls).relative_to(config.BASE_DIR)
        print(f"    {rel}/")
    print()
    print("  Next steps")
    print("  ──────────")
    print("  1. Edit CLASSES in config.py to match your real object names.")
    print("  2. Drop images into each   source/<class>/   folder.")
    print("  3. Run:  python setup_data.py   (no flags) to split & prepare.")
    print()


# ──────────────────────────────────────────────────────────────────────────────

def split_data() -> None:
    """Shuffle-split images from source/ into data/train/ and data/val/."""

    if not config.SOURCE_DIR.exists():
        sys.exit(
            f"\n  ERROR: Source folder not found: {config.SOURCE_DIR}\n"
            "  Run `python setup_data.py --init` first.\n"
        )

    # Collect images from source, discover classes from sub-folders
    source_classes = sorted(
        p.name for p in config.SOURCE_DIR.iterdir() if p.is_dir()
    )
    if not source_classes:
        sys.exit(
            f"\n  ERROR: No sub-folders found inside {config.SOURCE_DIR}\n"
            "  Run `python setup_data.py --init` to create them, then add images.\n"
        )

    random.seed(config.SPLIT_SEED)

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║         OmniXAI — Splitting Source Images                ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Source     : {config.SOURCE_DIR}")
    print(f"  Val split  : {config.VAL_SPLIT * 100:.0f}%  "
          f"(seed {config.SPLIT_SEED})")
    print(f"  Mode       : {'copy' if config.COPY_FILES else 'symlink'}")
    print()

    # Wipe old train / val so we get a clean split every run
    if config.DATA_DIR.exists():
        shutil.rmtree(config.DATA_DIR)

    totals = {"train": 0, "val": 0, "skipped": 0}

    print(f"  {'Class':<22}  {'Total':>6}  {'Train':>6}  {'Val':>5}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*5}")

    for cls in source_classes:
        src_cls = config.SOURCE_DIR / cls

        images = sorted(
            p for p in src_cls.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )

        if not images:
            totals["skipped"] += 1
            print(f"  {cls:<22}  {'(no images — skipped)':>18}")
            continue

        random.shuffle(images)
        n_val   = max(1, round(len(images) * config.VAL_SPLIT))
        n_train = len(images) - n_val

        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split_name, split_dir, imgs in (
            ("train", config.TRAIN_DIR, train_imgs),
            ("val",   config.VAL_DIR,   val_imgs),
        ):
            dest_cls = split_dir / cls
            dest_cls.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                dest = dest_cls / img.name
                if config.COPY_FILES:
                    shutil.copy2(img, dest)
                else:
                    # Use absolute source path for symlinks (works cross-drive)
                    try:
                        dest.symlink_to(img.resolve())
                    except OSError:
                        # Symlinks may need developer mode on Windows — fall back to copy
                        shutil.copy2(img, dest)

        totals["train"]   += n_train
        totals["val"]     += n_val

        print(f"  {cls:<22}  {len(images):>6}  {n_train:>6}  {n_val:>5}")

    print()
    print(f"  Total — train: {totals['train']}   val: {totals['val']}")
    if totals["skipped"]:
        print(f"  Classes skipped (no images): {totals['skipped']}")
    print()
    print(f"  data/train  →  {config.TRAIN_DIR}")
    print(f"  data/val    →  {config.VAL_DIR}")
    print()
    print("  Run:  python train.py")
    print()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--init" in sys.argv:
        init_source()
    else:
        split_data()

