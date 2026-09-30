"""
train.py — Full training pipeline for the OmniXAI object-recognition model.

Features
────────
• EfficientNet-B0 / B2 / ResNet-50 backbones (ImageNet pre-trained)
• Rich augmentation: flip, rotate, colour-jitter, random-erase, MixUp
• Mixed-precision (AMP) for RTX A2000 — faster training, lower VRAM
• Cosine LR annealing (or a step scheduler)
• Early stopping + best-model checkpointing
• Classification report + training-curve plot saved to models/

Usage
──────
    python train.py
"""
import json
import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

import config

# ──────────────────────────────────────────────────────────────────────────────
#  Device setup - GPU or CPU
# ──────────────────────────────────────────────────────────────────────────────

def setup_device() -> torch.device:
    """Set up compute device (GPU or CPU) and print information.
    
    Returns:
        torch.device: The configured compute device (CUDA or CPU)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Fetch and print the GPU information
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU : {name}  ({vram:.1f} GB VRAM)")
        # Enable the cuDNN benchmark: faster with a fixed input size
        torch.backends.cudnn.benchmark = True
    else:
        print("  WARNING: CUDA not available. Training on CPU (slow).")
    return device


# ──────────────────────────────────────────────────────────────────────────────
#  Image transforms - augmentation and normalisation
# ──────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation values (mean and standard deviation)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create image transforms for training and validation.
    
    Training uses extensive augmentation, validation uses only normalization.
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (Training transforms, Validation transforms)
    """
    # Base training operations
    train_ops = [
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.65, 1.0)),  # random crop scaling
    ]

    # Optional augmentations, switched on via config
    if config.AUG_HFLIP:
        train_ops.append(transforms.RandomHorizontalFlip())  # horizontal flip
    if config.AUG_VFLIP:
        train_ops.append(transforms.RandomVerticalFlip())    # vertical flip
    if config.AUG_ROTATION:
        train_ops.append(transforms.RandomRotation(config.AUG_ROTATION))  # random rotation
    if config.AUG_COLOR_JITTER:
        # vary brightness, contrast, saturation and hue
        train_ops.append(
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )
        )

    # Add normalisation
    train_ops += [
        transforms.ToTensor(),  # To tensor, scaled to [0, 1]
        transforms.Normalize(_MEAN, _STD),  # normalise with the ImageNet values
    ]

    # Optional random erase (cutout)
    if config.AUG_RANDOM_ERASE:
        train_ops.append(
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
        )

    # Validation operations (no augmentation, normalisation only)
    val_ops = [
        transforms.Resize(int(config.IMG_SIZE * 1.15)),  # slightly enlarge
        transforms.CenterCrop(config.IMG_SIZE),  # centre crop
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]

    return transforms.Compose(train_ops), transforms.Compose(val_ops)


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading - PyTorch DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders(
    train_tfm: transforms.Compose,
    val_tfm: transforms.Compose,
) -> Tuple[DataLoader, DataLoader, list]:
    """Create DataLoaders for training and validation.
    
    Args:
        train_tfm: Transforms for training data
        val_tfm: Transforms for validation data
        
    Returns:
        Tuple: (Training DataLoader, Validation DataLoader, Class list)
    """
    # Make sure the train and val folders exist
    if not config.TRAIN_DIR.exists():
        sys.exit(
            f"\n  ERROR: Training folder not found: {config.TRAIN_DIR}\n"
            "  Run `python setup_data.py` first, then add images."
        )
    if not config.VAL_DIR.exists():
        sys.exit(
            f"\n  ERROR: Validation folder not found: {config.VAL_DIR}\n"
            "  Run `python setup_data.py` first, then add images."
        )

    # Load the images as class folders
    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=train_tfm)
    val_ds   = datasets.ImageFolder(config.VAL_DIR,   transform=val_tfm)

    # Check for a mismatch between detected and configured classes
    detected = train_ds.classes
    config_classes = list(config.CLASSES.keys())
    if detected != config_classes:
        print(
            f"\n  NOTE: Detected classes differ from config.CLASSES.\n"
            f"        Detected : {detected}\n"
            f"        config   : {config_classes}\n"
            f"        The detected order will be used for this run.\n"
        )

    # Use PIN_MEMORY for faster host -> GPU copies when CUDA is available
    pin = config.PIN_MEMORY and torch.cuda.is_available()

    # Training DataLoader, shuffled
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # shuffle the training data
        num_workers=config.NUM_WORKERS,  # parallel data loading
        pin_memory=pin,  # pin memory for faster transfers
        persistent_workers=config.NUM_WORKERS > 0,  # keep the workers alive between epochs
    )
    
    # Validation DataLoader, not shuffled
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # no shuffling for validation
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    return train_loader, val_loader, detected


# ──────────────────────────────────────────────────────────────────────────────
#  Model building - EfficientNet / ResNet
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int, device: torch.device
) -> Tuple[nn.Module, nn.Module]:
    """Build a deep learning model with pre-configured backbone.
    
    Args:
        num_classes: Number of classification classes
        device: Target device (CPU or GPU)
        
    Returns:
        Tuple: (Model, GradCAM target layer for visualization)
    """
    # Pick the backbone architecture from config
    backbone = config.BACKBONE.lower()

    if backbone == "efficientnet_b0":
        # Load EfficientNet-B0 with optional ImageNet weights
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if config.PRETRAINED else None
        m = models.efficientnet_b0(weights=w)
        # Replace the final classifier with the right number of classes
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        target = m.features[-1]  # GradCAM target layer

    elif backbone == "efficientnet_b2":
        # Load EfficientNet-B2 (a bit larger and more accurate than B0)
        w = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if config.PRETRAINED else None
        m = models.efficientnet_b2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        target = m.features[-1]

    elif backbone == "resnet50":
        # Load ResNet-50 with optional ImageNet weights
        w = models.ResNet50_Weights.IMAGENET1K_V1 if config.PRETRAINED else None
        m = models.resnet50(weights=w)
        # Replace the fully connected layer
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        target = m.layer4[-1]  # GradCAM target layer

    else:
        sys.exit(f"\n  ERROR: Unknown backbone '{config.BACKBONE}'")

    # Move the model to the target device (GPU/CPU)
    return m.to(device), target


# ──────────────────────────────────────────────────────────────────────────────
#  MixUp - data augmentation
# ──────────────────────────────────────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to a batch.
    
    Mix two random images and their labels with beta-distributed weights.
    
    Args:
        x: Input images (batch)
        y: Input labels (batch)
        alpha: Beta distribution parameter (default 0.2)
        
    Returns:
        Tuple: (mixed images, original labels, permuted labels, weight lam)
    """
    # Draw the weight lambda from a Beta distribution
    lam   = float(np.random.beta(alpha, alpha))
    # Random permutation of the batch indices
    idx   = torch.randperm(x.size(0), device=x.device)
    # Mix the images: x_new = lambda * x + (1 - lambda) * x_permuted
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def mixup_loss(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Calculate loss for MixUp training examples.
    
    Combines losses for both original label samples.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: First original labels
        y_b: Second original labels (permuted)
        lam: Mix weight
        
    Returns:
        Mixed loss
    """
    # Weighted average of the two losses
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ──────────────────────────────────────────────────────────────────────────────
#  Training and validation steps
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch.
    
    Args:
        model: Model to train
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimization algorithm
        scaler: Gradient Scaler for AMP (Mixed Precision)
        device: Compute device (GPU/CPU)
        
    Returns:
        Tuple: (average loss, accuracy)
    """
    # Put the model in training mode
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    # Loop over all training batches
    for imgs, labels in tqdm(loader, desc="  train", leave=False, ncols=82):
        # Move the data to the target device
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Optional MixUp augmentation
        use_mixup = config.MIXUP_ALPHA > 0
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_batch(imgs, labels, config.MIXUP_ALPHA)

        # Clear the gradients from the previous step
        optimizer.zero_grad(set_to_none=True)

        # Forward pass, with optional mixed precision
        with torch.cuda.amp.autocast(enabled=config.USE_AMP and device.type == "cuda"):
            outputs = model(imgs)
            # Compute the loss (MixUp or plain)
            if use_mixup:
                loss = mixup_loss(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)

        # Backward pass with AMP scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping to avoid numerical instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss and accuracy
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()

    # Return the average metrics
    return total_loss / total, correct / total


@torch.no_grad()  # Disable gradient computation for faster validation
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, list, list]:
    """Validate the model without gradient computation.
    
    Args:
        model: Model to validate
        loader: Validation DataLoader
        criterion: Loss function
        device: Compute device (GPU/CPU)
        
    Returns:
        Tuple: (Loss, accuracy, predictions list, labels list)
    """
    # Put the model in eval mode (disables dropout, etc.)
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []  # Collect all predictions for the classification report
    all_labels = []  # Collect the ground-truth labels

    # Loop over all validation batches
    for imgs, labels in tqdm(loader, desc="    val", leave=False, ncols=82):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass with mixed precision (no gradients computed)
        with torch.cuda.amp.autocast(enabled=config.USE_AMP and device.type == "cuda"):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        # Accumulate the metrics
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        _, predicted = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        # Collect predictions and labels for the report below
        all_preds .extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Return the averages and the lists
    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
#  Training-history visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_history(history: dict) -> None:
    """Plot training and validation progress (loss and accuracy).
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    # Build the epoch axis
    epochs = range(1, len(history["train_loss"]) + 1)

    # Two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Training — {config.BACKBONE}", fontsize=13, weight="bold")

    # Plot the loss (left panel)
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.8)
    ax1.plot(epochs, history["val_loss"],   label="Val",   linewidth=1.8)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # Plot the accuracy in percent (right panel)
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train", linewidth=1.8)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val",   linewidth=1.8)
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.25)

    # Save and show the figure
    plt.tight_layout()
    out = config.MODEL_DIR / "training_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Training curves → {out}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  Main function - the full training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main training pipeline for OmniXAI object recognition model."""
    # Create the model output directory if it is missing
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Print the welcome banner
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║            OmniXAI  —  Training Pipeline                 ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    # Set up the compute device
    device = setup_device()

    # Print the training configuration
    print(f"  Backbone    : {config.BACKBONE}")
    print(f"  Image size  : {config.IMG_SIZE}×{config.IMG_SIZE}")
    print(f"  Batch size  : {config.BATCH_SIZE}")
    print(f"  Epochs      : {config.NUM_EPOCHS}  (early stop: {config.EARLY_STOP})")
    print(f"  AMP         : {config.USE_AMP}")
    print(f"  MixUp α     : {config.MIXUP_ALPHA}")
    print()

    # ── Data ─────────────────────────────────────────────────────────────────────
    # Build the image transforms
    train_tfm, val_tfm = build_transforms()
    # Build the DataLoaders for training and validation
    train_loader, val_loader, classes = build_loaders(train_tfm, val_tfm)
    num_classes = len(classes)

    # Print the dataset info
    print(f"  Classes     : {classes}")
    print(
        f"  Samples     : {len(train_loader.dataset)} train  /  "
        f"{len(val_loader.dataset)} val"
    )
    print()

    # ── Model ─────────────────────────────────────────────────────────────────────
    # Build the model with the configured backbone
    model, _ = build_model(num_classes, device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")
    print()

    # ── Optimiser & LR scheduler ──────────────────────────────────────────────────
    # Cross-entropy loss with label smoothing for regularisation
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    # AdamW optimiser with weight-decay regularisation
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Pick the LR scheduler from config
    if config.SCHEDULER == "cosine":
        # Cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7
        )
    else:
        # Step-wise LR reduction
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.STEP_LR_STEP,
            gamma=config.STEP_LR_DECAY,
        )

    # Gradient scaler for automatic mixed precision
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda"
    )

    # ── Training loop ────────────────────────────────────────────────────────────
    # Track the best validation accuracy
    best_val_acc     = 0.0
    # Early-stopping counter
    patience_counter = 0
    # History kept for plotting
    history: dict    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("  ┌──────────────────────────────────────────────────────────┐")

    # Best-epoch predictions and labels for the classification report
    final_preds, final_labels = [], []

    # Loop over all epochs
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Start the timer
        t0 = time.perf_counter()

        # Train one epoch
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        # Validate after training
        vl_loss, vl_acc, epoch_preds, epoch_labels = validate(
            model, val_loader, criterion, device
        )
        # Update the learning rate
        scheduler.step()

        # Elapsed time and current learning rate
        elapsed = time.perf_counter() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # Append the metrics to the history
        history["train_loss"].append(tr_loss)
        history["train_acc"] .append(tr_acc)
        history["val_loss"]  .append(vl_loss)
        history["val_acc"]   .append(vl_acc)

        # Check for a new best validation accuracy
        saved = ""
        if vl_acc > best_val_acc:
            # New best accuracy
            best_val_acc     = vl_acc
            patience_counter = 0  # reset the patience counter
            final_preds      = epoch_preds
            final_labels     = epoch_labels
            # Save the model checkpoint
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_acc":     vl_acc,
                    "classes":     classes,
                    "backbone":    config.BACKBONE,
                },
                config.MODEL_PATH,
            )
            saved = "  ✓ saved"
        else:
            # No improvement - bump the patience counter
            patience_counter += 1
            saved = f"  ({patience_counter}/{config.EARLY_STOP})"

        # Print the epoch summary
        print(
            f"  │ Ep {epoch:>3}/{config.NUM_EPOCHS}"
            f"  train {tr_acc*100:5.2f}% ℓ{tr_loss:.4f}"
            f"  val {vl_acc*100:5.2f}% ℓ{vl_loss:.4f}"
            f"  lr {lr_now:.1e}"
            f"  {elapsed:.0f}s{saved}"
        )

        # Early stop once patience is exhausted
        if patience_counter >= config.EARLY_STOP:
            print(f"  │  Early stopping after epoch {epoch}.")
            break

    print("  └──────────────────────────────────────────────────────────┘")
    print()
    print(f"  Best validation accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Model saved to           : {config.MODEL_PATH}")
    print()

    # ── Classification report ────────────────────────────────────────────────
    if final_preds:
        report = classification_report(
            final_labels, final_preds, target_names=classes, zero_division=0
        )
        print("  Classification Report (best val epoch)")
        print("  " + "─" * 55)
        for line in report.splitlines():
            print(f"    {line}")
        print()

        report_path = config.MODEL_DIR / "classification_report.txt"
        report_path.write_text(report)

    # ── Save history ──────────────────────────────────────────────────────────
    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    plot_history(history)


if __name__ == "__main__":
    main()
