"""
Training Script for VMamba Classifier
=====================================

HOW TO USE:
    1. Edit the CONFIG dictionary below to set your dataset, epochs, etc.
    2. Press Run.   (No command-line arguments needed.)

    Command-line arguments still work and will OVERRIDE the config values,
    so you can also do:
        python train.py --dataset pneumoniamnist --epochs 50

Outputs are saved to:
    checkpoints/<dataset_name>/
        best_model.pth          – best checkpoint (by val accuracy)
        last_model.pth          – checkpoint after final epoch
        training_results.pth    – all metrics + history
        training_curves.png     – loss / accuracy / LR plots
        confusion_matrix.png    – test-set confusion matrix
        final_report.txt        – text summary with ACC, AUC, F1
        config.json             – snapshot of all settings used
"""

# ============================================================================
# >>>  EDIT THIS CONFIG — then just press Run  <<<
# ============================================================================

CONFIG = {
    # --- Dataset ---------------------------------------------------------
    "dataset":      "dermamnist",      # "dermamnist" | "bloodmnist" | "cmmd"
    "data_root":    "./data",          # root folder for dataset downloads
    "npz_path":     None,              # set to path string to use a custom NPZ

    # --- Model -----------------------------------------------------------
    "model_arch":   "vmamba",          # "vmamba" (hierarchical 4-direction cross-scan)
                                       #   or "vim" (plain, bidirectional 1D scan)
    "image_size":   224,
    "patch_size":   4,                 # 4 for medical (fine detail), 16 for natural / Vim
    "in_channels":  None,              # None = auto-detect from dataset
    "d_state":      16,
    "dims":         [32, 64, 128, 256],
    "depths":       [2, 2, 4, 2],
    "vim_depth":    12,                # only used when model_arch == "vim"
    "vim_d_model":  192,               # only used when model_arch == "vim"
    "vim_mlp_ratio": 4.0,              # only used when model_arch == "vim" (0 disables MLP)
    "drop_rate":    0.0,
    "drop_path_rate": 0.1,

    # --- Training --------------------------------------------------------
    "epochs":       50,
    "batch_size":   32,
    "lr":           1e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "min_lr":       1e-6,
    "grad_clip":    1.0,

    # --- System ----------------------------------------------------------
    "num_workers":  8,
    "device":       "cuda",            # "cuda" or "cpu"
    "seed":         42,

    # --- Checkpointing / Logging -----------------------------------------
    "output_dir":   "./checkpoints",   # base dir; dataset name appended automatically
    "resume":       None,              # path to .pth to resume from
    "save_freq":    10,                # save periodic checkpoint every N epochs
    "eval_freq":    1,                 # validate every N epochs
    "log_freq":     50,                # progress-bar update every N batches
}

# ============================================================================
# End of user config — no need to edit below this line
# ============================================================================

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VMambaClassifier, VMambaConfig
from models.vim_classifier import VimClassifier, VimConfig
from data import DatasetType, get_dataloader, get_dataset_info


# ============================================================================
# Config → Args Merging
# ============================================================================

def build_args():
    """Merge CONFIG dict with optional CLI overrides.

    Priority:  CLI flag  >  CONFIG dict  >  defaults.
    If no CLI flags are given the CONFIG dict is used as-is.
    """
    parser = argparse.ArgumentParser(
        description="Train VMamba Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Every CONFIG key becomes a CLI arg (type inferred from default value)
    for key, default in CONFIG.items():
        if isinstance(default, bool):
            parser.add_argument(
                f"--{key}",
                type=lambda v: v.lower() in ("true", "1", "yes"),
                default=default,
            )
        elif isinstance(default, list):
            elem_type = type(default[0]) if default else int
            parser.add_argument(
                f"--{key}", type=elem_type, nargs="+", default=default)
        elif default is None:
            parser.add_argument(f"--{key}", type=str, default=default)
        else:
            parser.add_argument(
                f"--{key}", type=type(default), default=default)

    args = parser.parse_args()
    return args


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model Factory
# ============================================================================

def create_model(args, num_classes: int, in_channels: int):
    """Factory: build the requested model architecture.

    Supported ``args.model_arch``:
      - "vmamba": original hierarchical 4-direction cross-scan VMamba.
      - "vim":    plain bidirectional 1D Vim (Zhu et al. 2024). Uses
                  ``args.vim_depth``, ``args.vim_d_model``,
                  ``args.vim_mlp_ratio``; respects ``args.patch_size``.
    """
    arch = getattr(args, "model_arch", "vmamba").lower()

    if arch == "vmamba":
        config = VMambaConfig(
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=in_channels,
            dims=args.dims,
            depths=args.depths,
            d_state=args.d_state,
            num_classes=num_classes,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
        )
        return VMambaClassifier(config)

    if arch == "vim":
        config = VimConfig(
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=in_channels,
            d_model=args.vim_d_model,
            depth=args.vim_depth,
            d_state=args.d_state,
            num_classes=num_classes,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            mlp_ratio=args.vim_mlp_ratio,
        )
        return VimClassifier(config)

    raise ValueError(f"unknown model_arch: {arch!r} (expected 'vmamba' or 'vim')")


# ============================================================================
# Training Epoch
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, args):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=args.grad_clip)

        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % args.log_freq == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.1f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ============================================================================
# Evaluation (returns logits for AUC computation)
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_logits.append(outputs.cpu())
        all_labels.append(labels.cpu())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    all_logits = torch.cat(all_logits, dim=0)   # [N, C]
    all_labels = torch.cat(all_labels, dim=0)    # [N]

    return avg_loss, accuracy, all_logits, all_labels


# ============================================================================
# Comprehensive Metrics: ACC, AUC, F1
# ============================================================================

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor,
                    num_classes: int):
    """Compute accuracy, macro-F1, macro-AUC from raw logits."""
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    true = labels.numpy()

    acc = accuracy_score(true, preds) * 100.0

    # F1 (macro)
    f1 = f1_score(true, preds, average="macro", zero_division=0) * 100.0

    # AUC (macro, one-vs-rest)
    try:
        if num_classes == 2:
            auc = roc_auc_score(true, probs[:, 1]) * 100.0
        else:
            auc = roc_auc_score(true, probs, multi_class="ovr",
                                average="macro") * 100.0
    except ValueError:
        # Can happen if a class has no samples in this split
        auc = float("nan")

    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "preds": preds,
        "probs": probs,
        "true": true,
    }


# ============================================================================
# Plotting: Training Curves (2×2)
# ============================================================================

def plot_training_curves(history: dict, output_dir: Path):
    """Save a publication-ready 2x2 figure of training dynamics."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    val_epochs = np.array(history["val_epochs"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    if len(val_epochs) > 0:
        ax.plot(val_epochs, history["val_loss"], label="Val",
                linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], label="Train", linewidth=1.5)
    if len(val_epochs) > 0:
        ax.plot(val_epochs, history["val_acc"], label="Val",
                linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Val AUC & F1
    ax = axes[1, 0]
    if len(val_epochs) > 0:
        ax.plot(val_epochs, history["val_auc"], label="AUC",
                linewidth=1.5, marker="s", markersize=3)
        ax.plot(val_epochs, history["val_f1"], label="F1",
                linewidth=1.5, marker="^", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_title("(c) Validation AUC & F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], linewidth=1.5, color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(d) Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, y=1.01)
    plt.tight_layout()

    out = output_dir / "training_curves.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# Plotting: Confusion Matrix
# ============================================================================

def plot_confusion_matrix(true, preds, class_names, output_dir: Path):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(true, preds)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Test Set)")

    # Numbers inside cells
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    out = output_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args,
                    output_dir, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args) if not isinstance(args, dict) else args,
    }
    path = output_dir / filename
    torch.save(checkpoint, path)
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ============================================================================
# Text Report
# ============================================================================

def write_report(output_dir: Path, args, history, test_metrics,
                 class_names, train_time_sec):
    path = output_dir / "final_report.txt"
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("X-VMAMBA TRAINING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Config
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        cfg = vars(args) if not isinstance(args, dict) else args
        for k, v in cfg.items():
            f.write(f"  {k:<20} {v}\n")

        # Training summary
        f.write(f"\n\nTRAINING SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total epochs:          {len(history['train_loss'])}\n")
        f.write(f"  Training time:         {train_time_sec/60:.1f} min\n")
        if history["val_acc"]:
            f.write(f"  Best val accuracy:     {max(history['val_acc']):.2f}%\n")
            best_idx = int(np.argmax(history["val_acc"]))
            f.write(f"  Best val epoch:        "
                    f"{history['val_epochs'][best_idx]}\n")
        f.write(f"  Final train loss:      {history['train_loss'][-1]:.4f}\n")
        if history["val_loss"]:
            f.write(f"  Final val loss:        {history['val_loss'][-1]:.4f}\n")

        # Test metrics
        f.write(f"\n\nTEST SET RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Accuracy:   {test_metrics['acc']:.2f}%\n")
        f.write(f"  AUC:        {test_metrics['auc']:.2f}%\n")
        f.write(f"  F1 (macro): {test_metrics['f1']:.2f}%\n")

        # Per-class report
        f.write(f"\n\nPER-CLASS CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        report = classification_report(
            test_metrics["true"], test_metrics["preds"],
            target_names=class_names, digits=4, zero_division=0,
        )
        f.write(report)

    print(f"  Saved {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    args = build_args()

    # ── Setup ────────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Output directory: checkpoints/<dataset>/
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config snapshot
    cfg = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_type = DatasetType(args.dataset)
    dataset_info = get_dataset_info(dataset_type)

    num_classes = dataset_info["num_classes"]
    in_channels = (args.in_channels if args.in_channels is not None
                   else dataset_info["in_channels"])
    class_names = dataset_info.get("classes",
                                    [f"Class_{i}" for i in range(num_classes)])

    # Store derived values in args so they are saved in every checkpoint
    # (run_final_evaluation.py reads them back to rebuild the exact model)
    args.in_channels = in_channels
    args.num_classes = num_classes

    print(f"\nDataset:    {dataset_info['name']}")
    print(f"  Classes:    {num_classes}  {class_names}")
    print(f"  Channels:   {in_channels}")
    print(f"  Task:       {dataset_info['task']}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloader(
        dataset_type=dataset_type,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        npz_path=args.npz_path,
    )
    print(f"  Train: {len(train_loader.dataset):,}")
    print(f"  Val:   {len(val_loader.dataset):,}")
    print(f"  Test:  {len(test_loader.dataset):,}")

    # ── Model ────────────────────────────────────────────────────────────
    print("\nCreating model...")
    model = create_model(args, num_classes, in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:  {n_params:,}")
    print(f"  Trainable:   {n_train:,}")

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup → cosine decay
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=max(args.warmup_epochs, 1),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"\nResuming from {args.resume} ...")
        start_epoch, prev_metrics = load_checkpoint(
            model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
        best_val_acc = prev_metrics.get("best_val_acc", 0.0)
        print(f"  Resumed at epoch {start_epoch}, "
              f"best val acc so far: {best_val_acc:.2f}%")

    # ── History tracking ─────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": [],
        "val_epochs": [],
        "lr": [],
    }

    # ── Training loop ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["lr"].append(current_lr)

        msg = (f"Epoch {epoch+1:>3}/{args.epochs}  |  "
               f"Train Loss {train_loss:.4f}  Acc {train_acc:.1f}%  |  "
               f"LR {current_lr:.2e}")

        # --- Validate ---
        if (epoch + 1) % args.eval_freq == 0:
            val_loss, val_acc, val_logits, val_labels = evaluate(
                model, val_loader, criterion, device, "Val")

            val_m = compute_metrics(val_logits, val_labels, num_classes)

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_m["acc"])
            history["val_auc"].append(val_m["auc"])
            history["val_f1"].append(val_m["f1"])
            history["val_epochs"].append(epoch + 1)

            msg += (f"  |  Val Loss {val_loss:.4f}  "
                    f"Acc {val_m['acc']:.1f}%  "
                    f"AUC {val_m['auc']:.1f}%  "
                    f"F1 {val_m['f1']:.1f}%")

            # Save best
            if val_m["acc"] > best_val_acc:
                best_val_acc = val_m["acc"]
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"best_val_acc": best_val_acc,
                     "val_auc": val_m["auc"], "val_f1": val_m["f1"]},
                    args, output_dir, "best_model.pth",
                )
                msg += "  ** best **"

        print(msg)

        # --- Periodic save ---
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {"best_val_acc": best_val_acc},
                args, output_dir, f"checkpoint_epoch_{epoch+1}.pth",
            )

    train_time = time.time() - t_start
    print(f"\nTraining completed in {train_time/60:.1f} min")

    # Always save last model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1,
        {"best_val_acc": best_val_acc},
        args, output_dir, "last_model.pth",
    )

    # ── Final Test Evaluation ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    # Load best model
    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model (epoch {ckpt['epoch']+1}, "
              f"val acc {ckpt['metrics']['best_val_acc']:.2f}%)")

    test_loss, test_acc, test_logits, test_labels = evaluate(
        model, test_loader, criterion, device, "Test")

    test_m = compute_metrics(test_logits, test_labels, num_classes)

    print(f"\n{'='*50}")
    print(f"  TEST ACCURACY :  {test_m['acc']:.2f}%")
    print(f"  TEST AUC      :  {test_m['auc']:.2f}%")
    print(f"  TEST F1       :  {test_m['f1']:.2f}%")
    print(f"{'='*50}")

    # Per-class report
    print(f"\n{classification_report(test_m['true'], test_m['preds'], target_names=class_names, digits=4, zero_division=0)}")

    # ── Plots ────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(test_m["true"], test_m["preds"],
                          class_names, output_dir)

    # ── Save everything ──────────────────────────────────────────────────
    results = {
        "test_acc": test_m["acc"],
        "test_auc": test_m["auc"],
        "test_f1": test_m["f1"],
        "test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "args": vars(args),
    }
    torch.save(results, output_dir / "training_results.pth")
    print(f"  Saved {output_dir / 'training_results.pth'}")

    write_report(output_dir, args, history, test_m,
                 class_names, train_time)

    # ── Done ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ALL DONE")
    print(f"{'='*70}")
    print(f"\n  Output folder: {output_dir}/")
    for fp in sorted(output_dir.glob("*")):
        if fp.is_file():
            size_mb = fp.stat().st_size / 1024 / 1024
            print(f"    {fp.name:<30} {size_mb:.1f} MB")
    print()


if __name__ == "__main__":
    main()