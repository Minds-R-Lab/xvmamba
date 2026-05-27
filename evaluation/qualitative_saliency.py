#!/usr/bin/env python3
"""
Qualitative saliency maps for the new datasets.
==============================================

Picks N representative test images per dataset and saves a 1xK grid PNG
of  [original | Jacobian | Gramian | Grad-CAM]  for visual inspection.
Uses the existing seed-42 checkpoints and the proven analyzer pipeline.

Output structure:
    <output_dir>/<dataset>/example_<idx>.png   (one PNG per image)
    <output_dir>/<dataset>/grid_<dataset>.png  (combined grid for the paper)

Usage:
    cd xvmamba
    python evaluation/qualitative_saliency.py \\
        --datasets cifar100 fashionmnist eurosat \\
        --num_images 6 --seed 42
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

from controllability.analyzer import (
    ControllabilityAnalyzer,
    ControllabilityMethod,
)
from data import DatasetType, get_dataloader, get_dataset_info
from evaluation.comprehensive_evaluation import (
    GradCAMSaliency,
    load_model,
)

EPS = 1e-12


def aggregate_influence(analyzer, analysis, target_size):
    """Mirror of the aggregation used everywhere else for J/G saliency maps."""
    layer_maps = []
    for stage_idx, stage_caches in enumerate(analysis.stage_caches):
        for block_idx, cache in enumerate(stage_caches):
            result = analyzer.analyze_block(cache, stage_idx=stage_idx, block_idx=block_idx)
            imap = result.influence_map
            if imap is None:
                continue
            if imap.shape[0] != target_size[0] or imap.shape[1] != target_size[1]:
                imap = F.interpolate(
                    imap.unsqueeze(0).unsqueeze(0),
                    size=target_size, mode="bilinear", align_corners=False,
                ).squeeze()
            layer_maps.append(imap)
    if not layer_maps:
        return None
    return torch.stack(layer_maps, dim=0).mean(dim=0)


def minmax(m):
    """Min-max normalise to [0, 1]."""
    if isinstance(m, torch.Tensor):
        lo, hi = m.min(), m.max()
        if (hi - lo).item() < EPS:
            return torch.zeros_like(m)
        return (m - lo) / (hi - lo + EPS)
    m = np.asarray(m, dtype=np.float64)
    lo, hi = m.min(), m.max()
    if hi - lo < EPS: return np.zeros_like(m)
    return (m - lo) / (hi - lo + EPS)


def saliency_for_image(model, analyzer_J, analyzer_G, gradcam, image, device):
    image = image.to(device)
    H, W = image.shape[-2], image.shape[-1]
    if hasattr(model, "enable_analysis_mode"):
        model.enable_analysis_mode(store_states=False)
    with torch.no_grad():
        logits, analysis = model(image, return_analysis=True)
    probs = F.softmax(logits[0], dim=-1)
    c_star = int(probs.argmax().item())
    p_star = float(probs[c_star].item())

    J_t = aggregate_influence(analyzer_J, analysis, (H, W))
    G_t = aggregate_influence(analyzer_G, analysis, (H, W))
    if hasattr(model, "disable_analysis_mode"):
        model.disable_analysis_mode()
    if J_t is None or G_t is None:
        return None

    J = minmax(J_t).detach().cpu().numpy()
    G = minmax(G_t).detach().cpu().numpy()
    gc = gradcam.generate(image, c_star)
    if gc.shape != (H, W):
        gc = F.interpolate(gc.unsqueeze(0).unsqueeze(0).float(),
                           size=(H, W), mode="bilinear", align_corners=False).squeeze()
    GC = minmax(gc).detach().cpu().numpy()

    return {
        "image": image[0].detach().cpu().numpy(),
        "J": J, "G": G, "GC": GC,
        "pred_class": c_star, "pred_prob": p_star,
    }


def _prep_image_for_display(image):
    """Convert numpy [C, H, W] or [H, W, C] image to [H, W, 3] in [0,1]."""
    img = image.copy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = (img - img.min()) / (img.max() - img.min() + EPS)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def _overlay(image_rgb, heatmap, alpha=0.55, cmap_name="jet"):
    """Overlay heatmap on image with alpha blending. Returns RGB array."""
    cmap = plt.get_cmap(cmap_name)
    h_norm = np.clip(heatmap, 0, 1)
    heatmap_rgb = cmap(h_norm)[..., :3]   # (H, W, 3)
    blended = (1 - alpha) * image_rgb + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


def save_example_grid(image, J, G, GC, label, pred, prob, class_names, out_path):
    """Save a 1x5 figure: Input | Jacobian (Ours) | Gramian (Ours) | Grad-CAM | Random.
    Style matches the manuscript figures 9 and 10: jet colormap, alpha overlay on the
    original image."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    img_rgb = _prep_image_for_display(image)

    # Random saliency for the 5th column (uniform random noise, min-max normalised)
    R = np.random.rand(*J.shape)
    R = (R - R.min()) / (R.max() - R.min() + EPS)

    panels = [
        ("Input", None),
        ("Jacobian (Ours)", J),
        ("Gramian (Ours)", G),
        ("Grad-CAM", GC),
        ("Random", R),
    ]
    for ax, (title, m) in zip(axes, panels):
        if m is None:
            ax.imshow(img_rgb)
        else:
            ax.imshow(_overlay(img_rgb, m, alpha=0.55, cmap_name="jet"))
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    # Optional subtitle on the input panel
    true_lbl = class_names[label] if class_names and label < len(class_names) else f"class {label}"
    pred_lbl = class_names[pred] if class_names and pred < len(class_names) else f"class {pred}"
    ok = "OK" if pred == label else "X"
    axes[0].set_xlabel(f"true: {true_lbl}\npred: {pred_lbl}  ({prob:.2f})  [{ok}]",
                       fontsize=8)
    axes[0].xaxis.set_label_position("bottom")
    axes[0].xaxis.set_visible(True); axes[0].tick_params(left=False, bottom=False,
                                                          labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def make_combined_grid(records, dataset_name, out_path):
    """Combined Nx5 grid of multiple examples in the paper's style."""
    N = len(records)
    if N == 0: return
    fig, axes = plt.subplots(N, 5, figsize=(16, 3 * N))
    if N == 1:
        axes = axes[None, :]
    for i, r in enumerate(records):
        img_rgb = _prep_image_for_display(r["image"])
        # Random saliency for the 5th column
        R = np.random.rand(*r["J"].shape)
        R = (R - R.min()) / (R.max() - R.min() + EPS)
        panels = [
            ("Input", None),
            ("Jacobian (Ours)", r["J"]),
            ("Gramian (Ours)", r["G"]),
            ("Grad-CAM", r["GC"]),
            ("Random", R),
        ]
        for j, (title, m) in enumerate(panels):
            if m is None:
                axes[i, j].imshow(img_rgb)
            else:
                axes[i, j].imshow(_overlay(img_rgb, m, alpha=0.55, cmap_name="jet"))
            if i == 0:
                axes[i, j].set_title(title, fontsize=13)
            axes[i, j].axis("off")
        # Left-side row label with dataset name on the first row only
        if i == 0:
            axes[i, 0].set_ylabel(dataset_name, fontsize=11, rotation=90)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar100", "fashionmnist", "eurosat", "bloodmnist_vim"])
    ap.add_argument("--num_images", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_base", default="./checkpoints/multiseed")
    ap.add_argument("--output_dir", default="./figures/qualitative_new")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for ds_arg in args.datasets:
        # bloodmnist_vim ckpt subdir, but underlying dataset is bloodmnist
        if ds_arg == "bloodmnist_vim":
            ds_for_loader = "bloodmnist"
            ckpt_subdir = "bloodmnist_vim"
            display_name = "BloodMNIST (Vim)"
        else:
            ds_for_loader = ds_arg
            ckpt_subdir = ds_arg
            display_name = {
                "cifar100":"CIFAR-100","fashionmnist":"FashionMNIST",
                "eurosat":"EuroSAT","bloodmnist":"BloodMNIST",
                "dermamnist":"DermaMNIST","octmnist":"OCTMNIST",
                "pneumoniamnist":"PneumoniaMNIST",
            }.get(ds_arg, ds_arg)

        ckpt = f"{args.ckpt_base}/seed_{args.seed}/{ckpt_subdir}/best_model.pth"
        if not Path(ckpt).exists():
            print(f"[skip] {ds_arg}: ckpt missing at {ckpt}"); continue
        print(f"\n=== {display_name} ===")

        model, config = load_model(ckpt, device)
        model.eval()
        info = get_dataset_info(DatasetType(ds_for_loader))
        class_names = info.get("classes")

        analyzer_J = ControllabilityAnalyzer(method=ControllabilityMethod.JACOBIAN, normalize=True)
        analyzer_G = ControllabilityAnalyzer(method=ControllabilityMethod.GRAMIAN,  normalize=True)
        gradcam = GradCAMSaliency(model, device)

        _, _, test_loader = get_dataloader(
            dataset_type=DatasetType(ds_for_loader),
            batch_size=1, num_workers=0,
            image_size=getattr(config, "image_size", 224),
            data_root="./data",
        )

        out_ds = out_root / ds_arg
        out_ds.mkdir(parents=True, exist_ok=True)
        records = []
        for i, batch in enumerate(test_loader):
            if len(records) >= args.num_images: break
            try:
                image, label = batch
                rec = saliency_for_image(model, analyzer_J, analyzer_G, gradcam, image, device)
                if rec is None: continue
                rec["true_label"] = int(label.item())
                save_example_grid(
                    rec["image"], rec["J"], rec["G"], rec["GC"],
                    rec["true_label"], rec["pred_class"], rec["pred_prob"],
                    class_names, out_ds / f"example_{i:02d}_idx{i}.png",
                )
                records.append(rec)
                print(f"  saved example {len(records)}/{args.num_images}  (img {i}, pred={rec['pred_class']} conf={rec['pred_prob']:.2f})")
            except Exception as e:
                import traceback
                print(f"  [warn] img {i} failed: {type(e).__name__}: {e}")
                if i == 0: traceback.print_exc()
                continue

        if records:
            grid_path = out_ds / f"grid_{ds_arg}.png"
            make_combined_grid(records, display_name, grid_path)
            print(f"  ✓ saved combined grid: {grid_path}")


if __name__ == "__main__":
    main()
