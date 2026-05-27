#!/usr/bin/env python3
"""
Top-K Mask Preservation — class-agnostic-friendly faithfulness metric.
=====================================================================

For each test image and each saliency method we ask: if we keep only the
top-K% of pixels (by absolute saliency) and replace the rest with the
per-channel baseline, what fraction of the originally-predicted-class
probability does the model retain?

  preservation(K, M, x) = p(c* | mask_K(M, x) * x + (1 - mask_K) * baseline(x))
                                                     /
                          p(c* | x)

  where:
    c*       = model.argmax(x)
    M(x)     = saliency map of method M on image x
    mask_K   = binary mask keeping top K% of pixels by saliency
    baseline = per-channel mean (audit-fix protocol)

Why this metric is more honest than insertion/deletion for class-agnostic
methods:
  - Insertion/deletion AUC rewards saliency that exactly matches the
    model's gradient. A class-AGNOSTIC structural map distributes mass
    over anatomically/structurally relevant regions instead of the
    gradient-aligned high-magnitude pixels, so it pays a metric tax it
    shouldn't.
  - Top-K preservation asks the right question: did the method
    identify a SUFFICIENT subset of pixels? Class-agnostic methods that
    cover the meaningful structure can pass this test even when their
    saliency doesn't track the gradient direction.

Usage:
    cd xvmamba
    python evaluation/topk_preservation.py \
        --checkpoint checkpoints/multiseed/seed_42/bloodmnist/best_model.pth \
        --dataset bloodmnist \
        --num_samples 50 \
        --output_dir results/topk/seed_42/bloodmnist

Output:
    <output_dir>/topk_preservation.json with structure:
      {
        "dataset": ...,
        "num_samples": 50,
        "K_values": [0.05, 0.10, 0.20, 0.30, 0.50],
        "methods": {
            "Jacobian": {
                "preservation_per_K": [mean over images for each K],
                "preservation_AUC":    trapezoidal area under preservation curve,
                "per_sample_per_K":   [[p(c*|mask_K(image_i))] for each K],
            },
            ...
        }
      }
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))   # xvmamba/

from controllability.analyzer import (
    ControllabilityAnalyzer,
    ControllabilityMethod,
)
from data import DatasetType, get_dataloader
from evaluation.comprehensive_evaluation import (
    GradCAMSaliency,
    load_model,
)

EPS = 1e-12
K_VALUES = (0.05, 0.10, 0.20, 0.30, 0.50)


# =============================================================================
# Saliency aggregation (mirrors selectivity_disagreement.py / Table V protocol)
# =============================================================================

def aggregate_influence(analyzer, analysis, target_size):
    """Uniform-mean aggregation of per-block influence maps to target_size."""
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


def minmax(m: torch.Tensor) -> torch.Tensor:
    lo, hi = m.min(), m.max()
    if (hi - lo).item() < EPS:
        return torch.zeros_like(m)
    return (m - lo) / (hi - lo + EPS)


# =============================================================================
# Top-K preservation core
# =============================================================================

def topk_preservation_one_image(
    model,
    image: torch.Tensor,       # [1, C, H, W]
    saliency_map: torch.Tensor,# [H, W], min-max [0,1]
    baseline: torch.Tensor,    # [1, C, 1, 1]
    target_class: int,
    target_prob: float,
    K_values=K_VALUES,
) -> List[float]:
    """For each K in K_values, build the masked image (keep top-K%, replace
    rest with baseline), forward, return p(c*|masked) / p(c*|original)."""
    H, W = saliency_map.shape
    n_pixels = H * W
    flat = saliency_map.flatten()

    # We need pixel rank, sort descending once.
    sorted_idx = torch.argsort(flat, descending=True)

    out = []
    for K in K_values:
        n_keep = max(1, int(round(K * n_pixels)))
        # Build the binary spatial mask (1 = keep original pixel, 0 = use baseline)
        mask = torch.zeros(n_pixels, dtype=image.dtype, device=image.device)
        mask[sorted_idx[:n_keep]] = 1.0
        mask = mask.view(1, 1, H, W)                                  # [1,1,H,W]
        # Broadcast baseline to (1,C,H,W) and combine.
        baseline_full = baseline.expand_as(image)                     # [1,C,H,W]
        masked = mask * image + (1.0 - mask) * baseline_full
        with torch.no_grad():
            logits = model(masked)
            probs = F.softmax(logits[0], dim=-1)
            p = float(probs[target_class].item())
        if target_prob > EPS:
            out.append(p / target_prob)
        else:
            out.append(0.0)
    return out


def per_channel_baseline(image: torch.Tensor) -> torch.Tensor:
    """[1, C, H, W] -> [1, C, 1, 1] per-channel mean (audit-fix protocol)."""
    return image.mean(dim=[2, 3], keepdim=True)


# =============================================================================
# Per-image runner (computes J / G / GC / Random + Top-K preservation)
# =============================================================================

def analyze_image(model, analyzer_J, analyzer_G, gradcam, image_cpu, label, device):
    """Return a dict: method -> [preservation per K]."""
    image = image_cpu.to(device)
    H, W = image.shape[-2], image.shape[-1]
    baseline = per_channel_baseline(image)

    # Original prediction (used to define c* and to normalise preservation)
    if hasattr(model, "enable_analysis_mode"):
        model.enable_analysis_mode(store_states=False)
    with torch.no_grad():
        logits, analysis = model(image, return_analysis=True)
    probs0 = F.softmax(logits[0], dim=-1)
    c_star = int(probs0.argmax().item())
    p_star = float(probs0[c_star].item())

    # J and G aggregated maps
    J_t = aggregate_influence(analyzer_J, analysis, (H, W))
    G_t = aggregate_influence(analyzer_G, analysis, (H, W))
    if hasattr(model, "disable_analysis_mode"):
        model.disable_analysis_mode()
    if J_t is None or G_t is None:
        raise RuntimeError("analyzer returned no maps")
    J = minmax(J_t).to(device)
    G = minmax(G_t).to(device)

    # Grad-CAM map
    gc_raw = gradcam.generate(image, c_star)
    if gc_raw.shape != (H, W):
        gc_raw = F.interpolate(
            gc_raw.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze()
    GC = minmax(gc_raw).to(device)

    # Random map (controlled by Python seed inside this image only)
    rand_map = torch.rand(H, W, device=device)
    R = minmax(rand_map)

    method_maps = {
        "Jacobian": J,
        "Gramian":  G,
        "Grad-CAM": GC,
        "Random":   R,
    }
    out = {"target_class": c_star, "target_prob": p_star}
    for name, m in method_maps.items():
        out[name] = topk_preservation_one_image(
            model, image, m, baseline, c_star, p_star,
        )
    return out


# =============================================================================
# Aggregation
# =============================================================================

def auc_trapezoidal(K_values, values):
    """AUC by trapezoidal rule, normalised to [0,1] over the K range."""
    if len(K_values) != len(values):
        return 0.0
    auc = 0.0
    for i in range(1, len(K_values)):
        dx = K_values[i] - K_values[i - 1]
        auc += 0.5 * dx * (values[i] + values[i - 1])
    # Normalise by the range so 0.05->0.50 → 0.45 wide
    width = K_values[-1] - K_values[0]
    return auc / max(width, EPS)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Top-K mask preservation faithfulness")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data_root", default="./data")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    model.eval()
    image_size = getattr(config, "image_size", 224)
    print(f"[load] dataset: {args.dataset}  image_size={image_size}")

    analyzer_J = ControllabilityAnalyzer(method=ControllabilityMethod.JACOBIAN, normalize=True)
    analyzer_G = ControllabilityAnalyzer(method=ControllabilityMethod.GRAMIAN,  normalize=True)
    gradcam = GradCAMSaliency(model, device)

    ds_type = DatasetType(args.dataset)
    _, _, test_loader = get_dataloader(
        dataset_type=ds_type, batch_size=1, num_workers=0,
        image_size=image_size, data_root=args.data_root,
    )

    per_sample = {m: [] for m in ("Jacobian", "Gramian", "Grad-CAM", "Random")}
    target_classes = []
    target_probs   = []

    for i, batch in enumerate(test_loader):
        if i >= args.num_samples: break
        try:
            image, label = batch
            rec = analyze_image(model, analyzer_J, analyzer_G, gradcam,
                                image, int(label.item()), device)
        except Exception as e:
            import traceback
            print(f"[warn] image {i} failed: {type(e).__name__}: {e}")
            if i == 0: traceback.print_exc()
            continue
        target_classes.append(rec["target_class"])
        target_probs.append(rec["target_prob"])
        for m in per_sample:
            per_sample[m].append(rec[m])
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{args.num_samples}")

    # Aggregate per method: mean over images per K, plus AUC
    summary = {
        "dataset":      args.dataset,
        "checkpoint":   str(args.checkpoint),
        "num_samples":  len(target_probs),
        "K_values":     list(K_VALUES),
        "mean_target_prob": float(np.mean(target_probs)) if target_probs else None,
        "methods":      {},
    }
    for m, rows in per_sample.items():
        if not rows: continue
        arr = np.array(rows)                       # [N_images, len(K_VALUES)]
        mean_per_K = arr.mean(axis=0).tolist()
        std_per_K  = arr.std(axis=0).tolist()
        auc = auc_trapezoidal(K_VALUES, mean_per_K)
        summary["methods"][m] = {
            "preservation_per_K":      mean_per_K,
            "preservation_per_K_std":  std_per_K,
            "preservation_AUC":        auc,
            "per_sample_per_K":        rows,
        }

    (out_dir / "topk_preservation.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )

    # Pretty headline
    print()
    print("=" * 72)
    print(f"  Top-K mask preservation — {args.dataset}  (N={summary['num_samples']})")
    print(f"  baseline: per-channel mean.  mean p(c*|x) = {summary['mean_target_prob']:.3f}")
    print("=" * 72)
    K_str = "  ".join(f"K={int(100*k):2d}%" for k in K_VALUES)
    print(f"  {'method':10s}  {K_str:35s}  AUC")
    print("  " + "-" * 65)
    for m, d in summary["methods"].items():
        cells = "  ".join(f"{p:.3f}" for p in d["preservation_per_K"])
        print(f"  {m:10s}  {cells:35s}  {d['preservation_AUC']:.3f}")


if __name__ == "__main__":
    main()
