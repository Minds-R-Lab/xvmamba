#!/usr/bin/env python3
"""
Internal-Attention IoU — coverage-oriented faithfulness metric.
================================================================

Operationalises the observation that controllability maps cover the
*spatial region the model actually uses*, whereas Grad-CAM concentrates
on the gradient hotspot within that region. We use the model's own
last-block feature map as the pseudo-ground-truth for "where the model
is attending", then compute IoU between each saliency method's top-K%
binary mask and the model's top-25% attention region.

We compute the model attention map in FOUR channel-aggregation variants
so we can pick the most discriminative one before adding to the paper:

  (a) mean    : A_mean(h, w) = mean_c  F(h, w, c)
  (b) L2      : A_L2(h, w)   = sqrt( sum_c F(h, w, c)^2 )
  (c) max     : A_max(h, w)  = max_c   |F(h, w, c)|
  (d) cw      : A_cw(h, w)   = max(0, sum_c W[pred, c] * F(h, w, c))
                where W is the classifier head and `pred` is the
                model's predicted class for this image.

For each image we report:
  IoU(method_top_K, attention_variant_top_25%)
across K in {10%, 25%, 50%} and across the four variants.

Usage:
    cd xvmamba
    python evaluation/attention_iou.py \
        --checkpoint checkpoints/multiseed/seed_42/bloodmnist/best_model.pth \
        --dataset bloodmnist \
        --num_samples 50 \
        --output_dir results/attn_iou/seed_42/bloodmnist
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))

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
K_SALIENCY = (0.10, 0.25, 0.50)        # top-K% for each saliency method
ATTENTION_TOP = 0.25                    # always use top-25% of attention as ROI


# =============================================================================
# Saliency aggregation (uniform mean across blocks, mirrors Phase 6/8)
# =============================================================================

def aggregate_influence(analyzer, analysis, target_size):
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
# Model's internal attention via last-block feature map
# =============================================================================

class FeatureCapture:
    """Hooks the model's last block (before classifier) and captures its
    forward-pass activation. VMamba returns (B, H, W, C); Vim returns (B, L, C)."""

    def __init__(self, model):
        self.model = model
        self.activation = None
        if hasattr(model, "stages") and len(model.stages) > 0:
            self._target = model.stages[-1].blocks[-1]
            self.arch = "vmamba"
        elif hasattr(model, "blocks") and len(model.blocks) > 0:
            self._target = model.blocks[-1]
            self.arch = "vim"
        else:
            raise AttributeError("no last-block feature map locatable")
        self._handle = self._target.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activation = out.detach()

    def reset(self):
        self.activation = None

    def remove(self):
        self._handle.remove()


def four_attention_variants(
    feature_map: torch.Tensor,
    classifier_weights: torch.Tensor,
    predicted_class: int,
    arch: str,
) -> Dict[str, torch.Tensor]:
    """Compute the four channel-aggregation variants from the last-block
    feature map. Returns a dict variant_name -> 2D map (H, W)."""
    f = feature_map
    if arch == "vim":
        # f is (B, L, C); reshape to (B, H, W, C) with H = W = sqrt(L)
        B, L, C = f.shape
        side = int(round(L ** 0.5))
        if side * side != L:
            # try popping a class token (Vim sometimes prepends one)
            if (L - 1) > 0 and int(round((L - 1) ** 0.5)) ** 2 == (L - 1):
                f = f[:, 1:, :]
                side = int(round((L - 1) ** 0.5))
            else:
                raise ValueError(f"Vim feature length {L} is not a perfect square")
        f = f.view(B, side, side, C)
    # f is now (B, H, W, C)
    f = f[0]                                              # drop batch -> (H, W, C)
    out = {}
    out["mean"] = f.mean(dim=-1)                          # (H, W)
    out["l2"]   = torch.sqrt((f * f).sum(dim=-1))         # (H, W)
    out["max"]  = f.abs().max(dim=-1).values              # (H, W)
    # classifier-weighted: max(0, sum_c W[pred, c] * F(h, w, c))
    w_c = classifier_weights[predicted_class]             # (C,)
    cw = (f * w_c.view(1, 1, -1)).sum(dim=-1)             # (H, W) — signed
    out["cw"] = F.relu(cw)                                # only positive contributions
    # Min-max normalise each so they live on [0, 1]
    return {k: minmax(v) for k, v in out.items()}


def upsample(m: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinear-upsample a 2D tensor to (H, W)."""
    return F.interpolate(
        m.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze()


# =============================================================================
# IoU computation
# =============================================================================

def topk_binary_mask(map_2d: torch.Tensor, k_frac: float) -> torch.Tensor:
    flat = map_2d.flatten()
    n = flat.numel()
    n_keep = max(1, int(round(k_frac * n)))
    sorted_idx = torch.argsort(flat, descending=True)
    mask = torch.zeros(n, dtype=torch.bool, device=map_2d.device)
    mask[sorted_idx[:n_keep]] = True
    return mask.view_as(map_2d)


def iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return inter / max(union, 1)


# =============================================================================
# Per-image computation
# =============================================================================

def analyze_image(model, fc, analyzer_J, analyzer_G, gradcam,
                  image_cpu, label, device, classifier_weights):
    image = image_cpu.to(device)
    H, W = image.shape[-2], image.shape[-1]

    # --- Forward 1: analysis mode (J/G), captures last-block activation via hook
    if hasattr(model, "enable_analysis_mode"):
        model.enable_analysis_mode(store_states=False)
    fc.reset()
    with torch.no_grad():
        logits, analysis = model(image, return_analysis=True)

    probs = torch.softmax(logits[0], dim=-1)
    c_star = int(probs.argmax().item())
    p_star = float(probs[c_star].item())

    # J and G aggregated maps at input resolution. Important: do this BEFORE
    # disable_analysis_mode() because that call clears the SS2D caches.
    J_t = aggregate_influence(analyzer_J, analysis, (H, W))
    G_t = aggregate_influence(analyzer_G, analysis, (H, W))
    if hasattr(model, "disable_analysis_mode"):
        model.disable_analysis_mode()
    if J_t is None or G_t is None:
        raise RuntimeError("analyzer returned no maps")

    # Four attention variants from feature_map
    attn_variants = four_attention_variants(
        fc.activation, classifier_weights, c_star, fc.arch,
    )
    # Upsample to input resolution
    attn_up = {k: upsample(v, H, W) for k, v in attn_variants.items()}
    attn_masks = {k: topk_binary_mask(v, ATTENTION_TOP) for k, v in attn_up.items()}
    J = minmax(J_t).to(device)
    G = minmax(G_t).to(device)

    # Grad-CAM map at input resolution
    gc_raw = gradcam.generate(image, c_star)
    if gc_raw.shape != (H, W):
        gc_raw = upsample(gc_raw, H, W)
    GC = minmax(gc_raw).to(device)

    # Random
    R = minmax(torch.rand(H, W, device=device))

    method_maps = {
        "Jacobian": J,
        "Gramian":  G,
        "Grad-CAM": GC,
        "Random":   R,
    }

    # For each (saliency_method, K, attention_variant): compute IoU
    out = {"target_class": c_star, "target_prob": p_star,
           "results": {}}
    for K in K_SALIENCY:
        method_masks = {m: topk_binary_mask(method_maps[m], K) for m in method_maps}
        for var_name, attn_mask in attn_masks.items():
            for m, mm in method_masks.items():
                key = f"K{int(100*K)}_{var_name}_{m}"
                out["results"][key] = iou(mm, attn_mask)
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Internal-Attention IoU faithfulness")
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

    # Classifier weights for the 'cw' variant
    if not hasattr(model, "head") or not hasattr(model.head, "weight"):
        raise RuntimeError("model has no .head.weight — cannot compute cw variant")
    classifier_weights = model.head.weight.detach()   # (num_classes, dim)
    print(f"[load] classifier head weight shape: {tuple(classifier_weights.shape)}")

    fc = FeatureCapture(model)
    print(f"[load] last-block arch hooked: {fc.arch}")

    analyzer_J = ControllabilityAnalyzer(method=ControllabilityMethod.JACOBIAN, normalize=True)
    analyzer_G = ControllabilityAnalyzer(method=ControllabilityMethod.GRAMIAN,  normalize=True)
    gradcam = GradCAMSaliency(model, device)

    ds_type = DatasetType(args.dataset)
    _, _, test_loader = get_dataloader(
        dataset_type=ds_type, batch_size=1, num_workers=0,
        image_size=image_size, data_root=args.data_root,
    )

    per_image = []
    for i, batch in enumerate(test_loader):
        if i >= args.num_samples: break
        try:
            image, label = batch
            rec = analyze_image(model, fc, analyzer_J, analyzer_G, gradcam,
                                image, int(label.item()), device, classifier_weights)
            per_image.append(rec)
        except Exception as e:
            import traceback
            print(f"[warn] image {i} failed: {type(e).__name__}: {e}")
            if i == 0: traceback.print_exc()
            continue
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{args.num_samples}")

    fc.remove()

    # Aggregate: mean IoU across images for each (K, variant, method)
    summary = {
        "dataset":     args.dataset,
        "num_samples": len(per_image),
        "K_saliency":  list(K_SALIENCY),
        "attention_top": ATTENTION_TOP,
        "methods":     ["Jacobian", "Gramian", "Grad-CAM", "Random"],
        "variants":    ["mean", "l2", "max", "cw"],
        "iou_mean": {},
        "iou_std":  {},
    }
    keys = list(per_image[0]["results"].keys()) if per_image else []
    for key in keys:
        vals = [rec["results"][key] for rec in per_image]
        summary["iou_mean"][key] = float(np.mean(vals))
        summary["iou_std"][key]  = float(np.std(vals))

    out_path = out_dir / "attention_iou.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))

    # Pretty headline table per K, with all 4 variants × 4 methods
    print()
    print("=" * 78)
    print(f"  Internal-Attention IoU — {args.dataset}  (N={summary['num_samples']}, attention top {int(ATTENTION_TOP*100)}%)")
    print("=" * 78)
    for K in K_SALIENCY:
        print()
        print(f"  saliency at top-{int(100*K)}%")
        print(f"  {'variant':<10s}  {'Jacobian':<10s}  {'Gramian':<10s}  {'Grad-CAM':<10s}  {'Random':<10s}  J-vs-GC")
        print("  " + "-" * 60)
        for var in ("mean", "l2", "max", "cw"):
            row = {}
            for m in ("Jacobian", "Gramian", "Grad-CAM", "Random"):
                row[m] = summary["iou_mean"].get(f"K{int(100*K)}_{var}_{m}", float("nan"))
            J_GC = row['Jacobian'] - row['Grad-CAM']
            arrow = " ←J" if J_GC > 0 else (" ←GC" if J_GC < 0 else "")
            print(f"  {var:<10s}  {row['Jacobian']:<10.3f}  {row['Gramian']:<10.3f}  "
                  f"{row['Grad-CAM']:<10.3f}  {row['Random']:<10.3f}  {J_GC:+.3f}{arrow}")
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
