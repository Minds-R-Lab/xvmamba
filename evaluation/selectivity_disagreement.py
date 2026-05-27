#!/usr/bin/env python3
"""
Phase 6 — Selectivity-Disagreement Pipeline
===========================================

Tests two hypotheses that go beyond the current paper's framing:

  H1 (Δ_k as selectivity meter):
      The pointwise gap Δ_k = |J_k − G_k| between the Jacobian index
      (which tracks the full time-varying recurrence) and the Gramian
      proxy (which freezes the LTI parameters at position k) measures
      how much the model is actually exploiting Mamba's *selective*
      gating at position k. Where Δ_k is large, the LTV dynamics deviate
      from the frozen-LTI proxy; where it is small, the model is
      effectively behaving as a non-selective LTI SSM.

  H2 (D(x) as confidence signal):
      The per-image spatial disagreement
          D(x) = 1 − corr( J(x) , GradCAM(x, ŷ) )
      between the structural (class-agnostic) and attributive
      (class-specific) saliency is a class-free signal of how
      structurally consistent the model's prediction is. High D(x)
      should track misclassification and low model confidence.

What the pipeline does (one model, one dataset, no retraining):
  1. Loads the trained checkpoint exactly as comprehensive_evaluation.py does.
  2. For each of the first N test images:
       (a) Forward through model with analysis hooks → cached SSM state.
       (b) Run the Jacobian and Gramian analyzers → per-stage and
           aggregated influence maps  J_map, G_map  (both in [0,1]).
       (c) Run Grad-CAM for the predicted class → GC_map (in [0,1]).
       (d) Compute the per-image Δ map and scalar disagreement stats.
       (e) Record model confidence, prediction correctness, and the
           per-stage Δ magnitudes for H1's stage-level question.
  3. Aggregates everything and writes JSON + plots:
       - Per-stage Δ density (H1)
       - AUROC of D(x) for predicting misclassification (H2)
       - Correlation of D(x) with 1 − max-softmax (H2)
       - Scatter / heatmap-grid PNGs for the report
  4. (Optional) Saves a small set of full-resolution maps for visual
     inspection in the writeup.

Usage:
    cd xvmamba
    python evaluation/selectivity_disagreement.py \\
        --checkpoint checkpoints/bloodmnist/best_model.pth \\
        --dataset bloodmnist \\
        --num_samples 200 \\
        --output_dir results/exploratory/bloodmnist

  Outputs:
    results/exploratory/<dataset>/
        summary.json              # all aggregate stats
        per_image.csv             # one row per image
        per_stage_delta.png       # H1: Δ distribution per stage
        disagreement_auroc.png    # H2: ROC of D(x) for misclassification
        confidence_vs_D.png       # H2: confidence ↔ D(x) scatter
        examples/                 # 8–16 full-resolution heatmap grids

Notes:
  - Imports follow comprehensive_evaluation.py:
        from controllability.analyzer import (
            ControllabilityAnalyzer, ControllabilityMethod,
        )
  - GradCAMSaliency is reused as-is from comprehensive_evaluation.py.
  - Data loading uses the same `data` package wrapper:
        from data import DatasetType, get_dataloader
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make local xvmamba/ importable regardless of where the script is launched
# from (mirror the same pattern as comprehensive_evaluation.py).
THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[1]))  # xvmamba/

from controllability.analyzer import (
    ControllabilityAnalyzer,
    ControllabilityMethod,
)
from data import DatasetType, get_dataloader  # noqa: E402

# Reuse the project's GradCAMSaliency and load_model exactly as the
# headline evaluation does, to keep methodology consistent.
from evaluation.comprehensive_evaluation import (  # noqa: E402
    GradCAMSaliency,
    load_model,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SK = True
except ImportError:
    HAS_SK = False


# =============================================================================
# Numerical utilities
# =============================================================================

EPS = 1e-12


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2D array to [0, 1]; flat input → zeros."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + EPS)


def spatial_entropy(p: np.ndarray) -> float:
    """Shannon entropy (bits) of a 2D map treated as a distribution."""
    p = p.flatten().astype(np.float64)
    p = np.clip(p, 0, None)
    s = p.sum()
    if s < EPS:
        return 0.0
    p = p / s
    return float(-(p * np.log2(p + EPS)).sum())


def gini(x: np.ndarray) -> float:
    """Concentration / inequality measure on a non-negative 1D/2D array.
    0 = uniform, 1 = all mass at a single position."""
    x = np.clip(x.flatten().astype(np.float64), 0, None)
    if x.sum() < EPS:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def topk_mass(x: np.ndarray, frac: float = 0.10) -> float:
    """Fraction of total mass held by the top-`frac` of positions."""
    x = np.clip(x.flatten().astype(np.float64), 0, None)
    if x.sum() < EPS:
        return 0.0
    k = max(1, int(round(frac * x.size)))
    return float(np.sort(x)[-k:].sum() / x.sum())


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 2D maps (after centring)."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < EPS:
        return 0.0
    return float((a @ b) / den)


def upsample_to(map_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear-upsample a 2D map to (H, W)."""
    t = torch.from_numpy(map_2d.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0, 0].numpy().astype(np.float64)


# =============================================================================
# Per-image computation
# =============================================================================

@dataclass
class PerImage:
    idx: int
    label: int
    pred: int
    correct: int           # 0 / 1
    confidence: float      # softmax probability of predicted class

    # --- H1: J–G gap stats ---
    delta_mean: float          # mean Δ over the aggregated map
    delta_max: float           # max Δ
    delta_gini: float          # concentration of Δ
    delta_topk_mass: float     # top-10% mass fraction of Δ
    delta_per_stage: List[float]  # mean Δ at each VMamba stage

    # --- H2: structural-attributive disagreement ---
    corr_J_GC: float       # Pearson(J_aggregated, GradCAM)
    corr_G_GC: float       # Pearson(G_aggregated, GradCAM)
    corr_J_G: float        # Pearson(J_aggregated, G_aggregated)
    D_x: float             # 1 - corr_J_GC

    # --- bookkeeping ---
    stages_used: int = 0


def per_stage_delta_from_dicts(stage_J: Dict[int, list], stage_G: Dict[int, list]) -> List[float]:
    """Per-stage mean Δ from pre-aggregated per-stage block maps (already
    upsampled to the input resolution and normalised). For each stage we
    take the mean of (block-mean of J) and (block-mean of G), then mean(|J − G|)."""
    out: List[float] = []
    keys = sorted(set(stage_J) | set(stage_G))
    for s in keys:
        jb = stage_J.get(s, [])
        gb = stage_G.get(s, [])
        if not jb or not gb:
            out.append(0.0); continue
        j = torch.stack(jb, dim=0).mean(dim=0)
        g = torch.stack(gb, dim=0).mean(dim=0)
        # min-max per stage so Δ lives on [0, 1] for cross-image comparability
        def _mm(x):
            lo, hi = x.min(), x.max()
            if (hi - lo).item() < EPS: return torch.zeros_like(x)
            return (x - lo) / (hi - lo + EPS)
        j, g = _mm(j), _mm(g)
        out.append(float((j - g).abs().mean().item()))
    return out


def aggregate_influence(analyzer, analysis, target_size) -> Tuple[Optional[torch.Tensor], Dict[int, list]]:
    """Run analyzer.analyze_block per (stage, block), upsample to target_size,
    and return (uniform-mean aggregated map, per-stage block maps dict).
    Mirrors StructuralControllabilitySaliency._weighted_aggregate in
    comprehensive_evaluation.py but with uniform weighting (matches the
    aggregation used for the published Table V numbers)."""
    layer_maps: list = []
    stage_maps: Dict[int, list] = {}
    for stage_idx, stage_caches in enumerate(analysis.stage_caches):
        stage_maps.setdefault(stage_idx, [])
        for block_idx, cache in enumerate(stage_caches):
            result = analyzer.analyze_block(
                cache, stage_idx=stage_idx, block_idx=block_idx,
            )
            imap = result.influence_map
            if imap is None:
                continue
            if imap.shape[0] != target_size[0] or imap.shape[1] != target_size[1]:
                imap = F.interpolate(
                    imap.unsqueeze(0).unsqueeze(0),
                    size=target_size, mode="bilinear", align_corners=False,
                ).squeeze()
            layer_maps.append(imap)
            stage_maps[stage_idx].append(imap)
    if not layer_maps:
        return None, stage_maps
    aggregated = torch.stack(layer_maps, dim=0).mean(dim=0)
    return aggregated, stage_maps


def analyze_image(model, analyzer_J, analyzer_G, gradcam, image, label, device):
    """Run J / G / GC on one image and return a PerImage record."""
    model.eval()
    image_dev = image.to(device)
    H_in, W_in = image_dev.shape[-2], image_dev.shape[-1]

    # --- (a) Analysis-mode forward (matches comprehensive_evaluation.py) ---
    if hasattr(model, "enable_analysis_mode"):
        model.enable_analysis_mode(store_states=False)
    with torch.no_grad():
        logits, analysis = model(image_dev, return_analysis=True)
    probs = F.softmax(logits[0], dim=-1)
    pred = int(probs.argmax().item())
    confidence = float(probs[pred].item())
    correct = int(pred == int(label))

    # --- (b) Aggregate per-block J / G to (H_in, W_in) via uniform mean ---
    J_t, stage_J = aggregate_influence(analyzer_J, analysis, (H_in, W_in))
    G_t, stage_G = aggregate_influence(analyzer_G, analysis, (H_in, W_in))
    if hasattr(model, "disable_analysis_mode"):
        model.disable_analysis_mode()
    if J_t is None or G_t is None:
        raise RuntimeError("aggregate_influence returned no maps; check analysis.stage_caches")
    J = minmax(to_np(J_t))
    G = minmax(to_np(G_t))

    # --- (c) Grad-CAM for the predicted class (does its own fwd+bwd) ---
    GC_raw = gradcam.generate(image_dev, pred)
    GC = to_np(GC_raw)
    if GC.ndim != 2:
        raise RuntimeError(f"expected 2D Grad-CAM map, got GC.shape={GC.shape}")
    if GC.shape != J.shape:
        GC = upsample_to(GC, int(J.shape[0]), int(J.shape[1]))
    GC = minmax(GC)

    # --- (d) H1 stats: Δ map and concentration metrics ---
    Delta = np.abs(J - G)
    delta_stage = per_stage_delta_from_dicts(stage_J, stage_G)

    # --- (e) H2 stats: spatial agreement between maps ---
    corr_J_GC = spatial_corr(J, GC)
    corr_G_GC = spatial_corr(G, GC)
    corr_J_G  = spatial_corr(J, G)
    D_x = 1.0 - corr_J_GC

    return PerImage(
        idx=-1,                       # filled in by caller
        label=int(label),
        pred=pred,
        correct=correct,
        confidence=confidence,
        delta_mean=float(Delta.mean()),
        delta_max=float(Delta.max()),
        delta_gini=gini(Delta),
        delta_topk_mass=topk_mass(Delta, frac=0.10),
        delta_per_stage=delta_stage,
        corr_J_GC=corr_J_GC,
        corr_G_GC=corr_G_GC,
        corr_J_G=corr_J_G,
        D_x=D_x,
        stages_used=len(delta_stage),
    ), J, G, GC, Delta, image[0]


# =============================================================================
# Aggregation across images (H1 + H2 statistics)
# =============================================================================

def aggregate(records: List[PerImage]) -> Dict:
    """Compute the headline numbers that test H1 and H2."""
    arr = lambda key: np.array([getattr(r, key) for r in records], dtype=np.float64)
    correct = arr("correct")
    conf = arr("confidence")
    D = arr("D_x")

    out: Dict = {
        "n_images":            len(records),
        "n_correct":           int(correct.sum()),
        "accuracy":            float(correct.mean()),
        "mean_confidence":     float(conf.mean()),
        # --- H1 aggregate ---
        "H1": {
            "mean_delta":          float(arr("delta_mean").mean()),
            "median_delta":        float(np.median(arr("delta_mean"))),
            "mean_delta_gini":     float(arr("delta_gini").mean()),
            "mean_delta_topk":     float(arr("delta_topk_mass").mean()),
            # per-stage profile, averaged over images
        },
        # --- H2 aggregate ---
        "H2": {
            "mean_corr_J_GC":     float(arr("corr_J_GC").mean()),
            "mean_corr_G_GC":     float(arr("corr_G_GC").mean()),
            "mean_corr_J_G":      float(arr("corr_J_G").mean()),
            "mean_D":             float(D.mean()),
            "D_corr_with_uncertainty": float(np.corrcoef(D, 1.0 - conf)[0, 1]),
        },
    }

    # Per-stage Δ profile: list of (mean Δ, std Δ) across images at each stage.
    n_stages = max((r.stages_used for r in records), default=0)
    stage_profile = []
    for s in range(n_stages):
        vals = [r.delta_per_stage[s] for r in records if len(r.delta_per_stage) > s]
        if vals:
            stage_profile.append({
                "stage": s,
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "n":    len(vals),
            })
    out["H1"]["per_stage_delta"] = stage_profile

    # H2 AUROC: does D(x) predict misclassification?
    incorrect = 1 - correct.astype(np.int32)
    if HAS_SK and 0 < incorrect.sum() < len(incorrect):
        out["H2"]["AUROC_D_predicts_misclass"] = float(roc_auc_score(incorrect, D))
        # baseline: 1 - confidence (the obvious uncertainty proxy)
        out["H2"]["AUROC_baseline_1mC"] = float(roc_auc_score(incorrect, 1.0 - conf))
    else:
        out["H2"]["AUROC_D_predicts_misclass"] = None
        out["H2"]["AUROC_baseline_1mC"] = None
        if not HAS_SK:
            out["H2"]["AUROC_note"] = "sklearn not available; AUROC skipped"
        else:
            out["H2"]["AUROC_note"] = "AUROC undefined (all correct or all wrong)"

    return out


# =============================================================================
# Reporting / plotting
# =============================================================================

def save_per_image_csv(records: List[PerImage], path: Path):
    with open(path, "w", newline="") as f:
        if not records:
            return
        d0 = asdict(records[0])
        # Flatten the per-stage list into stage_0_delta, stage_1_delta, ...
        keys = [k for k in d0.keys() if k != "delta_per_stage"]
        max_stages = max(len(r.delta_per_stage) for r in records)
        stage_keys = [f"delta_stage_{i}" for i in range(max_stages)]
        w = csv.writer(f)
        w.writerow(keys + stage_keys)
        for r in records:
            row = [getattr(r, k) for k in keys]
            row += list(r.delta_per_stage) + [""] * (max_stages - len(r.delta_per_stage))
            w.writerow(row)


def plot_per_stage_delta(summary: Dict, out: Path):
    if not HAS_MPL: return
    sp = summary["H1"]["per_stage_delta"]
    if not sp: return
    stages = [s["stage"] for s in sp]
    means  = [s["mean"]  for s in sp]
    stds   = [s["std"]   for s in sp]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(stages, means, yerr=stds, capsize=4, color="#cc6677", alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("VMamba stage")
    ax.set_ylabel(r"Mean $|J_k - G_k|$")
    ax.set_title("H1: per-stage selectivity ($\\Delta_k$) profile")
    ax.set_xticks(stages)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_confidence_vs_D(records: List[PerImage], out: Path):
    if not HAS_MPL: return
    D = np.array([r.D_x for r in records])
    C = np.array([r.confidence for r in records])
    ok = np.array([r.correct for r in records]).astype(bool)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(C[ok],  D[ok],  s=18, color="#4477aa", alpha=0.7, label="correct")
    ax.scatter(C[~ok], D[~ok], s=22, color="#cc3311", alpha=0.9, label="incorrect", marker="x")
    ax.set_xlabel("Model confidence on predicted class")
    ax.set_ylabel(r"$D(x) = 1 - \mathrm{corr}(J, \mathrm{GradCAM})$")
    ax.set_title("H2: disagreement vs.\\ confidence")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_roc(records: List[PerImage], summary: Dict, out: Path):
    if not HAS_MPL or not HAS_SK: return
    from sklearn.metrics import roc_curve
    D = np.array([r.D_x for r in records])
    incorrect = 1 - np.array([r.correct for r in records])
    if 0 < incorrect.sum() < len(incorrect):
        fpr, tpr, _ = roc_curve(incorrect, D)
        fpr_b, tpr_b, _ = roc_curve(incorrect, 1.0 - np.array([r.confidence for r in records]))
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(fpr, tpr, color="#cc3311",
                label=f"D(x), AUROC={summary['H2']['AUROC_D_predicts_misclass']:.3f}")
        ax.plot(fpr_b, tpr_b, color="#4477aa", linestyle="--",
                label=f"1 - conf, AUROC={summary['H2']['AUROC_baseline_1mC']:.3f}")
        ax.plot([0,1], [0,1], color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("H2: AUROC of D(x) for misclassification")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        plt.close()


def save_examples(images_data, out_dir: Path, n: int = 12):
    """Save a small grid of (image | J | G | Δ | GC) heatmaps for visual
    inspection in the writeup."""
    if not HAS_MPL: return
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (idx, img_chw, J, G, GC, Delta, rec) in enumerate(images_data[:n]):
        fig, axes = plt.subplots(1, 5, figsize=(14, 3))
        img = img_chw.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + EPS)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
            axes[0].imshow(img, cmap="gray")
        else:
            axes[0].imshow(img)
        axes[0].set_title(f"img idx={idx}\\nlabel={rec.label} pred={rec.pred} ({'OK' if rec.correct else 'X'})")
        for ax, m, t in [
            (axes[1], J,     "J (Jacobian)"),
            (axes[2], G,     "G (Gramian)"),
            (axes[3], Delta, r"$\Delta = |J-G|$"),
            (axes[4], GC,    "Grad-CAM"),
        ]:
            ax.imshow(m, cmap="hot", vmin=0, vmax=max(EPS, m.max()))
            ax.set_title(t)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"D(x) = {rec.D_x:.3f}   conf = {rec.confidence:.3f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"example_{i:02d}_idx{idx}.png", dpi=160)
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Phase 6 selectivity-disagreement pipeline")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True,
                    help="bloodmnist | dermamnist | octmnist | pneumoniamnist | cifar100")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--save_examples", type=int, default=12,
                    help="number of full-resolution example heatmap grids to save")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[warn] CUDA unavailable; falling back to {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + analyzers + GradCAM ----
    print(f"[load] checkpoint: {args.checkpoint}")
    model, _config = load_model(args.checkpoint, device)
    model.eval()
    analyzer_J = ControllabilityAnalyzer(method=ControllabilityMethod.JACOBIAN, normalize=True)
    analyzer_G = ControllabilityAnalyzer(method=ControllabilityMethod.GRAMIAN,  normalize=True)
    gradcam = GradCAMSaliency(model, device)

    # ---- Data loader ----
    ds_type = DatasetType(args.dataset)
    num_classes = getattr(_config, "num_classes", None)
    image_size  = getattr(_config, "image_size", 224)
    _, _, test_loader = get_dataloader(
        dataset_type=ds_type, batch_size=1, num_workers=0,
        image_size=image_size, data_root=args.data_root,
    )
    print(f"[load] dataset: {args.dataset}  num_classes={num_classes}  image_size={image_size}")

    # ---- Loop ----
    records: List[PerImage] = []
    saved_examples: list = []
    for i, batch in enumerate(test_loader):
        if i >= args.num_samples:
            break
        try:
            image, label = batch
        except Exception:
            # tolerate dict-style yields
            image, label = batch["image"], batch["label"]
        try:
            rec, J, G, GC, Delta, img_chw = analyze_image(
                model, analyzer_J, analyzer_G, gradcam, image, int(label.item()), device,
            )
            rec.idx = i
            records.append(rec)
            if i < args.save_examples:
                saved_examples.append((i, img_chw.cpu(), J, G, GC, Delta, rec))
        except Exception as e:
            import traceback
            print(f"[warn] image {i} failed: {type(e).__name__}: {e}")
            if i == 0:
                # First failure: dump full traceback so debugging is one-shot.
                traceback.print_exc()
            continue
        if (i + 1) % 25 == 0:
            print(f"  processed {i+1}/{args.num_samples} images")

    print(f"[done] processed {len(records)} images")

    # ---- Aggregate ----
    summary = aggregate(records)
    summary["dataset"] = args.dataset
    summary["checkpoint"] = str(args.checkpoint)
    summary["num_samples_requested"] = args.num_samples
    summary["num_samples_actual"]    = len(records)
    summary["device"] = device

    # ---- Write outputs ----
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    save_per_image_csv(records, out_dir / "per_image.csv")
    plot_per_stage_delta(summary, out_dir / "per_stage_delta.png")
    plot_confidence_vs_D(records, out_dir / "confidence_vs_D.png")
    plot_roc(records, summary, out_dir / "disagreement_auroc.png")
    save_examples(saved_examples, out_dir / "examples", n=args.save_examples)

    # ---- Print headline ----
    print("\n" + "=" * 78)
    print(f"  Headline results — {args.dataset}  (n={summary['num_samples_actual']})")
    print("=" * 78)
    print(f"  accuracy on subset:                 {summary['accuracy']:.3f}")
    print(f"  mean confidence:                    {summary['mean_confidence']:.3f}")
    print(f"  H1  mean Δ (= |J − G|):             {summary['H1']['mean_delta']:.4f}")
    print(f"  H1  Δ Gini (concentration):         {summary['H1']['mean_delta_gini']:.4f}")
    print(f"  H1  Δ top-10% mass fraction:        {summary['H1']['mean_delta_topk']:.4f}")
    print(f"  H1  per-stage Δ profile (mean):")
    for s in summary["H1"]["per_stage_delta"]:
        print(f"        stage {s['stage']}: {s['mean']:.4f} ± {s['std']:.4f}")
    print(f"  H2  mean corr(J, GradCAM):          {summary['H2']['mean_corr_J_GC']:+.4f}")
    print(f"  H2  mean corr(J, G):                {summary['H2']['mean_corr_J_G']:+.4f}")
    print(f"  H2  D(x) corr with (1 - conf):      {summary['H2']['D_corr_with_uncertainty']:+.4f}")
    auroc_D = summary['H2'].get('AUROC_D_predicts_misclass')
    auroc_b = summary['H2'].get('AUROC_baseline_1mC')
    if auroc_D is not None:
        print(f"  H2  AUROC D(x) → misclass:          {auroc_D:.3f}")
        print(f"  H2  AUROC (1 - conf) baseline:      {auroc_b:.3f}")
    else:
        print(f"  H2  AUROC: {summary['H2'].get('AUROC_note', 'n/a')}")
    print("\n  outputs:")
    for fn in ("summary.json", "per_image.csv", "per_stage_delta.png",
               "confidence_vs_D.png", "disagreement_auroc.png"):
        p = out_dir / fn
        print(f"    {'✓' if p.exists() else 'x'}  {p}")
    print(f"    ✓  {out_dir/'examples/'}/   ({len(saved_examples)} grids)")


if __name__ == "__main__":
    main()
