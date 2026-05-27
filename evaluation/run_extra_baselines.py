"""
Evaluate the additional saliency baselines (Score-CAM, Integrated Gradients,
RISE) on existing checkpoints.

This complements `comprehensive_evaluation.py`: rather than re-running the
full four-test suite for the new methods, we focus on the three tests where
the new methods are most informative for the response letter --

  - Faithfulness (insertion AUC minus deletion AUC) -- this is the
    headline metric in Table V and the only one where Grad-CAM beat us
    on the lower-accuracy datasets in v1.

  - Cross-class consistency -- to confirm that the new gradient-based
    baselines remain class-dependent (their cross-class correlation is
    expected to be similar to Grad-CAM's, well below 1.0), reinforcing
    the structural-vs-attributive distinction.

  - Perturbation invariance (optional, --include-perturbation flag) --
    occlusion-based sanity check that the new maps actually identify
    influential regions.

Result JSON layout matches the existing `all_results.pth` so the new
numbers can be merged into the v2 manuscript tables without any further
post-processing.

Run:
    python evaluation/run_extra_baselines.py \\
        --checkpoint ./checkpoints/bloodmnist/best_model.pth \\
        --dataset bloodmnist \\
        --num_samples 50 \\
        --output_dir ./results/extra_baselines/bloodmnist
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm

# We reuse the evaluation infrastructure already in the codebase.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import DatasetType, get_dataloader
from evaluation.comprehensive_evaluation import (
    load_model,                 # checkpoint -> (model, config)
    FaithfulnessEvaluator,      # del/ins AUC computation
)
from evaluation.extra_baselines import (
    ScoreCAMSaliency,
    IntegratedGradientsSaliency,
    RISESaliency,
)


# ---------------------------------------------------------------------------
def _cross_class_correlation(maps: list[torch.Tensor]) -> float:
    """Mean pairwise Pearson correlation between a list of 2D maps."""
    if len(maps) < 2:
        return float("nan")
    vecs = torch.stack([m.flatten() for m in maps]).numpy()
    # Use numpy corrcoef for the upper triangle.
    corr = np.corrcoef(vecs)
    n = corr.shape[0]
    upper = corr[np.triu_indices(n, k=1)]
    return float(np.nanmean(upper))


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Extra-baselines evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_test_classes", type=int, default=5,
                        help="cross-class consistency: classes per image")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    # Compute-control knobs for the slow methods.
    parser.add_argument("--score_cam_top_k", type=int, default=32)
    parser.add_argument("--ig_steps", type=int, default=20)
    parser.add_argument("--rise_num_masks", type=int, default=500)
    parser.add_argument("--rise_batch_size", type=int, default=32)
    parser.add_argument("--include_perturbation", action="store_true",
                        help="also run a small perturbation-invariance check")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir or f"./results/extra_baselines/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXTRA BASELINES EVALUATION")
    print("=" * 72)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  Output:      {out}")
    print(f"  Device:      {device}")
    print()

    # ---- load model + data --------------------------------------------------
    model, config = load_model(args.checkpoint, device)
    dataset_type = DatasetType(args.dataset)
    _, _, test_loader = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )
    num_classes = config.num_classes

    # ---- build methods ------------------------------------------------------
    methods = {
        "Score-CAM":
            ScoreCAMSaliency(model, device, top_k=args.score_cam_top_k),
        "Integrated Gradients":
            IntegratedGradientsSaliency(model, device, steps=args.ig_steps),
        "RISE":
            RISESaliency(
                model, device,
                num_masks=args.rise_num_masks,
                batch_size=args.rise_batch_size,
            ),
    }
    fevaluator = FaithfulnessEvaluator(model, device)

    # =======================================================================
    # 1) Cross-class consistency
    # =======================================================================
    print("-" * 72)
    print("TEST: Cross-class consistency (target class invariance)")
    print("-" * 72)
    per_method_corr: dict[str, list[float]] = {n: [] for n in methods}

    sample_count = 0
    iterator = tqdm(test_loader, total=min(args.num_samples, len(test_loader)),
                    desc="Cross-class")
    for images, labels in iterator:
        if sample_count >= args.num_samples:
            break
        img = images[0:1].to(device)

        # Choose `num_test_classes` distinct target classes.
        with torch.no_grad():
            logits = model(img)
        pred = logits[0].argmax().item()
        # Top-k predicted classes (excluding rank-1) + add the true label.
        topk = logits[0].topk(num_classes).indices.tolist()
        candidate_classes = [pred] + [c for c in topk if c != pred][:args.num_test_classes - 1]
        candidate_classes = candidate_classes[:args.num_test_classes]

        for name, method in methods.items():
            maps = []
            for tc in candidate_classes:
                try:
                    maps.append(method.generate(img, tc))
                except Exception as e:
                    # Robust to per-image failures (e.g., zero-grad
                    # pathologies in IG); skip and continue.
                    continue
            if len(maps) >= 2:
                per_method_corr[name].append(_cross_class_correlation(maps))

        sample_count += 1

    crossclass_summary = {}
    print(f"{'Method':<22} {'Mean correlation':<20} {'Std':<10}")
    for name in methods:
        vals = per_method_corr[name]
        if vals:
            m = float(np.mean(vals))
            s = float(np.std(vals))
            crossclass_summary[name] = {"mean_correlation": m, "std": s, "n": len(vals)}
            print(f"{name:<22} {m:<20.4f} {s:<10.4f}")
        else:
            crossclass_summary[name] = {"mean_correlation": float("nan"), "std": 0.0, "n": 0}
            print(f"{name:<22} (no samples)")

    # =======================================================================
    # 2) Faithfulness (insertion AUC minus deletion AUC)
    # =======================================================================
    print("\n" + "-" * 72)
    print("TEST: Faithfulness (insertion AUC minus deletion AUC)")
    print("-" * 72)
    faith_raw = {n: {"del": [], "ins": []} for n in methods}

    sample_count = 0
    for images, labels in tqdm(test_loader,
                               total=min(args.num_samples, len(test_loader)),
                               desc="Faithfulness"):
        if sample_count >= args.num_samples:
            break
        img = images[0:1].to(device)
        with torch.no_grad():
            pred = int(model(img).argmax(dim=1).item())

        for name, method in methods.items():
            try:
                sal = method.generate(img, pred)
                del_auc = fevaluator.compute_deletion(img, sal, pred)
                ins_auc = fevaluator.compute_insertion(img, sal, pred)
                faith_raw[name]["del"].append(del_auc)
                faith_raw[name]["ins"].append(ins_auc)
            except Exception as e:
                continue

        sample_count += 1

    faith_summary = {}
    print(f"{'Method':<22} {'Deletion':<12} {'Insertion':<12} {'Score (I-D)':<12}")
    for name in methods:
        d, i = faith_raw[name]["del"], faith_raw[name]["ins"]
        if d:
            dm, im = float(np.mean(d)), float(np.mean(i))
            faith_summary[name] = {
                "deletion": dm, "insertion": im, "score": im - dm, "n": len(d),
            }
            print(f"{name:<22} {dm:<12.4f} {im:<12.4f} {im - dm:<12.4f}")
        else:
            faith_summary[name] = {"deletion": float("nan"), "insertion": float("nan"),
                                    "score": float("nan"), "n": 0}
            print(f"{name:<22} (no samples)")

    # =======================================================================
    # 3) Perturbation (optional)
    # =======================================================================
    perturb_summary = None
    if args.include_perturbation:
        print("\n" + "-" * 72)
        print("TEST: Perturbation invariance (top/bottom 10%)")
        print("-" * 72)
        perturb_summary = _perturbation(methods, model, test_loader, device,
                                         args.num_samples)
        for name, s in perturb_summary.items():
            print(f"{name:<22} high {s['high_drop']:+.4f}   low {s['low_drop']:+.4f}")

    # ---- save ---------------------------------------------------------------
    payload = {
        "cross_class_consistency": crossclass_summary,
        "faithfulness": faith_summary,
        "perturbation": perturb_summary,
        "config": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "num_samples": args.num_samples,
            "score_cam_top_k": args.score_cam_top_k,
            "ig_steps": args.ig_steps,
            "rise_num_masks": args.rise_num_masks,
        },
    }
    (out / "extra_baselines.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out / 'extra_baselines.json'}")
    return 0


# ---------------------------------------------------------------------------
def _perturbation(methods, model, dataloader, device, num_samples):
    """Quick perturbation-invariance check (top/bottom 10%) for the new methods."""
    summary = {n: {"high_drop": 0.0, "low_drop": 0.0, "n": 0} for n in methods}
    sample_count = 0
    for images, labels in tqdm(dataloader,
                               total=min(num_samples, len(dataloader)),
                               desc="Perturbation"):
        if sample_count >= num_samples:
            break
        img = images[0:1].to(device)
        with torch.no_grad():
            base_logits = model(img)
            base_probs = torch.softmax(base_logits, dim=-1)[0]
            pred = int(base_logits.argmax(dim=1).item())
            base_conf = float(base_probs[pred].item())

        H, W = img.shape[2], img.shape[3]
        topk = int(0.1 * H * W)
        for name, method in methods.items():
            try:
                sal = method.generate(img, pred).to(device)
                flat = sal.flatten()
                top_idx = flat.topk(topk).indices
                bot_idx = flat.topk(topk, largest=False).indices

                mask_high = torch.ones(H * W, device=device)
                mask_high[top_idx] = 0.0
                mask_high = mask_high.view(1, 1, H, W)

                mask_low = torch.ones(H * W, device=device)
                mask_low[bot_idx] = 0.0
                mask_low = mask_low.view(1, 1, H, W)

                with torch.no_grad():
                    conf_high = float(torch.softmax(model(img * mask_high), dim=-1)[0, pred].item())
                    conf_low = float(torch.softmax(model(img * mask_low), dim=-1)[0, pred].item())

                summary[name]["high_drop"] += base_conf - conf_high
                summary[name]["low_drop"] += base_conf - conf_low
                summary[name]["n"] += 1
            except Exception:
                continue
        sample_count += 1

    for name in methods:
        n = max(summary[name]["n"], 1)
        summary[name]["high_drop"] /= n
        summary[name]["low_drop"] /= n
    return summary


if __name__ == "__main__":
    sys.exit(main())
