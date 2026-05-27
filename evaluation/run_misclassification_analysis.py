"""
Misclassification stratified analysis (Phase 3.5, R4 Q3).

R4 asked: "do the generated controllability maps provide meaningful
diagnostic insights into incorrect predictions?"

This script answers it empirically. We stratify the test set by correct
vs. incorrect prediction and report, for each stratum:

  - number of samples
  - mean confidence on the predicted class
  - faithfulness score for the Jacobian controllability index
  - faithfulness score for Grad-CAM (as a class-specific point of comparison)
  - mean pixel overlap between controllability and Grad-CAM saliency maps

The output is a JSON with all numbers + paths to the per-stratum example
indices, which the user can then use to pull qualitative misclassification
figures.

Run:
    python evaluation/run_misclassification_analysis.py \\
        --checkpoint ./checkpoints/bloodmnist/best_model.pth \\
        --dataset bloodmnist \\
        --num_samples 200 \\
        --output_dir ./results/misclassification/bloodmnist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import DatasetType, get_dataloader
from evaluation.comprehensive_evaluation import (
    load_model, StructuralControllability, GradCAMSaliency, FaithfulnessEvaluator,
)


def _pixel_overlap(map_a: torch.Tensor, map_b: torch.Tensor, frac: float = 0.1) -> float:
    """Fraction of pixels in the top-`frac` of `map_a` that are also in
    the top-`frac` of `map_b`."""
    a = map_a.flatten()
    b = map_b.flatten()
    k = max(1, int(round(len(a) * frac)))
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="evaluate up to N test images (we need a reasonable "
                             "number to get a non-trivial 'incorrect' stratum)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir or f"./results/misclassification/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MISCLASSIFICATION STRATIFIED ANALYSIS (Phase 3.5, R4 Q3)")
    print("=" * 72)
    print(f"  Dataset: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  Output: {out}")

    model, config = load_model(args.checkpoint, device)
    # Don't enable analysis mode here -- StructuralControllability toggles
    # it per-call (enable -> forward -> disable). Leaving analysis mode on
    # globally adds per-block cache-capture overhead to every faithfulness
    # forward, inflating per-image time by ~30x.
    ctrl = StructuralControllability(model, device, "jacobian")
    grad = GradCAMSaliency(model, device)
    fev = FaithfulnessEvaluator(model, device)

    dataset_type = DatasetType(args.dataset)
    _, _, test_loader = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )

    # We accumulate per-stratum lists, then summarise.
    rec = {
        "correct":   {"conf": [], "ctrl_score": [], "gradcam_score": [], "overlap": [], "idx": []},
        "incorrect": {"conf": [], "ctrl_score": [], "gradcam_score": [], "overlap": [], "idx": []},
    }

    sample_count = 0
    for img_idx, (images, labels) in enumerate(
        tqdm(test_loader, total=min(args.num_samples, len(test_loader)),
             desc="Stratify")):
        if sample_count >= args.num_samples:
            break
        img = images[0:1].to(device)
        true_label = int(labels[0].item())

        with torch.no_grad():
            logits = model(img)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(logits.argmax(dim=1).item())
        conf = float(probs[pred].item())
        stratum = "correct" if pred == true_label else "incorrect"

        try:
            sal_ctrl = ctrl.generate(img, pred)
            sal_grad = grad.generate(img, pred)
            del_c = fev.compute_deletion(img, sal_ctrl, pred)
            ins_c = fev.compute_insertion(img, sal_ctrl, pred)
            del_g = fev.compute_deletion(img, sal_grad, pred)
            ins_g = fev.compute_insertion(img, sal_grad, pred)
            overlap = _pixel_overlap(sal_ctrl, sal_grad, frac=0.1)
        except Exception:
            sample_count += 1
            continue

        rec[stratum]["conf"].append(conf)
        rec[stratum]["ctrl_score"].append(ins_c - del_c)
        rec[stratum]["gradcam_score"].append(ins_g - del_g)
        rec[stratum]["overlap"].append(overlap)
        rec[stratum]["idx"].append(img_idx)

        sample_count += 1

    summary = {}
    print(f"\n{'Stratum':<12} {'n':>5} {'mean conf':>10} "
          f"{'ctrl score':>11} {'GC score':>11} {'overlap@10%':>12}")
    for stratum in ("correct", "incorrect"):
        d = rec[stratum]
        n = len(d["conf"])
        if n == 0:
            print(f"{stratum:<12} {0:>5}  (no samples)")
            summary[stratum] = {"n": 0}
            continue
        s = {
            "n": n,
            "mean_confidence": float(np.mean(d["conf"])),
            "controllability_faithfulness": {
                "mean": float(np.mean(d["ctrl_score"])),
                "std":  float(np.std(d["ctrl_score"])),
            },
            "gradcam_faithfulness": {
                "mean": float(np.mean(d["gradcam_score"])),
                "std":  float(np.std(d["gradcam_score"])),
            },
            "top10_overlap_with_gradcam": {
                "mean": float(np.mean(d["overlap"])),
                "std":  float(np.std(d["overlap"])),
            },
            "example_indices": d["idx"][:10],   # first 10 indices for qualitative figs
        }
        summary[stratum] = s
        print(f"{stratum:<12} {n:>5} {s['mean_confidence']:>10.4f} "
              f"{s['controllability_faithfulness']['mean']:>11.4f} "
              f"{s['gradcam_faithfulness']['mean']:>11.4f} "
              f"{s['top10_overlap_with_gradcam']['mean']:>12.4f}")

    payload = {
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "num_samples_attempted": sample_count,
        "stratified_summary": summary,
    }
    (out / "misclassification_analysis.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out / 'misclassification_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
