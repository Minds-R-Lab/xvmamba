#!/usr/bin/env python3
"""
multiseed_aggregate.py
----------------------
Aggregate per-seed `all_results.pth` files (produced by
`comprehensive_evaluation.py`) into a single per-dataset summary
reporting mean ± std across seeds.

What this measures (and what it does NOT measure):
    For each metric we already track (insertion AUC, deletion AUC,
    faithfulness = I - D, cross-class consistency, perturbation
    drop-diff at each k), we compute:
        per_seed_mean[i] = mean over the test images, for seed i
        agg_mean         = mean of per_seed_mean over seeds
        agg_std          = standard deviation of per_seed_mean over seeds

    Interpretation: `agg_std` captures variance across TRAINING RUNS
    (random initialisation, data order). It is a different and
    complementary source of variance from the bootstrap-over-images
    CIs reported in Table V, which capture variance across TEST IMAGES
    given a fixed checkpoint.

Output (JSON):
    {
      "dataset": "bloodmnist",
      "n_seeds": 3,
      "seeds_used": [42, 137, 2024],
      "methods": {
        "jacobian": {
            "faithfulness": {"mean": 0.631, "std": 0.013, "per_seed": [...]},
            "insertion":    {...},
            "deletion":     {...},
            "cross_class":  {"mean": 1.0, "std": 0.0, ...},
            ...
        },
        "gramian":  {...},
        "gradcam":  {...},
        "random":   {...},
      }
    }

Usage:
    python3 evaluation/multiseed_aggregate.py \\
        --dataset bloodmnist \\
        --output results/multiseed/multiseed_summary_bloodmnist.json \\
        results/multiseed/seed_42/bloodmnist/all_results.pth \\
        results/multiseed/seed_137/bloodmnist/all_results.pth \\
        results/multiseed/seed_2024/bloodmnist/all_results.pth
"""
import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev

import torch


# Methods we expect in all_results.pth. Keys match comprehensive_evaluation.py.
METHODS = ["jacobian", "gramian", "gradcam", "random"]

# Per-method metrics we want to aggregate. Each entry is
# (display_name, dotted_path_into_all_results[method]).
# We use a lenient lookup that tries each candidate path so the script
# survives minor changes to the dict layout produced by
# comprehensive_evaluation.py.
METRIC_PATHS = {
    # Faithfulness: I - D (per-sample list lives under different names in
    # different versions of the eval script).
    "faithfulness":  ["faithfulness._raw", "faithfulness.scores", "scores._raw", "faithfulness_score._raw"],
    "insertion":     ["insertion._raw", "insertion_auc._raw", "insertion_auc.scores"],
    "deletion":      ["deletion._raw", "deletion_auc._raw", "deletion_auc.scores"],
    # Cross-class consistency: per-sample mean pairwise correlation list.
    "cross_class":   ["cross_class._raw", "cross_class.scores", "consistency._raw"],
    # Perturbation drop-diff at high-saliency removal (audit fix B1).
    "drop_diff_high": ["perturbation.high.drop_diff._raw", "perturbation.drop_diff_high._raw", "drop_diff._raw"],
}


def _lookup(d, dotted):
    """Walk a dotted path into a nested dict; return None on miss."""
    cur = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _per_sample_mean(results_for_method, candidate_paths):
    """Return the mean over the per-sample list for one (seed, method)
    combination, trying each candidate dotted path in order."""
    for path in candidate_paths:
        v = _lookup(results_for_method, path)
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().tolist()
        if isinstance(v, (list, tuple)) and len(v) > 0:
            try:
                return float(sum(v) / len(v))
            except TypeError:
                continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_seed_from_path(path: Path) -> int:
    """Look for `seed_<int>` anywhere in the parent path."""
    for part in path.parts:
        m = re.fullmatch(r"seed_(\d+)", part)
        if m:
            return int(m.group(1))
    raise ValueError(f"could not infer seed from path: {path}")


def aggregate(seed_files, dataset):
    per_seed_payloads = []
    for f in seed_files:
        f = Path(f)
        try:
            payload = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(f, map_location="cpu")
        per_seed_payloads.append((_extract_seed_from_path(f), payload))

    out = {
        "dataset": dataset,
        "n_seeds": len(per_seed_payloads),
        "seeds_used": [s for s, _ in per_seed_payloads],
        "methods": {},
    }

    for method in METHODS:
        method_out = {}
        for metric_name, paths in METRIC_PATHS.items():
            per_seed_values = []
            for seed, payload in per_seed_payloads:
                # the per-method dict may live under several layouts
                method_dict = payload.get(method) if isinstance(payload, dict) else None
                if method_dict is None and isinstance(payload, dict):
                    # fall back: results may be keyed (method, metric) at the top
                    method_dict = {k.replace(f"{method}_", ""): v
                                   for k, v in payload.items()
                                   if isinstance(k, str) and k.startswith(method + "_")}
                if not method_dict:
                    continue
                val = _per_sample_mean(method_dict, paths)
                if val is not None:
                    per_seed_values.append((seed, val))
            if len(per_seed_values) >= 2:
                vals = [v for _, v in per_seed_values]
                method_out[metric_name] = {
                    "mean":   mean(vals),
                    "std":    pstdev(vals),       # population std across seeds
                    "per_seed": {s: v for s, v in per_seed_values},
                    "n":      len(vals),
                }
        out["methods"][method] = method_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True, help="path to write summary JSON")
    ap.add_argument("seed_files", nargs="+",
                    help="all_results.pth files (one per seed)")
    args = ap.parse_args()

    summary = aggregate(args.seed_files, args.dataset)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Brief stdout summary so the runbook log shows the headline numbers.
    print(f"\n=== multi-seed summary: {args.dataset} ({summary['n_seeds']} seeds) ===")
    for method, mout in summary["methods"].items():
        line_parts = [f"{method:10s}"]
        for metric in ("faithfulness", "insertion", "deletion", "cross_class"):
            if metric in mout:
                m = mout[metric]
                line_parts.append(f"{metric}={m['mean']:+.4f}±{m['std']:.4f}")
        print("  " + "  ".join(line_parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
