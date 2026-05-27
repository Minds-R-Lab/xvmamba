"""
Bootstrap confidence intervals for evaluation metrics (Phase 2.5, R3 #4).

Reads cached per-sample evaluation results (`all_results.pth` from
`comprehensive_evaluation.py`) and computes 95% percentile bootstrap CIs
for each method's faithfulness / cross-class / perturbation scores.

The bootstrap is the standard non-parametric way to put confidence
intervals around a sample mean when the per-sample values are
exchangeable. We use the percentile method with 1000 resamples (Efron &
Tibshirani 1993, ``An Introduction to the Bootstrap''). For paired
statistics like the faithfulness score `S = ins - del`, we bootstrap the
paired per-sample (ins_i, del_i) tuples together to preserve the
correlation, then compute mean(S) on each resample.

Result format: for each `(method, metric)` pair, we emit
    {"mean": ..., "ci_low": ..., "ci_high": ..., "std": ..., "n": ...}
which can be folded directly into the manuscript tables as
"0.735 [0.71, 0.76]".

Run:
    python evaluation/bootstrap_ci.py \\
        --results results/agg_reeval/bloodmnist/all_results.pth \\
        --output results/agg_reeval/bloodmnist/bootstrap_ci.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
def _percentile_bootstrap(
    samples: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Percentile bootstrap CI on the mean of `samples`.

    Args:
        samples:     1-D array of per-sample values.
        n_resamples: number of bootstrap resamples (default 1000).
        alpha:       two-sided coverage; default 0.05 -> 95% CI.
        rng:         numpy random Generator (for reproducibility).

    Returns:
        dict with `mean`, `ci_low`, `ci_high`, `std`, `n`.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    n = len(samples)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "std": float("nan"), "n": n}
    rng = rng or np.random.default_rng(0)
    boot_means = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = samples[idx].mean()
    return {
        "mean":    float(samples.mean()),
        "ci_low":  float(np.percentile(boot_means, 100.0 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100.0 * (1 - alpha / 2))),
        "std":     float(samples.std(ddof=1)),     # sample std (N-1 denominator)
        "n":       int(n),
    }


def _paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Percentile bootstrap CI on the mean of (a_i - b_i), preserving
    pairing. Used for the faithfulness score (`insertion - deletion`)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap requires equal lengths: {len(a)} vs {len(b)}")
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "std": float("nan"), "n": n}
    rng = rng or np.random.default_rng(0)
    diffs = a - b
    boot_means = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[k] = diffs[idx].mean()
    return {
        "mean":    float(diffs.mean()),
        "ci_low":  float(np.percentile(boot_means, 100.0 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100.0 * (1 - alpha / 2))),
        "std":     float(diffs.std(ddof=1)),
        "n":       int(n),
    }


# ---------------------------------------------------------------------------
def _process(results: dict, n_resamples: int) -> dict:
    """Walk the `all_results.pth` dict and bootstrap each per-sample list.

    The eval pipeline stores per-sample lists under `*_raw` sub-keys
    alongside the summary stats:
      results["perturbation_invariance_raw"][METHOD] = {"high_drop": [...], "low_drop": [...], "ratio": [...], "drop_diff": [...]}
      results["cross_class_consistency_raw"][METHOD] = [per-image mean correlations]
      results["faithfulness_raw"][METHOD]            = {"del": [...], "ins": [...]}
    """
    rng = np.random.default_rng(0)
    out = {}

    # Faithfulness: bootstrap the paired score (ins - del) and the two
    # AUCs separately.
    faith = results.get("faithfulness_raw", {})
    out["faithfulness"] = {}
    for method, d in faith.items():
        d_del = d.get("del", [])
        d_ins = d.get("ins", [])
        del_ci = _percentile_bootstrap(d_del, n_resamples=n_resamples, rng=rng)
        ins_ci = _percentile_bootstrap(d_ins, n_resamples=n_resamples, rng=rng)
        if d_del and d_ins and len(d_del) == len(d_ins):
            score_ci = _paired_bootstrap_diff(d_ins, d_del, n_resamples=n_resamples, rng=rng)
        else:
            score_ci = {"mean": float("nan"), "ci_low": float("nan"),
                        "ci_high": float("nan"), "std": float("nan"), "n": 0}
        out["faithfulness"][method] = {
            "deletion":  del_ci,
            "insertion": ins_ci,
            "score":     score_ci,
        }

    # Cross-class consistency: per-image mean correlation. The raw test
    # returns a dict { method: list[float] }.
    cc = results.get("cross_class_consistency_raw", {})
    out["cross_class_consistency"] = {}
    for method, vals in cc.items():
        out["cross_class_consistency"][method] = _percentile_bootstrap(
            vals, n_resamples=n_resamples, rng=rng,
        )

    # Perturbation: high_drop, low_drop, drop_diff. Explicitly DO NOT
    # bootstrap the "ratio" column (unstable; see audit).
    pert = results.get("perturbation_invariance_raw", {})
    out["perturbation_invariance"] = {}
    for method, d in pert.items():
        high_ci = _percentile_bootstrap(d.get("high_drop", []), n_resamples=n_resamples, rng=rng)
        low_ci  = _percentile_bootstrap(d.get("low_drop", []),  n_resamples=n_resamples, rng=rng)
        # Prefer pre-computed drop_diff list (added by the B1 audit fix);
        # fall back to paired bootstrap on (high_drop, low_drop) if the
        # results were produced by a pre-B1-fix version.
        if d.get("drop_diff"):
            diff_ci = _percentile_bootstrap(d["drop_diff"], n_resamples=n_resamples, rng=rng)
        elif d.get("high_drop") and d.get("low_drop") and len(d["high_drop"]) == len(d["low_drop"]):
            diff_ci = _paired_bootstrap_diff(
                d["high_drop"], d["low_drop"], n_resamples=n_resamples, rng=rng,
            )
        else:
            diff_ci = {"mean": float("nan"), "ci_low": float("nan"),
                       "ci_high": float("nan"), "std": float("nan"), "n": 0}
        out["perturbation_invariance"][method] = {
            "high_drop": high_ci,
            "low_drop":  low_ci,
            "drop_diff": diff_ci,
        }

    return out


def _fmt(ci: dict, decimals: int = 3) -> str:
    """Human-readable `mean [low, high]` string for log/markdown output."""
    if ci["n"] == 0 or np.isnan(ci["mean"]):
        return "n/a"
    return f"{ci['mean']:.{decimals}f} [{ci['ci_low']:.{decimals}f}, {ci['ci_high']:.{decimals}f}]"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="path to all_results.pth (or compatible .pth) from comprehensive_evaluation")
    ap.add_argument("--output", default=None,
                    help="JSON path for bootstrap results (default: alongside input)")
    ap.add_argument("--n_resamples", type=int, default=1000)
    args = ap.parse_args()

    in_path = Path(args.results)
    out_path = Path(args.output) if args.output else in_path.with_name("bootstrap_ci.json")

    results = torch.load(in_path, weights_only=False)
    bootstrapped = _process(results, n_resamples=args.n_resamples)

    out_path.write_text(json.dumps(bootstrapped, indent=2))
    print(f"Saved: {out_path}")
    print()

    # Pretty-print a markdown-style summary for the response letter / manuscript.
    print("=" * 72)
    print(f"Bootstrap 95% CI summary  (input: {in_path}, n_resamples={args.n_resamples})")
    print("=" * 72)

    print("\nFaithfulness (insertion AUC minus deletion AUC):")
    print(f"  {'Method':<22} {'Del AUC':<25} {'Ins AUC':<25} {'Score (I-D)':<25}")
    for m, d in bootstrapped["faithfulness"].items():
        print(f"  {m:<22} {_fmt(d['deletion']):<25} {_fmt(d['insertion']):<25} {_fmt(d['score']):<25}")

    print("\nCross-class consistency (mean pairwise Pearson):")
    print(f"  {'Method':<22} {'Mean correlation':<28}")
    for m, d in bootstrapped["cross_class_consistency"].items():
        print(f"  {m:<22} {_fmt(d, decimals=4):<28}")

    print("\nPerturbation invariance (confidence drop, perturbed top vs bottom 10%):")
    print(f"  {'Method':<22} {'High drop':<25} {'Low drop':<25} {'(High - Low)':<25}")
    for m, d in bootstrapped["perturbation_invariance"].items():
        print(f"  {m:<22} {_fmt(d['high_drop']):<25} {_fmt(d['low_drop']):<25} {_fmt(d['drop_diff']):<25}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
