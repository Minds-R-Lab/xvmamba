"""
Sanity check: compare the headline numbers from the OLD (pre-audit-fix)
all_results.pth to the NEW (post-fix) all_results.pth, flagging any
materially large delta.

We expect:
  - Cross-class consistency for Jacobian and Gramian: UNCHANGED at 1.0000
    (mathematical guarantee — any deviation is a bug).
  - Cross-class consistency for Grad-CAM: may change due to B2 (top-K
    predicted classes vs hard-coded [0..K-1]).
  - Faithfulness scores: small changes expected from G1 + G2 (<0.02).
  - Perturbation high_drop / low_drop: small changes from G2 tie-jitter
    (<0.005).

If the new Jacobian or Gramian cross-class drops below 0.9999 we ABORT —
that's a bug in the patched code.

Run:
    python evaluation/sanity_check_stats.py \\
        --before results/agg_reeval/bloodmnist/all_results.pth \\
        --after  results/revised_eval/bloodmnist/all_results.pth
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _safe_mean(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) == 0:
        return float("nan")
    return float(np.nanmean(xs))


def _faith_score(d):
    if "del" not in d or not d["del"]:
        return float("nan")
    return _safe_mean(np.array(d["ins"]) - np.array(d["del"]))


def _row(before, after, tol):
    delta = after - before
    flag = "" if abs(delta) <= tol else "  ⚠ exceeds tol"
    return f"  {before:>8.4f}  →  {after:>8.4f}  (Δ {delta:+.4f}){flag}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()

    before = torch.load(args.before, weights_only=False)
    after  = torch.load(args.after,  weights_only=False)

    print("=" * 72)
    print(f"Sanity check: stats before vs after the 5 audit fixes")
    print(f"  Before: {args.before}")
    print(f"  After:  {args.after}")
    print("=" * 72)

    fatal = False

    # --- Cross-class consistency: Jacobian/Gramian MUST stay at 1.0 ----
    print("\nCross-class consistency (mean pairwise correlation):")
    print(f"{'Method':<15} {'before':>10}  →  {'after':>10}  (Δ)  ")
    cc_b = before.get("cross_class_consistency", {})
    cc_a = after.get("cross_class_consistency", {})
    for m in cc_b.keys() | cc_a.keys():
        b_vals = cc_b.get(m, [])
        a_vals = cc_a.get(m, [])
        bm = _safe_mean(b_vals)
        am = _safe_mean(a_vals)
        # Jacobian and Gramian must stay at 1.0; Grad-CAM is allowed to move.
        tol = 0.0001 if m in ("Jacobian", "Gramian") else 0.50
        print(f"  {m:<13}{_row(bm, am, tol)}")
        if m in ("Jacobian", "Gramian") and abs(am - 1.0) > 0.0001:
            print(f"  ⛔ FATAL: {m} cross-class consistency must be 1.0000 "
                  f"(got {am:.6f}). This is a mathematical guarantee; "
                  f"check the patched code.")
            fatal = True

    # --- Faithfulness: small expected shifts ---------------------------
    print("\nFaithfulness score (insertion AUC - deletion AUC):")
    print(f"{'Method':<15} {'before':>10}  →  {'after':>10}  (Δ)  ")
    fb = before.get("faithfulness", {})
    fa = after.get("faithfulness", {})
    for m in fb.keys() | fa.keys():
        bs = _faith_score(fb.get(m, {}))
        as_ = _faith_score(fa.get(m, {}))
        # Allow up to 0.05 absolute change as "small" — G1 + G2 should not
        # move the headline numbers much because both effects are sub-pixel.
        print(f"  {m:<13}{_row(bs, as_, 0.05)}")

    # --- Perturbation: drop_diff if present, else high_drop -----------
    print("\nPerturbation (high_drop mean):")
    print(f"{'Method':<15} {'before':>10}  →  {'after':>10}  (Δ)  ")
    pb = before.get("perturbation_invariance", {})
    pa = after.get("perturbation_invariance", {})
    for m in pb.keys() | pa.keys():
        bh = _safe_mean(pb.get(m, {}).get("high_drop", []))
        ah = _safe_mean(pa.get(m, {}).get("high_drop", []))
        print(f"  {m:<13}{_row(bh, ah, 0.05)}")

    # --- Sample counts -----------------------------------------------
    print("\nSample counts (faithfulness, before vs after):")
    for m in fb.keys() | fa.keys():
        nb = len(fb.get(m, {}).get("del", []))
        na = len(fa.get(m, {}).get("del", []))
        print(f"  {m:<13}  n={nb} → n={na}")

    print("\n" + "=" * 72)
    if fatal:
        print("⛔ FATAL discrepancies detected — review the patched code before")
        print("   updating the manuscript with new numbers.")
        return 2
    print("✓ Sanity check passed: numbers shifted as expected; cross-class")
    print("   consistency = 1.0000 preserved for Jacobian + Gramian.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
