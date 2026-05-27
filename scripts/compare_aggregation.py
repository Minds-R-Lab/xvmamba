"""
Compare L2-norm vs arithmetic-mean channel aggregation in the Jacobian
controllability analyzer.

Background: the manuscript (Eq. 36) specifies arithmetic mean across the
inner channel dimension D. Earlier revisions of `controllability/analyzer.py`
used `torch.norm(..., dim=-1)` instead. On 2026-05-14 we patched the code
to match the paper. This script quantifies how much the per-position
saliency rankings change between the two aggregations.

We compare on:
  (a) random synthetic SSM parameters that mimic the cache shape;
  (b) "realistic" parameters with the same A range as a trained VMamba-Tiny.

Reported metrics (averaged over n_trials random draws):
  - Spearman rank correlation between old and new per-position scores.
  - Pearson correlation between old and new scores.
  - Mean absolute change in the top-10% positions (i.e., how much the
    high-saliency set shifts).
  - Mean magnitude ratio (new / old) at the global scale.

Run:
    python scripts/compare_aggregation.py --batch 4 --length 64 \\
        --d_inner 128 --d_state 16 --n_trials 20
"""
import argparse
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllability.analyzer import JacobianControllability


def compute_with_aggregation(A_bar, B_bar, C, agg: str):
    """Run the backward Jacobian recursion with a specified channel aggregation."""
    batch, length, d_inner, d_state = A_bar.shape
    total = torch.zeros(batch, length)
    Q = torch.zeros(batch, d_state, d_inner)
    for k in range(length - 1, -1, -1):
        A_k = A_bar[:, k]
        B_k = B_bar[:, k]
        C_k_abs = C[:, k].abs()
        CB = torch.einsum('bn,bin->bi', C_k_abs, B_k.abs())
        QB = torch.einsum('bni,bin->bi', Q, B_k.abs())
        if agg == 'l2':
            direct = torch.norm(CB, dim=-1)
            prop = torch.norm(QB, dim=-1)
        elif agg == 'mean':
            direct = CB.mean(dim=-1)
            prop = QB.mean(dim=-1)
        else:
            raise ValueError(agg)
        total[:, k] = direct + prop
        if k > 0:
            C_expanded = C_k_abs.unsqueeze(-1).expand(-1, -1, d_inner)
            A_k_t = A_k.permute(0, 2, 1)
            Q = A_k_t * (C_expanded + Q)
    return total


def spearman(x, y):
    """Spearman rank correlation between 1-D tensors."""
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    return float(np.corrcoef(x, y)[0, 1])


def topk_overlap(x, y, frac: float = 0.1):
    """Fractional overlap of top-frac positions between two scoring schemes."""
    k = max(1, int(round(len(x) * frac)))
    sx = set(int(i) for i in torch.topk(x, k).indices)
    sy = set(int(i) for i in torch.topk(y, k).indices)
    return len(sx & sy) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--length', type=int, default=64)
    ap.add_argument('--d_inner', type=int, default=128)
    ap.add_argument('--d_state', type=int, default=16)
    ap.add_argument('--n_trials', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = []
    for trial in range(args.n_trials):
        # Realistic parameter ranges:
        #   A_bar in (0, 1) with most mass near 0.7-0.99 (slow decay, typical of trained VMamba).
        #   B_bar has small magnitude (~0.1).
        #   C is moderate magnitude (~1.0).
        A_bar = torch.sigmoid(torch.randn(args.batch, args.length, args.d_inner, args.d_state)) * 0.99 + 0.005
        B_bar = torch.randn(args.batch, args.length, args.d_inner, args.d_state) * 0.1
        C = torch.randn(args.batch, args.length, args.d_state)

        total_l2 = compute_with_aggregation(A_bar, B_bar, C, 'l2')
        total_mean = compute_with_aggregation(A_bar, B_bar, C, 'mean')

        # Compare per-batch-item.
        for b in range(args.batch):
            x_l2 = total_l2[b]
            x_mean = total_mean[b]
            rows.append({
                'trial': trial,
                'batch_item': b,
                'spearman': spearman(x_l2, x_mean),
                'pearson': pearson(x_l2, x_mean),
                'top10_overlap': topk_overlap(x_l2, x_mean, 0.1),
                'mean_l2_score': float(x_l2.mean()),
                'mean_mean_score': float(x_mean.mean()),
                'ratio_mean_over_l2': float(x_mean.mean() / x_l2.mean()),
            })

    print(f"Trials: {args.n_trials}, batch={args.batch}, length={args.length}, "
          f"d_inner={args.d_inner}, d_state={args.d_state}\n")
    print(f"  Spearman rank corr (l2 vs mean):  "
          f"mean={np.mean([r['spearman'] for r in rows]):.4f}  "
          f"min={np.min([r['spearman'] for r in rows]):.4f}  "
          f"max={np.max([r['spearman'] for r in rows]):.4f}")
    print(f"  Pearson corr (l2 vs mean):        "
          f"mean={np.mean([r['pearson'] for r in rows]):.4f}  "
          f"min={np.min([r['pearson'] for r in rows]):.4f}")
    print(f"  Top-10% positional overlap:       "
          f"mean={np.mean([r['top10_overlap'] for r in rows]):.4f}  "
          f"min={np.min([r['top10_overlap'] for r in rows]):.4f}")
    print(f"  Score ratio mean/L2:              "
          f"mean={np.mean([r['ratio_mean_over_l2'] for r in rows]):.4f}  "
          f"theoretical bound 1/sqrt(D)={1/np.sqrt(args.d_inner):.4f}")
    print()
    print("Interpretation: Spearman ~1 means rankings are nearly identical; "
          "lower values mean the patched aggregation re-orders positions.")


if __name__ == '__main__':
    main()
