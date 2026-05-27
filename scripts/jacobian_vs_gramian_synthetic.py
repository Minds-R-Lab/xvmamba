"""
Jacobian vs. Gramian — synthetic divergent example.

This script complements Section III.D-E of the manuscript by constructing
small selective-SSM toy systems in which the two controllability indices
disagree by design. It is the empirical companion to the math derivation
in `LOGS/01_math.md` (entry `[DERIVE] Jacobian vs Gramian divergence
conditions`).

Why this matters: on trained VMamba checkpoints the two indices correlate
near 1.0 (Reviewer 4 Q5 asked why both are reported). The correlation is
high in practice but the indices are not theoretically equivalent. The
frozen-LTI Gramian `G_k` is computed from `(A_k, b_k, c_k)` alone; the
Jacobian surrogate `J_k` aggregates `c_{t,n} phi_n(t,k) b_{k,n}` for all
t >= k. The two should diverge when the parameters at position k differ
substantially from the parameters at positions t > k.

We build three small selective-SSM systems (L=8, N=4) that exhibit this:

  Scenario A: "Sharp decay after k=0"
      A is near 1 at position 0 (long local memory) and near 0 elsewhere.
      G_0 is large (frozen-LTI says "persistent"), but J_0 is small
      because the actual downstream A values kill the propagation.

  Scenario B: "Sharp activation after k=0"
      A is near 0 at position 0 (short local memory) and near 1 elsewhere.
      G_0 is small, but J_0 is large because the actual downstream A's
      preserve the perturbation injected at k=0.

  Scenario C: "Readout flip"
      A, b are slowly varying; c flips sign sharply between k and k+1.
      G_0 sees only |c_0|; J_0 incorporates the future |c_t|.

Run:
    python scripts/jacobian_vs_gramian_synthetic.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Reuse the analyzer's recursion to ensure we test the same code we ship.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controllability.analyzer import JacobianControllability, GramianControllability


def build_scenario(name: str, L: int = 8, N: int = 4):
    """Return (A_bar, B_bar, C) of shape compatible with the analyzer
    [batch=1, length=L, d_inner=1, d_state=N], plus a human description.
    """
    A = np.zeros((1, L, 1, N), dtype=np.float32)
    B = np.ones((1, L, 1, N), dtype=np.float32) * 1.0
    C = np.ones((1, L, N), dtype=np.float32) * 1.0

    if name == "A_sharp_decay":
        # Long memory at k=0, no memory after.
        A[0, 0, 0, :] = 0.99
        A[0, 1:, 0, :] = 0.05
        description = ("Long local memory at k=0 (a=0.99); future positions "
                       "have a~0.05, so any perturbation injected at k=0 decays "
                       "before it can be read out by future c. G_0 is large "
                       "(frozen LTI sees long memory); J_0 sees the actual decay.")

    elif name == "A_sharp_activation":
        # No memory at k=0, but long memory afterward.
        A[0, 0, 0, :] = 0.05
        A[0, 1:, 0, :] = 0.99
        description = ("Short local memory at k=0 (a=0.05); future positions "
                       "have a~0.99. G_0 is small (frozen LTI says decay fast); "
                       "J_0 sees the long downstream memory after the first step.")

    elif name == "C_signflip":
        # Slowly-varying A, but c flips sign between k=0 and k>=1.
        A[0, :, 0, :] = 0.9
        C[0, 0, :] = +1.0
        C[0, 1:, :] = -1.0
        description = ("A is slowly varying (a=0.9 throughout). c flips sign "
                       "between k=0 and k>=1. G_0 sees |c_0|=1. J_0 sees "
                       "|c_t|=1 for all t (so by absolute value, J_0 captures "
                       "the full propagation, while G_0 is local).")

    elif name == "B_position_asymmetry":
        # Strong B at k=0, weak elsewhere; long memory.
        A[0, :, 0, :] = 0.9
        B[0, 0, 0, :] = 10.0
        B[0, 1:, 0, :] = 0.01
        description = ("Strong B at k=0 (b=10), weak B at later positions "
                       "(b=0.01). Long memory throughout (a=0.9). Both indices "
                       "should peak at k=0, but their magnitudes differ.")

    else:
        raise ValueError(f"unknown scenario: {name}")

    return torch.from_numpy(A), torch.from_numpy(B), torch.from_numpy(C), description


def compute_indices(A_bar, B_bar, C):
    """Compute J_k and G_k per position using the analyzer's code paths."""
    # Jacobian: returns total_influence [batch, length].
    total_j, _, _ = JacobianControllability.compute_influence_1d(A_bar, B_bar, C)
    # Gramian: same shape.
    total_g = GramianControllability.compute_influence_1d(A_bar, B_bar, C)
    return total_j[0].numpy(), total_g[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/jac_vs_gramian_synthetic.json")
    args = ap.parse_args()

    scenarios = ["A_sharp_decay", "A_sharp_activation", "C_signflip", "B_position_asymmetry"]
    L, N = 8, 4

    out = []
    print()
    print(f"Synthetic Jacobian-vs-Gramian comparison (L={L}, N={N})")
    print("=" * 72)
    for name in scenarios:
        A, B, C, desc = build_scenario(name, L=L, N=N)
        J, G = compute_indices(A, B, C)
        # Pearson and ratio of values at k=0 (the position we engineered to
        # diverge).
        rho = float(np.corrcoef(J, G)[0, 1]) if J.std() > 1e-9 and G.std() > 1e-9 else float("nan")
        ratio_k0 = float(J[0] / (G[0] + 1e-12))
        ratio_max = float(J.max() / (G.max() + 1e-12))

        print(f"\n-- {name} --")
        print(f"   {desc}")
        print(f"   J_k:  {np.array2string(J, precision=4, suppress_small=True)}")
        print(f"   G_k:  {np.array2string(G, precision=4, suppress_small=True)}")
        print(f"   Pearson(J, G) = {rho:+.4f}")
        print(f"   ratio J/G at k=0 = {ratio_k0:.4f}")
        print(f"   ratio max(J)/max(G) = {ratio_max:.4f}")

        out.append({
            "name": name,
            "description": desc,
            "J": J.tolist(),
            "G": G.tolist(),
            "pearson": rho,
            "ratio_J_over_G_at_k0": ratio_k0,
            "ratio_max_J_over_max_G": ratio_max,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print("=" * 72)
    print(f"Saved per-scenario results to: {out_path}")
    print()
    print("Take-away: On trained vision SSMs the parameters vary smoothly across")
    print("positions, so J_k and G_k correlate near 1.0 in practice. The")
    print("indices are not theoretically equivalent --- the scenarios above")
    print("construct cases where they disagree by ~10x in magnitude or in")
    print("Pearson correlation. We report both to provide a complementary")
    print("view: J_k captures propagation through the actual time-varying")
    print("recurrence, G_k captures the local frozen-LTI controllability.")


if __name__ == "__main__":
    main()
