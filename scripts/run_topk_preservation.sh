#!/usr/bin/env bash
#
# run_topk_preservation.sh
# ------------------------
# Phase 8 — Top-K mask-preservation faithfulness across every (dataset,
# seed) cell that already has a trained checkpoint. Reuses the existing
# multiseed/ checkpoint layout. No retraining; ~10-15 min per cell.
#
# Run from xvmamba/ with MambaMedEnv conda env active. Idempotent.
#
# Usage:
#     ./scripts/run_topk_preservation.sh
#
# Override knobs:
#     SEEDS="42 137 2024"
#     DATASETS="bloodmnist dermamnist octmnist pneumoniamnist cifar100 bloodmnist_vim fashionmnist eurosat"
#     NUM_SAMPLES=50
#     OUT_BASE=./results/topk
#
# Estimated runtime: ~10 min per (dataset, seed) × 24 cells (8 datasets × 3 seeds) ≈ 4 hours.
# Only datasets with existing checkpoints are evaluated; missing cells are skipped.

set -euo pipefail

SEEDS="${SEEDS:-42 137 2024}"
DATASETS="${DATASETS:-bloodmnist dermamnist octmnist pneumoniamnist cifar100 bloodmnist_vim fashionmnist eurosat}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda}"
CKPT_BASE="${CKPT_BASE:-./checkpoints/multiseed}"
OUT_BASE="${OUT_BASE:-./results/topk}"

if [[ ! -f "evaluation/topk_preservation.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Datasets: ${DATASETS}    Seeds: ${SEEDS}    N=${NUM_SAMPLES}"

for seed in ${SEEDS}; do
    for ds in ${DATASETS}; do
        # If the ckpt subdir is "bloodmnist_vim" we still need to tell
        # comprehensive_evaluation the underlying dataset is "bloodmnist".
        if [[ "${ds}" == "bloodmnist_vim" ]]; then
            ds_arg="bloodmnist"
        else
            ds_arg="${ds}"
        fi
        ckpt="${CKPT_BASE}/seed_${seed}/${ds}/best_model.pth"
        if [[ ! -f "${ckpt}" ]]; then
            echo "[skip] ${ds} seed ${seed}: ckpt missing"; continue
        fi
        out="${OUT_BASE}/seed_${seed}/${ds}"
        if [[ -f "${out}/topk_preservation.json" ]]; then
            echo "[skip] ${ds} seed ${seed}: already done"; continue
        fi
        mkdir -p "${out}"
        echo
        echo "============================================================"
        echo "  TOPK  ${ds}  seed=${seed}  ckpt=${ckpt}"
        echo "============================================================"
        python3 evaluation/topk_preservation.py \
            --checkpoint "${ckpt}" \
            --dataset "${ds_arg}" \
            --num_samples "${NUM_SAMPLES}" \
            --output_dir "${out}" \
            --device "${DEVICE}" \
            2>&1 | tee "${out}/run.log"
    done
done

# Cross-(dataset, seed) aggregation
echo
echo "============================================================"
echo "  Cross-dataset Top-K preservation summary (mean ± std across seeds)"
echo "============================================================"
python3 - <<PYAGG
"""Aggregate Top-K preservation across seeds for each dataset and method.
Print a compact comparison table that says, for each dataset, whether
controllability or Grad-CAM has the higher preservation AUC."""
import json, glob
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("${OUT_BASE}")
METHODS = ["Jacobian", "Gramian", "Grad-CAM", "Random"]

datasets = sorted({Path(f).parts[-2] for f in glob.glob(str(ROOT / "seed_*/*/topk_preservation.json"))})
print(f"discovered: {datasets}")

print()
print(f"{'dataset':<18s}  {'method':<10s}  {'AUC mean ± std':<22s}  per-K means (5/10/20/30/50%)")
print("-" * 105)
results = {}
for ds in datasets:
    results[ds] = {}
    per_method = {m: [] for m in METHODS}
    per_method_perK = {m: [] for m in METHODS}
    for seed_file in sorted(ROOT.glob(f"seed_*/{ds}/topk_preservation.json")):
        d = json.loads(seed_file.read_text())
        for m in METHODS:
            if m in d["methods"]:
                per_method[m].append(d["methods"][m]["preservation_AUC"])
                per_method_perK[m].append(d["methods"][m]["preservation_per_K"])
    for m in METHODS:
        aucs = per_method[m]
        if not aucs: continue
        mu = mean(aucs); sd = pstdev(aucs) if len(aucs) > 1 else 0.0
        # mean per-K across seeds
        if per_method_perK[m]:
            stacked = list(zip(*per_method_perK[m]))
            per_K_mean = [mean(col) for col in stacked]
            per_K_s = " ".join(f"{v:.3f}" for v in per_K_mean)
        else:
            per_K_s = "—"
        results[ds][m] = {"mean": mu, "std": sd, "per_K_mean": per_K_mean if per_method_perK[m] else None}
        print(f"{ds:<18s}  {m:<10s}  {mu:.3f} ± {sd:.3f}{'':6s}  {per_K_s}")

# Headline comparison: per-dataset, does J or GC win on AUC?
print()
print("=== Headline: which method has higher Top-K preservation AUC? ===")
for ds in datasets:
    if "Jacobian" in results[ds] and "Grad-CAM" in results[ds]:
        J = results[ds]["Jacobian"]["mean"]
        G = results[ds]["Grad-CAM"]["mean"]
        winner = "Jacobian" if J > G else ("Grad-CAM" if G > J else "tie")
        gap = J - G
        print(f"  {ds:<18s}  J={J:.3f}  GC={G:.3f}  gap={gap:+.3f}  winner: {winner}")

# Save full summary
summary_path = ROOT / "topk_summary.json"
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nwrote {summary_path}")
PYAGG

echo
echo "Done."
echo "  Per-cell outputs:  ${OUT_BASE}/seed_<seed>/<dataset>/topk_preservation.json"
echo "  Aggregate:         ${OUT_BASE}/topk_summary.json"
