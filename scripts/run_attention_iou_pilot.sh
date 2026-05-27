#!/usr/bin/env bash
#
# run_attention_iou_pilot.sh
# --------------------------
# Phase 9 PILOT — Test the Internal-Attention IoU metric on three
# representative datasets (BloodMNIST where Jacobian wins on existing
# metrics, DermaMNIST where Grad-CAM marginally wins, OCTMNIST where
# Grad-CAM clearly wins) using only seed 42 first. ~30 minutes total.
#
# All four channel-aggregation variants (mean, L2, max, classifier-
# weighted) are computed per image. We will then read the headline
# table and decide which variant to commit to before sweeping the
# remaining 24 cells.
#
# Usage:
#     cd xvmamba
#     ./scripts/run_attention_iou_pilot.sh
#
# Override:
#     DATASETS="bloodmnist dermamnist octmnist"   # default pilot set
#     SEEDS="42"                                  # default 1 seed for pilot
#     NUM_SAMPLES=50

set -euo pipefail

SEEDS="${SEEDS:-42}"
DATASETS="${DATASETS:-bloodmnist dermamnist octmnist}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda}"
CKPT_BASE="${CKPT_BASE:-./checkpoints/multiseed}"
OUT_BASE="${OUT_BASE:-./results/attn_iou}"

if [[ ! -f "evaluation/attention_iou.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "PILOT — datasets=${DATASETS}  seeds=${SEEDS}  N=${NUM_SAMPLES}"

for seed in ${SEEDS}; do
    for ds in ${DATASETS}; do
        # The ckpt subdir is "bloodmnist_vim" but the underlying
        # DatasetType is "bloodmnist" (Vim is just a different architecture
        # trained on the same dataset).
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
        if [[ -f "${out}/attention_iou.json" ]]; then
            echo "[skip] ${ds} seed ${seed}: already done"; continue
        fi
        mkdir -p "${out}"
        echo
        echo "============================================================"
        echo "  ATTN-IoU  ${ds}  seed=${seed}"
        echo "============================================================"
        python3 evaluation/attention_iou.py \
            --checkpoint "${ckpt}" \
            --dataset "${ds_arg}" \
            --num_samples "${NUM_SAMPLES}" \
            --output_dir "${out}" \
            --device "${DEVICE}" \
            2>&1 | tee "${out}/run.log"
    done
done

echo
echo "============================================================"
echo "  PILOT SUMMARY — across-dataset comparison per variant"
echo "============================================================"
python3 - <<PYAGG
"""Aggregate the pilot results and report which channel-aggregation
variant gives Jacobian the largest IoU advantage over Grad-CAM."""
import json, glob
from pathlib import Path

ROOT = Path("${OUT_BASE}")
files = sorted(glob.glob(str(ROOT / "seed_*/*/attention_iou.json")))
if not files:
    print("(no results found)"); raise SystemExit
datasets = sorted({Path(f).parts[-2] for f in files})

variants = ["mean", "l2", "max", "cw"]
K_values = [10, 25, 50]
methods  = ["Jacobian", "Gramian", "Grad-CAM", "Random"]

print()
print(f"{'variant':<8s}  {'K':<5s}  {'metric':<22s}  {'  '.join(f'{ds:<14s}' for ds in datasets)}")
print("-" * (40 + len(datasets) * 16))

# For each variant and K, report J's IoU, GC's IoU, and (J - GC) for each dataset
for var in variants:
    for K in K_values:
        for m in ("Jacobian", "Grad-CAM"):
            row = [f"K={K}%/{m:<8s}"]
            for ds in datasets:
                # collect across all seed_* files for this ds
                vals = []
                for f in glob.glob(str(ROOT / f"seed_*/{ds}/attention_iou.json")):
                    d = json.loads(Path(f).read_text())
                    key = f"K{K}_{var}_{m}"
                    if key in d["iou_mean"]:
                        vals.append(d["iou_mean"][key])
                if vals:
                    row.append(f"{sum(vals)/len(vals):<14.3f}")
                else:
                    row.append(f"{'n/a':<14s}")
            print(f"{var:<8s}  {K:<5d}  {row[0]:<22s}  {'  '.join(row[1:])}")
        # J - GC delta line
        deltas = []
        for ds in datasets:
            vals_J  = []
            vals_GC = []
            for f in glob.glob(str(ROOT / f"seed_*/{ds}/attention_iou.json")):
                d = json.loads(Path(f).read_text())
                kJ  = f"K{K}_{var}_Jacobian"
                kGC = f"K{K}_{var}_Grad-CAM"
                if kJ in d["iou_mean"]:  vals_J.append(d["iou_mean"][kJ])
                if kGC in d["iou_mean"]: vals_GC.append(d["iou_mean"][kGC])
            if vals_J and vals_GC:
                deltas.append(f"{(sum(vals_J)/len(vals_J)) - (sum(vals_GC)/len(vals_GC)):+.3f}".ljust(14))
            else:
                deltas.append(f"{'n/a':<14s}")
        print(f"{var:<8s}  {K:<5d}  {'(J − GC) delta':<22s}  {'  '.join(deltas)}")
        print()

# Pick the variant with largest mean (J - GC) delta across datasets,
# averaged over K
print("=" * 70)
print("Average (J − GC) IoU advantage, per variant (avg over K + datasets):")
print("=" * 70)
import statistics
for var in variants:
    deltas = []
    for K in K_values:
        for ds in datasets:
            J_vals  = []
            GC_vals = []
            for f in glob.glob(str(ROOT / f"seed_*/{ds}/attention_iou.json")):
                d = json.loads(Path(f).read_text())
                kJ  = f"K{K}_{var}_Jacobian"
                kGC = f"K{K}_{var}_Grad-CAM"
                if kJ in d["iou_mean"]:  J_vals.append(d["iou_mean"][kJ])
                if kGC in d["iou_mean"]: GC_vals.append(d["iou_mean"][kGC])
            if J_vals and GC_vals:
                deltas.append((sum(J_vals)/len(J_vals)) - (sum(GC_vals)/len(GC_vals)))
    if deltas:
        avg = sum(deltas)/len(deltas)
        winner = "J" if avg > 0 else "GC"
        print(f"  {var:<8s}: avg (J − GC) = {avg:+.3f}    winner: {winner}")
PYAGG

echo
echo "Pilot done. Per-cell JSON in ${OUT_BASE}/seed_<seed>/<dataset>/attention_iou.json"
echo "If a variant looks promising, sweep all 24 cells with:"
echo "  DATASETS='bloodmnist dermamnist octmnist pneumoniamnist cifar100 bloodmnist_vim fashionmnist eurosat' \\"
echo "  SEEDS='42 137 2024' ./scripts/run_attention_iou_pilot.sh"
