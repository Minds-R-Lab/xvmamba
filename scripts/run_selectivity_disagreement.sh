#!/usr/bin/env bash
#
# Phase 6 runner — selectivity / structural-attributive disagreement.
#
# Runs the same pipeline (selectivity_disagreement.py) over multiple
# (checkpoint, dataset) combinations, then prints a cross-dataset
# summary table comparing H1 (Δ_k) and H2 (D(x) AUROC) results.
#
# Usage:  cd xvmamba && ./scripts/run_selectivity_disagreement.sh
#
# Override defaults via env vars:
#     DATASETS="bloodmnist dermamnist"
#     SEEDS="42 137 2024"            # uses multi-seed checkpoints if present
#     NUM_SAMPLES=200                # number of test images per (ds, seed)
#     OUT_BASE=./results/exploratory
#
# Estimated runtime on RTX 3090:
#     ~5–8 min per (dataset, seed) for N=200 images (no retraining).
#     Total for 2 datasets × 3 seeds ≈ 30–50 minutes.

set -euo pipefail

DATASETS="${DATASETS:-bloodmnist dermamnist}"
SEEDS="${SEEDS:-42 137 2024}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
OUT_BASE="${OUT_BASE:-./results/exploratory}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "evaluation/selectivity_disagreement.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Datasets: ${DATASETS}    Seeds: ${SEEDS}    N=${NUM_SAMPLES}"
echo

for seed in ${SEEDS}; do
    for ds in ${DATASETS}; do
        # Prefer the multi-seed checkpoint if it exists; otherwise the
        # single-seed default that the headline evaluation uses.
        for ckpt_candidate in \
            "./checkpoints/multiseed/seed_${seed}/${ds}/best_model.pth" \
            "./checkpoints/${ds}/best_model.pth"; do
            if [[ -f "${ckpt_candidate}" ]]; then
                ckpt="${ckpt_candidate}"
                break
            fi
        done
        if [[ -z "${ckpt:-}" ]]; then
            echo "[skip] no checkpoint found for ${ds} seed ${seed}" >&2
            continue
        fi

        out="${OUT_BASE}/seed_${seed}/${ds}"
        if [[ -f "${out}/summary.json" ]]; then
            echo "[skip] ${out}/summary.json already exists"
            continue
        fi
        mkdir -p "${out}"
        echo
        echo "============================================================"
        echo "  ${ds}  seed=${seed}  ckpt=${ckpt}"
        echo "============================================================"
        python3 evaluation/selectivity_disagreement.py \
            --checkpoint "${ckpt}" \
            --dataset "${ds}" \
            --num_samples "${NUM_SAMPLES}" \
            --output_dir "${out}" \
            --device "${DEVICE}" \
            2>&1 | tee "${out}/run.log"
        unset ckpt
    done
done

# Cross-dataset / cross-seed summary
echo
echo "============================================================"
echo "  Cross-dataset summary (H1 mean Δ, H2 AUROC)"
echo "============================================================"
python3 - <<PYAGG
from pathlib import Path
import json, glob
rows = []
for f in sorted(glob.glob("${OUT_BASE}/seed_*/*/summary.json")):
    d = json.loads(Path(f).read_text())
    rows.append({
        "seed": Path(f).parts[-3].replace("seed_", ""),
        "dataset": d.get("dataset"),
        "n":   d.get("num_samples_actual"),
        "acc": d.get("accuracy"),
        "delta_mean": d.get("H1", {}).get("mean_delta"),
        "delta_gini": d.get("H1", {}).get("mean_delta_gini"),
        "corr_J_GC": d.get("H2", {}).get("mean_corr_J_GC"),
        "D_mean":    d.get("H2", {}).get("mean_D"),
        "D_corr_unc":d.get("H2", {}).get("D_corr_with_uncertainty"),
        "AUROC_D":   d.get("H2", {}).get("AUROC_D_predicts_misclass"),
        "AUROC_b":   d.get("H2", {}).get("AUROC_baseline_1mC"),
    })
if not rows:
    print("(no summary.json files found)")
else:
    hdr = ["seed","dataset","n","acc","delta_mean","delta_gini","corr_J_GC","D_mean","D_corr_unc","AUROC_D","AUROC_b"]
    width = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in hdr}
    line = "  ".join(h.ljust(width[h]) for h in hdr)
    print(line); print("-" * len(line))
    for r in rows:
        def fmt(v):
            if v is None: return "n/a"
            if isinstance(v, float): return f"{v:.4f}"
            return str(v)
        print("  ".join(fmt(r[h]).ljust(width[h]) for h in hdr))
PYAGG

echo
echo "Done. Per-(seed, dataset) outputs in ${OUT_BASE}/seed_<seed>/<dataset>/"
