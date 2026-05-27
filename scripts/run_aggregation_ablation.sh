#!/usr/bin/env bash
#
# run_aggregation_ablation.sh
# ---------------------------
# Phase 3.1 (R3 #1) -- aggregation ablation.
# Phase 3.2 (R4 Q4) -- per-direction consistency.
#
# Both are eval-only on existing checkpoints (no training required).
#
# Usage:
#     ./scripts/run_aggregation_ablation.sh
#     DATASETS=bloodmnist ./scripts/run_aggregation_ablation.sh

set -euo pipefail

DATASETS="${DATASETS:-bloodmnist dermamnist}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda}"
METHOD="${METHOD:-jacobian}"        # jacobian or gramian
OUTPUT_BASE="${OUTPUT_BASE:-./results/aggregation_ablation}"

if [[ ! -f "evaluation/run_aggregation_ablation.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

for ds in ${DATASETS}; do
    ckpt="checkpoints/${ds}/best_model.pth"
    if [[ ! -f "${ckpt}" ]]; then
        echo "skipping ${ds}: checkpoint not found" >&2
        continue
    fi
    out="${OUTPUT_BASE}/${ds}"
    mkdir -p "${out}"

    echo
    echo "========================================================================"
    echo "Aggregation ablation on ${ds}  (method=${METHOD}, N=${NUM_SAMPLES})"
    echo "========================================================================"

    python3 evaluation/run_aggregation_ablation.py \
        --checkpoint "${ckpt}" \
        --dataset "${ds}" \
        --num_samples "${NUM_SAMPLES}" \
        --method "${METHOD}" \
        --output_dir "${out}" \
        --device "${DEVICE}" \
        2>&1 | tee "${out}/run.log"
done

echo
echo "Done. Per-dataset JSON in ${OUTPUT_BASE}/<dataset>/aggregation_ablation.json"
