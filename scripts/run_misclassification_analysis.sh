#!/usr/bin/env bash
#
# run_misclassification_analysis.sh
# ---------------------------------
# Phase 3.5 (R4 Q3). Stratifies the test set by correct vs incorrect
# prediction and reports per-stratum faithfulness, confidence, and the
# overlap between controllability and Grad-CAM maps.
#
# Uses DermaMNIST and CIFAR-100 by default because they have non-trivial
# misclassification rates (77% and 55% accuracy respectively). BloodMNIST
# at 99% accuracy would have too few wrong predictions to be informative.
#
# Usage:
#     ./scripts/run_misclassification_analysis.sh
#     NUM_SAMPLES=500 ./scripts/run_misclassification_analysis.sh    # for tighter stats

set -euo pipefail

DATASETS="${DATASETS:-dermamnist cifar100}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/misclassification}"

if [[ ! -f "evaluation/run_misclassification_analysis.py" ]]; then
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
    echo "Misclassification stratified analysis on ${ds}  (N=${NUM_SAMPLES})"
    echo "========================================================================"

    python3 evaluation/run_misclassification_analysis.py \
        --checkpoint "${ckpt}" \
        --dataset "${ds}" \
        --num_samples "${NUM_SAMPLES}" \
        --output_dir "${out}" \
        --device "${DEVICE}" \
        2>&1 | tee "${out}/run.log"
done

echo
echo "Done. Per-dataset JSON in ${OUTPUT_BASE}/<dataset>/misclassification_analysis.json"
