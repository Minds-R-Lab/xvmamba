#!/usr/bin/env bash
#
# run_extra_baselines.sh
# ----------------------
# Evaluate Score-CAM, Integrated Gradients, and RISE on the existing
# checkpoints (BloodMNIST, DermaMNIST, CIFAR-100). This is the Phase 2.4
# delivery requested by Reviewers 2, 3, 4.
#
# Each baseline is eval-only — no training required. The script reuses
# the existing FaithfulnessEvaluator from comprehensive_evaluation.py for
# the deletion/insertion AUC computation, so the new numbers slot
# directly into Table V of the manuscript without further post-processing.
#
# Usage:
#     ./scripts/run_extra_baselines.sh
#     DATASETS="bloodmnist" ./scripts/run_extra_baselines.sh    # one dataset
#     RISE_NUM_MASKS=2000 ./scripts/run_extra_baselines.sh      # higher-quality RISE
#
# Runtime guidance (RTX 3090, num_samples=50, default knobs):
#   Score-CAM (top_k=32)  ~ 30 channels * 50 images = 1500 forward passes
#   Integrated Gradients  ~ 20 steps  * 50 images = 1000 forward+backward passes
#   RISE (num_masks=500)  ~ 500 masks * 50 images = 25 000 forward passes (batched)
# Total: ~30-60 min per dataset.

set -euo pipefail

# ----- config (env-overridable) ---------------------------------------------

DATASETS="${DATASETS:-bloodmnist dermamnist}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_TEST_CLASSES="${NUM_TEST_CLASSES:-5}"
SCORE_CAM_TOP_K="${SCORE_CAM_TOP_K:-32}"
IG_STEPS="${IG_STEPS:-20}"
RISE_NUM_MASKS="${RISE_NUM_MASKS:-500}"
RISE_BATCH_SIZE="${RISE_BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/extra_baselines}"

# ----- sanity checks --------------------------------------------------------

if [[ ! -f "evaluation/comprehensive_evaluation.py" ]]; then
    echo "error: must be run from xvmamba/ directory" >&2
    exit 1
fi
if [[ ! -f "evaluation/extra_baselines.py" ]]; then
    echo "error: evaluation/extra_baselines.py missing — Phase 2.4 not deployed?" >&2
    exit 1
fi
if [[ ! -f "evaluation/run_extra_baselines.py" ]]; then
    echo "error: evaluation/run_extra_baselines.py missing" >&2
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

# ----- iterate over datasets -----------------------------------------------

for ds in ${DATASETS}; do
    ckpt="checkpoints/${ds}/best_model.pth"
    if [[ ! -f "${ckpt}" ]]; then
        # Try cifar100 variant.
        ckpt="checkpoints/${ds}/best_model.pth"
    fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "skipping ${ds}: checkpoint not found at ${ckpt}" >&2
        continue
    fi

    out="${OUTPUT_BASE}/${ds}"
    mkdir -p "${out}"

    echo
    echo "========================================================================"
    echo "Extra baselines on ${ds}"
    echo "  num_samples=${NUM_SAMPLES}  Score-CAM top_k=${SCORE_CAM_TOP_K}"
    echo "  IG steps=${IG_STEPS}  RISE masks=${RISE_NUM_MASKS}"
    echo "========================================================================"

    python3 evaluation/run_extra_baselines.py \
        --checkpoint "${ckpt}" \
        --dataset "${ds}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_test_classes "${NUM_TEST_CLASSES}" \
        --score_cam_top_k "${SCORE_CAM_TOP_K}" \
        --ig_steps "${IG_STEPS}" \
        --rise_num_masks "${RISE_NUM_MASKS}" \
        --rise_batch_size "${RISE_BATCH_SIZE}" \
        --output_dir "${out}" \
        --device "${DEVICE}" \
        2>&1 | tee "${out}/run.log"
done

echo
echo "Done. Per-dataset JSON in ${OUTPUT_BASE}/<dataset>/extra_baselines.json"
