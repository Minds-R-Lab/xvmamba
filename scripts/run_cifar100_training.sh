#!/usr/bin/env bash
#
# run_cifar100_training.sh
# ------------------------
# Train VMamba-Tiny on CIFAR-100 for the TNNLS revision (non-medical
# benchmark per Reviewers 1, 2, 3). After training, run the comprehensive
# evaluation with the same protocol used for the MedMNIST datasets so the
# new numbers slot directly into the existing tables.
#
# Run from the xvmamba/ directory with the MambaMedEnv conda environment
# active.
#
# Usage:
#     ./scripts/run_cifar100_training.sh
#
# Output:
#     checkpoints/cifar100/best_model.pth     -- trained checkpoint
#     results/cifar100/                       -- evaluation outputs
#     logs/cifar100_train.log                 -- full training log

set -euo pipefail

# ----- config ---------------------------------------------------------------

EPOCHS="${EPOCHS:-50}"           # increase to 100 for a final-quality run
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PATCH_SIZE="${PATCH_SIZE:-4}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda}"

# ----- sanity checks --------------------------------------------------------

if [[ ! -f "scripts/train.py" ]]; then
    echo "error: must be run from xvmamba/ directory" >&2
    exit 1
fi

# Confirm CIFAR-100 wiring is in place (added 2026-05-14).
if ! grep -q "DatasetType.CIFAR100" data/datasets.py; then
    echo "error: data/datasets.py does not have CIFAR-100 wired in" >&2
    echo "  expected: 'DatasetType.CIFAR100' present" >&2
    exit 1
fi

# Confirm patched aggregation is in analyzer.py.
if grep -q "torch.norm(CB, dim=-1)" controllability/analyzer.py; then
    echo "error: analyzer.py still uses torch.norm; patch not applied?" >&2
    exit 1
fi

mkdir -p logs results/cifar100 checkpoints/cifar100

# Confirm GPU.
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

# ----- train ---------------------------------------------------------------

echo
echo "========================================================================"
echo "Training VMamba-Tiny on CIFAR-100"
echo "  epochs: ${EPOCHS}, batch: ${BATCH_SIZE}, lr: ${LEARNING_RATE}"
echo "========================================================================"

# CIFAR-100 is 32x32; we resize to 224x224 in the loader (matches the
# MedMNIST 224x224 protocol used in v1). Batch size 32 fits comfortably
# on an RTX 3090 with mixed precision.
python3 scripts/train.py \
    --dataset cifar100 \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --patch_size "${PATCH_SIZE}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --data_root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    2>&1 | tee logs/cifar100_train.log

echo
echo "Training complete. Checkpoint at: checkpoints/cifar100/best_model.pth"

# ----- evaluate ------------------------------------------------------------

if [[ ! -f "checkpoints/cifar100/best_model.pth" ]]; then
    echo "warning: expected checkpoint not found; skipping eval" >&2
    exit 0
fi

echo
echo "========================================================================"
echo "Running comprehensive evaluation on CIFAR-100"
echo "  num_samples: ${EVAL_NUM_SAMPLES}"
echo "========================================================================"

python3 evaluation/comprehensive_evaluation.py \
    --checkpoint checkpoints/cifar100/best_model.pth \
    --dataset cifar100 \
    --num_samples "${EVAL_NUM_SAMPLES}" \
    --num_test_classes 5 \
    --output_dir results/cifar100 \
    --device "${DEVICE}" \
    2>&1 | tee logs/cifar100_eval.log

echo
echo "Done."
echo "Training log: logs/cifar100_train.log"
echo "Eval log:     logs/cifar100_eval.log"
echo "Eval JSON:    results/cifar100/"
