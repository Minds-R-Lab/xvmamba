#!/usr/bin/env bash
#
# run_vim_training.sh
# -------------------
# Train Vim-Tiny on one MedMNIST dataset + run comprehensive evaluation.
# Vim is the second VSSM variant required by Reviewers 1, 2, 3.
#
# Vim (Zhu et al. 2024) differs from VMamba along three axes:
#   1. bidirectional 1D scan instead of 4-direction cross-scan
#   2. plain (non-hierarchical) architecture — one resolution throughout
#   3. ViT-style patch embedding (default patch_size=16)
#
# The controllability analyzer requires no changes: each Vim block exposes
# an SS2DCache with `fwd_h` + `bwd_h` directions, matching the interface
# used for VMamba.
#
# Run from the xvmamba/ directory with MambaMedEnv conda env active.
#
# Usage examples:
#     # train + eval on bloodmnist (default)
#     ./scripts/run_vim_training.sh
#
#     # train on dermamnist with 24 blocks (matches paper Vim-Ti)
#     DATASET=dermamnist VIM_DEPTH=24 ./scripts/run_vim_training.sh
#
#     # quick smoke test (4 blocks, 5 epochs)
#     EPOCHS=5 VIM_DEPTH=4 ./scripts/run_vim_training.sh

set -euo pipefail

# ----- config (env-overridable) ---------------------------------------------

DATASET="${DATASET:-bloodmnist}"            # primary medical dataset (others: dermamnist, octmnist, pneumoniamnist, cifar100)
EPOCHS="${EPOCHS:-50}"                       # matches VMamba training protocol
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
VIM_DEPTH="${VIM_DEPTH:-12}"                 # paper Vim-Ti uses 24; 12 fits faster
VIM_D_MODEL="${VIM_D_MODEL:-192}"            # paper Vim-Ti d_model
VIM_MLP_RATIO="${VIM_MLP_RATIO:-4.0}"        # 0 disables MLP (pure Vim block)
PATCH_SIZE="${PATCH_SIZE:-16}"               # ViT/Vim-style large patches (override for medical fine detail)
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-./checkpoints}"
EVAL_OUTPUT_BASE="${EVAL_OUTPUT_BASE:-./results}"

# ----- sanity checks --------------------------------------------------------

if [[ ! -f "scripts/train.py" ]]; then
    echo "error: must be run from xvmamba/ directory" >&2
    exit 1
fi

# Confirm Vim wiring is in place.
for k in 'model_arch' 'vim_depth' 'VimClassifier'; do
    if ! grep -q "$k" scripts/train.py; then
        echo "error: scripts/train.py missing '$k' — Vim wiring not applied?" >&2
        exit 1
    fi
done

if [[ ! -f "models/vim_classifier.py" || ! -f "models/vim_blocks.py" ]]; then
    echo "error: models/vim_classifier.py or models/vim_blocks.py missing" >&2
    exit 1
fi

if grep -q "torch.norm(CB, dim=-1)" controllability/analyzer.py; then
    echo "error: analyzer.py still uses torch.norm; aggregation patch not applied?" >&2
    exit 1
fi

mkdir -p logs

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

# ----- quick model sanity test before kicking off long training -------------

echo
echo "========================================================================"
echo "Vim model sanity check (small forward pass + analyzer + train step)"
echo "========================================================================"

# Note: mamba-ssm's CUDA kernels require GPU tensors. We run the sanity
# test on GPU when available; otherwise on CPU using the slow-Python
# SelectiveSSM fallback. Without this, mamba-ssm raises
#   "Expected x.is_cuda() to be true, but got false"
# from causal_conv1d_fwd.
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
import torch
from models.vim_classifier import create_vim_tiny
from controllability.analyzer import ControllabilityAnalyzer

# Use CUDA if available — required when mamba-ssm CUDA kernels are installed.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Sanity-test device: {device}")

m = create_vim_tiny(num_classes=10, image_size=224, depth=4).to(device)
x = torch.randn(2, 3, 224, 224, device=device)

# Forward pass.
y = m(x)
assert y.shape == (2, 10), f"unexpected logits shape: {y.shape}"

# Backward step (catches any autograd issues with the bidirectional reverse).
loss = y.sum()
loss.backward()
assert m.head.weight.grad is not None, "head gradient is None — backward path broken"

# Analyzer.
m.zero_grad()
m.enable_analysis_mode()
with torch.no_grad():
    logits, analysis = m(x, return_analysis=True)
assert len(analysis.stage_caches) == 1, "Vim should report 1 stage"
assert len(analysis.stage_caches[0]) == 4, "expected 4 blocks (depth=4 test)"
cache0 = analysis.stage_caches[0][0]
assert len(cache0.direction_caches) == 2, "expected fwd_h + bwd_h directions"
analyzer = ControllabilityAnalyzer(method='jacobian')
result = analyzer.analyze(analysis)
print(f"  logits ok {tuple(logits.shape)}; "
      f"controllability map {tuple(result.aggregated_map.shape)}; "
      f"params={m.get_num_params():,}")
print("  All Vim sanity checks passed.")
PY

# ----- train ---------------------------------------------------------------

echo
echo "========================================================================"
echo "Training Vim on ${DATASET}"
echo "  epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE}"
echo "  vim_depth=${VIM_DEPTH} d_model=${VIM_D_MODEL} mlp_ratio=${VIM_MLP_RATIO}"
echo "  patch_size=${PATCH_SIZE} image_size=${IMAGE_SIZE}"
echo "========================================================================"

# Vim training writes to checkpoints/${DATASET}/; we redirect to a Vim-specific
# subdir so it doesn't collide with the existing VMamba checkpoint.
VIM_OUTPUT_DIR="${OUTPUT_BASE}_vim"
mkdir -p "${VIM_OUTPUT_DIR}"

python3 scripts/train.py \
    --dataset "${DATASET}" \
    --model_arch vim \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --vim_depth "${VIM_DEPTH}" \
    --vim_d_model "${VIM_D_MODEL}" \
    --vim_mlp_ratio "${VIM_MLP_RATIO}" \
    --patch_size "${PATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --data_root "${DATA_ROOT}" \
    --device "${DEVICE}" \
    --output_dir "${VIM_OUTPUT_DIR}" \
    2>&1 | tee "logs/vim_${DATASET}_train.log"

# ----- evaluate ------------------------------------------------------------

CKPT="${VIM_OUTPUT_DIR}/${DATASET}/best_model.pth"
if [[ ! -f "${CKPT}" ]]; then
    echo "warning: expected checkpoint not found at ${CKPT}; skipping eval" >&2
    exit 0
fi

EVAL_OUT="${EVAL_OUTPUT_BASE}/vim_${DATASET}"
mkdir -p "${EVAL_OUT}"

echo
echo "========================================================================"
echo "Running comprehensive evaluation on Vim/${DATASET}"
echo "========================================================================"

python3 evaluation/comprehensive_evaluation.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET}" \
    --num_samples "${EVAL_NUM_SAMPLES}" \
    --num_test_classes 5 \
    --output_dir "${EVAL_OUT}" \
    --device "${DEVICE}" \
    2>&1 | tee "logs/vim_${DATASET}_eval.log"

echo
echo "Done."
echo "Checkpoint:    ${CKPT}"
echo "Training log:  logs/vim_${DATASET}_train.log"
echo "Eval log:      logs/vim_${DATASET}_eval.log"
echo "Eval outputs:  ${EVAL_OUT}/"
