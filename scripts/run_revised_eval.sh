#!/usr/bin/env bash
#
# run_revised_eval.sh
# -------------------
# Re-run comprehensive_evaluation on existing checkpoints with the 5 audit
# fixes applied (B1: drop Ratio column / add drop_diff; B2: top-K predicted
# classes for cross-class; G1: per-channel baseline; G2: tie-jitter; G3:
# expose orig_conf filter), then run bootstrap CIs on the new per-sample
# results so the manuscript tables can report numbers with 95% CIs.
#
# Usage:
#     ./scripts/run_revised_eval.sh
#     DATASETS="bloodmnist" ./scripts/run_revised_eval.sh
#     N_BOOTSTRAP=2000 ./scripts/run_revised_eval.sh
#
# Estimated runtime on RTX 3090 for the default DATASETS:
#   BloodMNIST (50 samples): ~25 min
#   DermaMNIST (50 samples): ~25 min
#   Bootstrap post-processing: ~5 seconds per dataset

set -euo pipefail

DATASETS="${DATASETS:-bloodmnist dermamnist}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_TEST_CLASSES="${NUM_TEST_CLASSES:-5}"
MIN_ORIG_CONF="${MIN_ORIG_CONF:-0.0}"     # G3 default: no filter
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE="${OUTPUT_BASE:-./results/revised_eval}"

if [[ ! -f "evaluation/comprehensive_evaluation.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi
if [[ ! -f "evaluation/bootstrap_ci.py" ]]; then
    echo "error: evaluation/bootstrap_ci.py missing" >&2; exit 1
fi

# Verify the audit fixes are in the file. Use fixed-string grep (-F) to
# avoid having to escape literal parens / brackets / dots.
for needle in '1e-9 * torch.randn_like' 'image.mean(dim=[2, 3]' 'min_orig_conf' 'test_classes_per_image' "'drop_diff':"; do
    if ! grep -q -F -- "${needle}" evaluation/comprehensive_evaluation.py; then
        echo "error: audit fix marker '${needle}' missing from evaluation/comprehensive_evaluation.py" >&2
        exit 1
    fi
done
echo "✓ All 5 audit fix markers present."

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

for ds in ${DATASETS}; do
    ckpt="checkpoints/${ds}/best_model.pth"
    if [[ ! -f "${ckpt}" ]]; then
        echo "skipping ${ds}: checkpoint not found at ${ckpt}" >&2
        continue
    fi
    out="${OUTPUT_BASE}/${ds}"
    mkdir -p "${out}"

    echo
    echo "========================================================================"
    echo "REVISED eval on ${ds}  (N=${NUM_SAMPLES}, K=${NUM_TEST_CLASSES})"
    echo "  Fixes applied: B1 (drop ratio), B2 (top-K classes),"
    echo "                 G1 (per-channel baseline), G2 (tie jitter),"
    echo "                 G3 (min_orig_conf=${MIN_ORIG_CONF})"
    echo "========================================================================"

    python3 evaluation/comprehensive_evaluation.py \
        --checkpoint "${ckpt}" \
        --dataset "${ds}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_test_classes "${NUM_TEST_CLASSES}" \
        --output_dir "${out}" \
        --device "${DEVICE}" \
        2>&1 | tee "${out}/run.log"

    echo
    echo "Bootstrap CIs:"
    python3 evaluation/bootstrap_ci.py \
        --results "${out}/all_results.pth" \
        --output  "${out}/bootstrap_ci.json" \
        --n_resamples "${N_BOOTSTRAP}" \
        2>&1 | tee "${out}/bootstrap.log"
done

echo
echo "Done. Outputs in ${OUTPUT_BASE}/<dataset>/"
echo "  comprehensive_report.txt — original-format report (with revised numbers)"
echo "  all_results.pth          — per-sample lists"
echo "  bootstrap_ci.json        — 95% CIs for the manuscript tables"
echo "  bootstrap.log            — pretty-printed CI summary"
