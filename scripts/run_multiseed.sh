#!/usr/bin/env bash
#
# run_multiseed.sh
# ----------------
# Phase 2.5 — multi-seed runs for statistical reliability.
#
# Trains VMamba-Tiny from scratch on BloodMNIST and DermaMNIST with three
# distinct random seeds each, then runs the (audit-fixed) comprehensive
# evaluation on every resulting checkpoint and finally aggregates the
# faithfulness scores across seeds to report mean ± std alongside the
# bootstrap-over-images CIs we already report.
#
# Layout produced under ./checkpoints/multiseed/ :
#   seed_42/<dataset>/best_model.pth        training run with seed 42
#   seed_137/<dataset>/best_model.pth       training run with seed 137
#   seed_2024/<dataset>/best_model.pth      training run with seed 2024
# Layout produced under ./results/multiseed/ :
#   seed_42/<dataset>/comprehensive_report.txt + all_results.pth
#   seed_137/<dataset>/...
#   seed_2024/<dataset>/...
#   multiseed_summary_<dataset>.json        aggregated mean ± std per metric
#
# Usage:
#     # from inside xvmamba/  (the directory containing checkpoints/ and
#     # evaluation/comprehensive_evaluation.py)
#     ./scripts/run_multiseed.sh
#
#     # to override defaults:
#     DATASETS="bloodmnist"     ./scripts/run_multiseed.sh
#     SEEDS="42 137"            ./scripts/run_multiseed.sh
#     EPOCHS=30 NUM_SAMPLES=50  ./scripts/run_multiseed.sh
#     SKIP_TRAIN=1              ./scripts/run_multiseed.sh   # reuse existing ckpts
#     SKIP_EVAL=1               ./scripts/run_multiseed.sh   # only aggregate
#
# Estimated runtime on RTX 3090 (BloodMNIST + DermaMNIST, 3 seeds each):
#     Training:  ~1 GPU-day per (seed, dataset) ⇒ ~6 GPU-days total
#     Eval    :  ~25 min per (seed, dataset)    ⇒ ~2.5 hours total
#     Aggregate: a few seconds
# Total: roughly one week of GPU time end-to-end.
#
# Notes:
#   - The training script (xvmamba/scripts/train.py) already exposes --seed
#     and --output_dir as CLI flags, so we route each seed to its own
#     per-seed subdir so subsequent seeds don't clobber earlier checkpoints.
#   - The evaluation script reuses the audit-fixed
#     comprehensive_evaluation.py with the 5 fixes already in place
#     (per-channel baseline, tie-jitter, top-K predicted classes,
#     min_orig_conf flag, drop_diff column). No re-patching required.
#   - The aggregator (multiseed_aggregate.py) reports mean ± std *across
#     seeds* over the per-sample mean of each metric. This is the second
#     source of variance (training stochasticity), complementary to the
#     bootstrap-over-images CIs which capture the first source
#     (test-image sampling).

set -euo pipefail

DATASETS="${DATASETS:-bloodmnist dermamnist}"
SEEDS="${SEEDS:-42 137 2024}"
EPOCHS="${EPOCHS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_TEST_CLASSES="${NUM_TEST_CLASSES:-5}"
DEVICE="${DEVICE:-cuda}"
CKPT_BASE="${CKPT_BASE:-./checkpoints/multiseed}"
EVAL_BASE="${EVAL_BASE:-./results/multiseed}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# --- Pre-flight checks ------------------------------------------------------
if [[ ! -f "evaluation/comprehensive_evaluation.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi
if [[ ! -f "scripts/train.py" ]]; then
    echo "error: scripts/train.py missing" >&2; exit 1
fi

# Verify the audit fixes are present (so the per-seed eval is on the
# same protocol as the headline numbers).
for needle in '1e-9 * torch.randn_like' 'image.mean(dim=[2, 3]' 'min_orig_conf' 'test_classes_per_image' "'drop_diff':"; do
    if ! grep -q -F -- "${needle}" evaluation/comprehensive_evaluation.py; then
        echo "error: audit fix marker '${needle}' missing" >&2
        exit 1
    fi
done
echo "✓ all 5 audit fix markers present"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Datasets: ${DATASETS}"
echo "Seeds   : ${SEEDS}"
echo

# --- Training stage ---------------------------------------------------------
if [[ "${SKIP_TRAIN}" != "1" ]]; then
    for seed in ${SEEDS}; do
        for ds in ${DATASETS}; do
            out="${CKPT_BASE}/seed_${seed}"
            ckpt="${out}/${ds}/best_model.pth"
            if [[ -f "${ckpt}" ]]; then
                echo "[train] already have ${ckpt}; skipping"
                continue
            fi
            echo
            echo "============================================================"
            echo "TRAIN ${ds}  seed=${seed}  epochs=${EPOCHS}"
            echo "============================================================"
            mkdir -p "${out}"
            python3 scripts/train.py \
                --dataset "${ds}" \
                --seed "${seed}" \
                --epochs "${EPOCHS}" \
                --output_dir "${out}" \
                --device "${DEVICE}" \
                2>&1 | tee "${out}/${ds}_train.log"
        done
    done
else
    echo "[train] SKIP_TRAIN=1, reusing existing checkpoints"
fi

# --- Evaluation stage -------------------------------------------------------
if [[ "${SKIP_EVAL}" != "1" ]]; then
    for seed in ${SEEDS}; do
        for ds in ${DATASETS}; do
            ckpt="${CKPT_BASE}/seed_${seed}/${ds}/best_model.pth"
            if [[ ! -f "${ckpt}" ]]; then
                echo "[eval] missing ${ckpt}; skipping" >&2
                continue
            fi
            out="${EVAL_BASE}/seed_${seed}/${ds}"
            if [[ -f "${out}/all_results.pth" ]]; then
                echo "[eval] already have ${out}/all_results.pth; skipping"
                continue
            fi
            mkdir -p "${out}"
            echo
            echo "============================================================"
            echo "EVAL  ${ds}  seed=${seed}  N=${NUM_SAMPLES}  K=${NUM_TEST_CLASSES}"
            echo "============================================================"
            python3 evaluation/comprehensive_evaluation.py \
                --checkpoint "${ckpt}" \
                --dataset "${ds}" \
                --num_samples "${NUM_SAMPLES}" \
                --num_test_classes "${NUM_TEST_CLASSES}" \
                --output_dir "${out}" \
                --device "${DEVICE}" \
                2>&1 | tee "${out}/run.log"
        done
    done
else
    echo "[eval] SKIP_EVAL=1, reusing existing eval outputs"
fi

# --- Aggregation stage ------------------------------------------------------
echo
echo "============================================================"
echo "AGGREGATE  mean ± std across seeds"
echo "============================================================"
for ds in ${DATASETS}; do
    # Collect per-seed all_results.pth files for this dataset.
    seed_files=()
    for seed in ${SEEDS}; do
        f="${EVAL_BASE}/seed_${seed}/${ds}/all_results.pth"
        [[ -f "${f}" ]] && seed_files+=("${f}")
    done
    if [[ ${#seed_files[@]} -lt 2 ]]; then
        echo "[agg] ${ds}: fewer than 2 seed results found; skipping"
        continue
    fi
    out="${EVAL_BASE}/multiseed_summary_${ds}.json"
    python3 evaluation/multiseed_aggregate.py \
        --dataset "${ds}" \
        --output "${out}" \
        "${seed_files[@]}"
    echo "  -> ${out}"
done

echo
echo "Done. Per-seed outputs in ${EVAL_BASE}/seed_<seed>/<dataset>/,"
echo "per-dataset aggregates in ${EVAL_BASE}/multiseed_summary_<dataset>.json."
