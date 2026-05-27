#!/usr/bin/env bash
#
# run_table_v_full_multiseed.sh
# -----------------------------
# Option C — bring every row of Table V to a uniform 3-seed multi-seed
# protocol under the audit-fixed evaluation. Trains every missing
# (dataset, seed) combination, copies existing seed-42 checkpoints into
# the multiseed layout, then runs the audit-fixed comprehensive
# evaluation on all 18 (6 datasets × 3 seeds) checkpoints and finally
# aggregates them into a single uniform table.
#
# Run from the xvmamba/ directory.  Idempotent: skips work that's
# already complete (checkpoint or eval output exists). Safe to Ctrl-C
# and resume.
#
#   ./scripts/run_table_v_full_multiseed.sh
#
# Override knobs (all optional):
#   SEEDS="42 137 2024"
#   VMAMBA_DATASETS="octmnist pneumoniamnist cifar100 bloodmnist dermamnist"
#   VIM_DATASETS="bloodmnist"                # Vim is parameterised separately
#   EPOCHS=50          BATCH_SIZE=32
#   NUM_SAMPLES=50     NUM_TEST_CLASSES=5
#
# Expected end-to-end runtime on RTX 3090:
#   Training (the long bit):
#     OCT × 3 seeds         ≈ 10–12 h
#     Pneumonia × 3 seeds   ≈ 6–9 h
#     CIFAR-100 × 2 seeds   ≈ 20–24 h
#     Vim × 2 seeds         ≈ 20–24 h
#   Evaluations:
#     18 × ~25 min          ≈ 7–8 h
#   Total: ~65–75 GPU-hours (≈ 3 days continuous).

set -euo pipefail

SEEDS="${SEEDS:-42 137 2024}"
# VMamba runs (CIFAR-100 uses patch_size=4 like the v1 medical recipe).
VMAMBA_DATASETS="${VMAMBA_DATASETS:-octmnist pneumoniamnist cifar100 bloodmnist dermamnist}"
# Vim runs (separate config: depth=12, d_model=192, patch=16).
VIM_DATASETS="${VIM_DATASETS:-bloodmnist}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"

# Eval (audit-fixed protocol)
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_TEST_CLASSES="${NUM_TEST_CLASSES:-5}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
DEVICE="${DEVICE:-cuda}"

CKPT_BASE="${CKPT_BASE:-./checkpoints/multiseed}"
EVAL_BASE="${EVAL_BASE:-./results/multiseed}"
LOG_DIR="${LOG_DIR:-./logs/table_v}"

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------
if [[ ! -f "evaluation/comprehensive_evaluation.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi

# Audit-fix markers — must be present for evaluation rigour.
for needle in '1e-9 * torch.randn_like' 'image.mean(dim=[2, 3]' 'min_orig_conf' 'test_classes_per_image' "'drop_diff':"; do
    if ! grep -q -F -- "${needle}" evaluation/comprehensive_evaluation.py; then
        echo "error: audit fix marker '${needle}' missing" >&2
        exit 1
    fi
done

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Option C — full multi-seed sweep for Table V"
echo "  SEEDS=${SEEDS}"
echo "  VMAMBA_DATASETS=${VMAMBA_DATASETS}"
echo "  VIM_DATASETS=${VIM_DATASETS}"
echo "============================================================"

# ----------------------------------------------------------------------------
# STAGE 1 — Copy existing seed-42 checkpoints into the multiseed layout
# ----------------------------------------------------------------------------
echo; echo "----- Stage 1: copy existing seed-42 checkpoints into multiseed/ -----"

copy_if_missing() {
    # $1 = source ckpt path     $2 = destination ckpt path
    local src="$1"; local dst="$2"
    if [[ -f "${dst}" ]]; then
        echo "  [skip] ${dst} already exists"; return 0
    fi
    if [[ ! -f "${src}" ]]; then
        echo "  [skip] source ${src} not present"; return 0
    fi
    mkdir -p "$(dirname "${dst}")"
    cp "${src}" "${dst}"
    echo "  [copy] ${src} -> ${dst}"
}

# VMamba seed-42 hand-overs (Blood and Derma are already in multiseed/).
copy_if_missing "./checkpoints/cifar100/best_model.pth"        "${CKPT_BASE}/seed_42/cifar100/best_model.pth"
copy_if_missing "./checkpoints/octmnist/best_model.pth"        "${CKPT_BASE}/seed_42/octmnist/best_model.pth"
copy_if_missing "./checkpoints/pneumoniamnist/best_model.pth"  "${CKPT_BASE}/seed_42/pneumoniamnist/best_model.pth"
# Vim BloodMNIST seed-42.
copy_if_missing "./checkpoints_vim/bloodmnist/best_model.pth"  "${CKPT_BASE}/seed_42/bloodmnist_vim/best_model.pth"

# ----------------------------------------------------------------------------
# STAGE 2 — Train missing (dataset, seed) combinations
# ----------------------------------------------------------------------------
echo; echo "----- Stage 2: train missing seeds -----"

train_vmamba() {
    # $1 = dataset, $2 = seed
    local ds="$1"; local seed="$2"
    local out="${CKPT_BASE}/seed_${seed}"
    local ckpt="${out}/${ds}/best_model.pth"
    if [[ -f "${ckpt}" ]]; then
        echo "  [skip] vmamba ${ds} seed ${seed}: ${ckpt} exists"; return 0
    fi
    local patch=4
    [[ "${ds}" == "cifar100" ]] && patch=4
    mkdir -p "${out}"
    echo
    echo "  ============================================================"
    echo "  TRAIN  vmamba  ${ds}  seed=${seed}  patch=${patch}"
    echo "  ============================================================"
    python3 scripts/train.py \
        --dataset "${ds}" \
        --model_arch vmamba \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --patch_size "${patch}" \
        --num_workers "${NUM_WORKERS}" \
        --data_root "${DATA_ROOT}" \
        --device "${DEVICE}" \
        --output_dir "${out}" \
        2>&1 | tee "${LOG_DIR}/vmamba_${ds}_seed${seed}_train.log"
}

train_vim() {
    # $1 = dataset, $2 = seed.  Vim writes to ${out}/${ds}_vim/ by convention.
    local ds="$1"; local seed="$2"
    local out="${CKPT_BASE}/seed_${seed}"
    local ckpt="${out}/${ds}_vim/best_model.pth"
    if [[ -f "${ckpt}" ]]; then
        echo "  [skip] vim ${ds} seed ${seed}: ${ckpt} exists"; return 0
    fi
    mkdir -p "${out}"
    echo
    echo "  ============================================================"
    echo "  TRAIN  vim     ${ds}  seed=${seed}"
    echo "  ============================================================"
    # Vim writes <output_dir>/<dataset>/best_model.pth; we want
    # multiseed/seed_<seed>/<dataset>_vim/best_model.pth so route via a
    # sibling subdir.
    local tmpdir
    tmpdir="$(mktemp -d -p "${out}" vimout.XXXX)"
    python3 scripts/train.py \
        --dataset "${ds}" \
        --model_arch vim \
        --seed "${seed}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --vim_depth 12 \
        --vim_d_model 192 \
        --vim_mlp_ratio 4.0 \
        --patch_size 16 \
        --image_size 224 \
        --num_workers "${NUM_WORKERS}" \
        --data_root "${DATA_ROOT}" \
        --device "${DEVICE}" \
        --output_dir "${tmpdir}" \
        2>&1 | tee "${LOG_DIR}/vim_${ds}_seed${seed}_train.log"
    if [[ -f "${tmpdir}/${ds}/best_model.pth" ]]; then
        mkdir -p "${out}/${ds}_vim"
        mv "${tmpdir}/${ds}"/* "${out}/${ds}_vim/"
        rm -rf "${tmpdir}"
    else
        echo "  [error] expected ${tmpdir}/${ds}/best_model.pth not produced" >&2
    fi
}

for seed in ${SEEDS}; do
    for ds in ${VMAMBA_DATASETS}; do
        train_vmamba "${ds}" "${seed}"
    done
    for ds in ${VIM_DATASETS}; do
        train_vim "${ds}" "${seed}"
    done
done

# ----------------------------------------------------------------------------
# STAGE 3 — Audit-fixed comprehensive evaluation on every (dataset, seed)
# ----------------------------------------------------------------------------
echo; echo "----- Stage 3: audit-fixed evaluation -----"

eval_one() {
    # $1 = dataset, $2 = seed, $3 = checkpoint subdir (ds for vmamba; ds_vim for vim)
    local ds="$1"; local seed="$2"; local ckpt_subdir="$3"
    local ckpt="${CKPT_BASE}/seed_${seed}/${ckpt_subdir}/best_model.pth"
    if [[ ! -f "${ckpt}" ]]; then
        echo "  [skip] eval ${ckpt_subdir} seed ${seed}: ckpt missing"; return 0
    fi
    local out="${EVAL_BASE}/seed_${seed}/${ckpt_subdir}"
    if [[ -f "${out}/all_results.pth" ]]; then
        echo "  [skip] eval ${ckpt_subdir} seed ${seed}: ${out}/all_results.pth exists"; return 0
    fi
    mkdir -p "${out}"
    echo
    echo "  ============================================================"
    echo "  EVAL  ${ckpt_subdir}  seed=${seed}  N=${NUM_SAMPLES}"
    echo "  ============================================================"
    python3 evaluation/comprehensive_evaluation.py \
        --checkpoint "${ckpt}" \
        --dataset "${ds}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_test_classes "${NUM_TEST_CLASSES}" \
        --output_dir "${out}" \
        --device "${DEVICE}" \
        2>&1 | tee "${out}/run.log"
    echo "  ----- bootstrap CIs"
    python3 evaluation/bootstrap_ci.py \
        --results "${out}/all_results.pth" \
        --output  "${out}/bootstrap_ci.json" \
        --n_resamples "${N_BOOTSTRAP}" \
        2>&1 | tee "${out}/bootstrap.log" || true
}

for seed in ${SEEDS}; do
    for ds in ${VMAMBA_DATASETS}; do
        eval_one "${ds}" "${seed}" "${ds}"
    done
    for ds in ${VIM_DATASETS}; do
        eval_one "${ds}" "${seed}" "${ds}_vim"
    done
done

# ----------------------------------------------------------------------------
# STAGE 4 — Cross-(dataset, seed) aggregation into one Table V CSV
# ----------------------------------------------------------------------------
echo; echo "----- Stage 4: aggregate -----"

python3 - <<PYAGG
"""Aggregate every results/multiseed/seed_*/<ds_or_ds_vim>/all_results.pth
into mean ± std across seeds for the headline metrics: faithfulness
(I − D), cross-class consistency, perturbation drop_diff_high. Write a
single JSON + CSV that feeds directly into the Table V update."""
import json, glob, csv
from pathlib import Path
from statistics import mean, pstdev
import torch

METHODS = ["Jacobian", "Gramian", "Grad-CAM", "Random"]
BASE = Path("${EVAL_BASE}")

def load_per_seed(method, ds):
    rows = {}  # seed -> dict of metric -> value
    for f in sorted(BASE.glob(f"seed_*/{ds}/all_results.pth")):
        seed = int(f.parts[-3].split("_")[1])
        d = torch.load(f, map_location="cpu", weights_only=False)
        fr = d["faithfulness_raw"][method]
        ins = sum(fr["ins"]) / len(fr["ins"])
        de  = sum(fr["del"]) / len(fr["del"])
        ccs = d.get("cross_class_consistency_raw", {}).get(method)
        cc  = (sum(ccs) / len(ccs)) if ccs else None
        pr = d.get("perturbation_invariance_raw", {}).get(method, {})
        dd = (sum(pr["drop_diff"]) / len(pr["drop_diff"])) if pr else None
        rows[seed] = {
            "faithfulness": ins - de,
            "insertion":    ins,
            "deletion":     de,
            "cross_class":  cc,
            "drop_diff_high": dd,
        }
    return rows

def agg(rows, key):
    vals = [v[key] for v in rows.values() if v.get(key) is not None]
    if len(vals) < 1: return None
    return {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals), "per_seed": {s: v[key] for s, v in rows.items() if v.get(key) is not None}}

# Discover datasets present (vmamba and vim subdirs)
datasets = set()
for f in BASE.glob("seed_*/*/all_results.pth"):
    datasets.add(f.parts[-2])
datasets = sorted(datasets)
print(f"discovered: {datasets}")

summary = {}
for ds in datasets:
    summary[ds] = {}
    for method in METHODS:
        rows = load_per_seed(method, ds)
        if not rows: continue
        summary[ds][method] = {
            "faithfulness":   agg(rows, "faithfulness"),
            "insertion":      agg(rows, "insertion"),
            "deletion":       agg(rows, "deletion"),
            "cross_class":    agg(rows, "cross_class"),
            "drop_diff_high": agg(rows, "drop_diff_high"),
        }

(BASE / "table_v_summary.json").write_text(json.dumps(summary, indent=2, default=float))

# CSV with the four headline numbers (faithfulness mean±std for each method).
out_csv = BASE / "table_v_summary.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset","method","faithfulness_mean","faithfulness_std","n_seeds","cross_class_mean","cross_class_std"])
    for ds in datasets:
        for m in METHODS:
            fm = summary[ds].get(m, {}).get("faithfulness")
            cc = summary[ds].get(m, {}).get("cross_class")
            w.writerow([
                ds, m,
                f"{fm['mean']:.4f}" if fm else "",
                f"{fm['std']:.4f}"  if fm else "",
                fm["n"]             if fm else 0,
                f"{cc['mean']:.4f}" if cc else "",
                f"{cc['std']:.4f}"  if cc else "",
            ])
print(f"wrote {out_csv}")

# Pretty stdout summary
print()
print(f"{'dataset':<18s} {'method':<10s} {'faithfulness':>18s} {'cross_class':>18s}")
print("-" * 70)
for ds in datasets:
    for m in METHODS:
        fm = summary[ds].get(m, {}).get("faithfulness")
        cc = summary[ds].get(m, {}).get("cross_class")
        fm_s = f"{fm['mean']:+.4f} ± {fm['std']:.4f}" if fm else "  -"
        cc_s = f"{cc['mean']:+.4f} ± {cc['std']:.4f}" if cc else "  -"
        print(f"{ds:<18s} {m:<10s} {fm_s:>18s} {cc_s:>18s}")
PYAGG

echo
echo "Done. Outputs:"
echo "  ${CKPT_BASE}/seed_<seed>/<dataset>[_vim]/best_model.pth"
echo "  ${EVAL_BASE}/seed_<seed>/<dataset>[_vim]/all_results.pth"
echo "  ${EVAL_BASE}/table_v_summary.{json,csv}"
