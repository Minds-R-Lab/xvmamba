#!/usr/bin/env bash
#
# run_phase7_extra_datasets.sh
# ----------------------------
# Phase 7 — Train and evaluate VMamba-Tiny on FashionMNIST and EuroSAT
# (the two new datasets added in response to Reviewer 1's request for
# broader non-medical benchmark coverage, including a remote-sensing
# dataset).
#
# Runs the same 3-seed audit-fixed protocol used for Table V in
# Option C, so the new datasets land on identical methodology and slot
# directly into the Table V update.
#
# Run from xvmamba/ with MambaMedEnv conda env active. Idempotent: skips
# any (dataset, seed) cell whose checkpoint or eval output already exists.
#
# Usage:
#     ./scripts/run_phase7_extra_datasets.sh
#
# Override knobs (env vars):
#     DATASETS="fashionmnist eurosat"
#     SEEDS="42 137 2024"
#     EPOCHS=50          BATCH_SIZE=32
#     NUM_SAMPLES=50     NUM_TEST_CLASSES=5
#     DEVICE=cuda
#
# Expected runtime on RTX 3090 (3 seeds × 2 datasets):
#     FashionMNIST: ~2-3 h per seed × 3 = 6-9 h
#     EuroSAT:      ~2-3 h per seed × 3 = 6-9 h
#     Eval (6 × ~25 min)         ≈ 2.5 h
#     Total: ~15-21 GPU hours

set -euo pipefail

SEEDS="${SEEDS:-42 137 2024}"
DATASETS="${DATASETS:-fashionmnist eurosat}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
PATCH_SIZE="${PATCH_SIZE:-4}"

NUM_SAMPLES="${NUM_SAMPLES:-50}"
NUM_TEST_CLASSES="${NUM_TEST_CLASSES:-5}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
DEVICE="${DEVICE:-cuda}"

CKPT_BASE="${CKPT_BASE:-./checkpoints/multiseed}"
EVAL_BASE="${EVAL_BASE:-./results/multiseed}"
LOG_DIR="${LOG_DIR:-./logs/phase7}"

# ----------------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------------
if [[ ! -f "evaluation/comprehensive_evaluation.py" ]]; then
    echo "error: run from xvmamba/ directory" >&2; exit 1
fi
if ! grep -q "FASHIONMNIST = " data/datasets.py; then
    echo "error: data/datasets.py missing FashionMNIST wiring" >&2; exit 1
fi
if ! grep -q "EUROSAT = " data/datasets.py; then
    echo "error: data/datasets.py missing EuroSAT wiring" >&2; exit 1
fi
for needle in '1e-9 * torch.randn_like' 'image.mean(dim=[2, 3]' 'min_orig_conf' 'test_classes_per_image' "'drop_diff':"; do
    if ! grep -q -F -- "${needle}" evaluation/comprehensive_evaluation.py; then
        echo "error: audit fix marker '${needle}' missing" >&2; exit 1
    fi
done

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Phase 7 — FashionMNIST + EuroSAT multi-seed sweep"
echo "  SEEDS=${SEEDS}    DATASETS=${DATASETS}"
echo "============================================================"

# ----------------------------------------------------------------------------
# STAGE 1 — Train every (dataset, seed) combination
# ----------------------------------------------------------------------------
echo; echo "----- Stage 1: train -----"
for seed in ${SEEDS}; do
    for ds in ${DATASETS}; do
        out="${CKPT_BASE}/seed_${seed}"
        ckpt="${out}/${ds}/best_model.pth"
        if [[ -f "${ckpt}" ]]; then
            echo "  [skip] ${ds} seed ${seed}: ${ckpt} exists"; continue
        fi
        mkdir -p "${out}"
        echo
        echo "  ============================================================"
        echo "  TRAIN  vmamba  ${ds}  seed=${seed}  patch=${PATCH_SIZE}"
        echo "  ============================================================"
        python3 scripts/train.py \
            --dataset "${ds}" \
            --model_arch vmamba \
            --seed "${seed}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --patch_size "${PATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            --data_root "${DATA_ROOT}" \
            --device "${DEVICE}" \
            --output_dir "${out}" \
            2>&1 | tee "${LOG_DIR}/vmamba_${ds}_seed${seed}_train.log"
    done
done

# ----------------------------------------------------------------------------
# STAGE 2 — Audit-fixed comprehensive evaluation
# ----------------------------------------------------------------------------
echo; echo "----- Stage 2: evaluate -----"
for seed in ${SEEDS}; do
    for ds in ${DATASETS}; do
        ckpt="${CKPT_BASE}/seed_${seed}/${ds}/best_model.pth"
        if [[ ! -f "${ckpt}" ]]; then
            echo "  [skip] eval ${ds} seed ${seed}: ckpt missing"; continue
        fi
        out="${EVAL_BASE}/seed_${seed}/${ds}"
        if [[ -f "${out}/all_results.pth" ]]; then
            echo "  [skip] eval ${ds} seed ${seed}: ${out}/all_results.pth exists"; continue
        fi
        mkdir -p "${out}"
        echo
        echo "  ============================================================"
        echo "  EVAL  ${ds}  seed=${seed}  N=${NUM_SAMPLES}"
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
    done
done

# ----------------------------------------------------------------------------
# STAGE 3 — Aggregate FashionMNIST and EuroSAT into a Phase 7 summary
# ----------------------------------------------------------------------------
echo; echo "----- Stage 3: aggregate Phase 7 -----"
python3 - <<PYAGG
"""Aggregate FashionMNIST and EuroSAT into a single Phase 7 JSON ready
to feed into the manuscript Table V / Table III / Table IV update."""
import json, glob
from pathlib import Path
from statistics import mean, pstdev
import math
import torch

ROOT = Path("${EVAL_BASE}")
SEEDS = ${SEEDS//\"/}
DATASETS = "${DATASETS}".split()
METHODS = ["Jacobian","Gramian","Grad-CAM","Random"]

def pearson(x, y):
    n = len(x)
    if n < 2: return 0.0
    mx, my = sum(x)/n, sum(y)/n
    sx2 = sum((xi-mx)**2 for xi in x)
    sy2 = sum((yi-my)**2 for yi in y)
    sxy = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    den = math.sqrt(sx2*sy2)
    return sxy/den if den > 0 else 0.0

def get_test_acc(d, s):
    try:
        r = torch.load(f"checkpoints/multiseed/seed_{s}/{d}/training_results.pth",
                       map_location="cpu", weights_only=False)
        return r.get("test_acc")
    except FileNotFoundError:
        return None

def per_seed_metrics(d, method):
    fr = d["faithfulness_raw"][method]
    ins = sum(fr["ins"])/len(fr["ins"])
    de  = sum(fr["del"])/len(fr["del"])
    out = {"insertion": ins, "deletion": de, "faithfulness": ins-de}
    ccs = d.get("cross_class_consistency_raw",{}).get(method)
    if ccs: out["cross_class"] = sum(ccs)/len(ccs)
    pr = d.get("perturbation_invariance_raw",{}).get(method, {})
    if pr:
        if "drop_diff" in pr: out["drop_diff_high"] = sum(pr["drop_diff"])/len(pr["drop_diff"])
        if "high_drop" in pr: out["high_drop"]      = sum(pr["high_drop"])/len(pr["high_drop"])
        if "low_drop"  in pr: out["low_drop"]       = sum(pr["low_drop"])/len(pr["low_drop"])
    return out

def agg_msd(xs):
    if not xs: return None
    return {"mean": mean(xs), "std": pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}

summary = {}
for ds in DATASETS:
    print(f"\n===== {ds.upper()} =====")
    test_accs = [get_test_acc(ds, s) for s in SEEDS]
    valid = [a for a in test_accs if a is not None]
    acc_agg = agg_msd(valid) if valid else None
    print(f"  test_acc per seed: {test_accs}  ->  {acc_agg}")

    rows = {}
    for s in SEEDS:
        p = ROOT / f"seed_{s}" / ds / "all_results.pth"
        if p.exists():
            rows[s] = torch.load(p, map_location="cpu", weights_only=False)
    if not rows:
        print(f"  no eval data yet")
        continue

    summary[ds] = {"test_acc": acc_agg, "methods": {}}
    for m in METHODS:
        per = {s: per_seed_metrics(rows[s], ds, m) for s in rows} if False else \
              {s: per_seed_metrics(rows[s], m) for s in rows}
        method_summary = {}
        for k in ["faithfulness","insertion","deletion","cross_class",
                  "drop_diff_high","high_drop","low_drop"]:
            vals = [per[s][k] for s in per if k in per[s]]
            a = agg_msd(vals)
            if a:
                a["per_seed"] = {s: per[s][k] for s in per if k in per[s]}
                method_summary[k] = a
        summary[ds]["methods"][m] = method_summary
        # Pretty print
        fm = method_summary.get("faithfulness")
        cc = method_summary.get("cross_class")
        line = [f"  {m:10s}"]
        if fm: line.append(f"faith {fm['mean']:+.4f}±{fm['std']:.4f}")
        if cc: line.append(f"cc {cc['mean']:+.4f}±{cc['std']:.4f}")
        print("  ".join(line))

    # Pairwise per-image corr(J_score, G_score) for R4.5 extension
    J_per_seed_score_corr = []
    for s in rows:
        Jr = rows[s]["faithfulness_raw"]["Jacobian"]
        Gr = rows[s]["faithfulness_raw"]["Gramian"]
        Js = [a-b for a,b in zip(Jr["ins"], Jr["del"])]
        Gs = [a-b for a,b in zip(Gr["ins"], Gr["del"])]
        J_per_seed_score_corr.append(pearson(Js, Gs))
    summary[ds]["pearson_J_G_per_image"] = agg_msd(J_per_seed_score_corr)
    if J_per_seed_score_corr:
        a = agg_msd(J_per_seed_score_corr)
        print(f"  r(J_score, G_score) across images: {a['mean']:.4f} ± {a['std']:.4f}")

out = ROOT / "phase7_summary.json"
out.write_text(json.dumps(summary, indent=2, default=float))
print(f"\nwrote {out}")
PYAGG

echo
echo "Done. Outputs:"
echo "  ${CKPT_BASE}/seed_<seed>/{fashionmnist,eurosat}/best_model.pth"
echo "  ${EVAL_BASE}/seed_<seed>/{fashionmnist,eurosat}/all_results.pth"
echo "  ${EVAL_BASE}/phase7_summary.json"
