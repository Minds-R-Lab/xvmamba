#!/usr/bin/env bash
#
# run_aggregation_reeval.sh
# -------------------------
# Re-run the comprehensive evaluation on BloodMNIST and DermaMNIST with the
# patched channel aggregation (L2 norm -> arithmetic mean, matching Eq. 36
# of the manuscript). Produces a clean comparison vs. the v1 paper numbers.
#
# Run from the xvmamba/ directory with the MambaMedEnv conda environment
# active. Single GPU is sufficient.
#
# Usage:
#     ./scripts/run_aggregation_reeval.sh
#
# Output:
#     results/agg_reeval/<dataset>/  -- per-dataset JSON + figures
#     results/agg_reeval/SUMMARY.md  -- side-by-side comparison table

set -euo pipefail

# ----- config ---------------------------------------------------------------

NUM_SAMPLES="${NUM_SAMPLES:-50}"          # matches v1 paper protocol
NUM_TEST_CLASSES="${NUM_TEST_CLASSES:-5}"  # cross-class consistency sample size
OUT_DIR="${OUT_DIR:-results/agg_reeval}"
DATASETS=( "bloodmnist" "dermamnist" )
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_DIR}"

# ----- sanity checks --------------------------------------------------------

if [[ ! -f "controllability/analyzer.py" ]]; then
    echo "error: must be run from xvmamba/ directory" >&2
    exit 1
fi

# Quick grep to confirm the patch is present.
if grep -q "torch.norm(CB, dim=-1)" controllability/analyzer.py; then
    echo "error: analyzer.py still uses torch.norm; patch not applied?" >&2
    exit 1
fi

if ! grep -q "CB.mean(dim=-1)" controllability/analyzer.py; then
    echo "error: analyzer.py is missing the .mean(dim=-1) patch" >&2
    exit 1
fi

# Confirm GPU is visible (don't hard-fail; will fall back to CPU if --device cpu).
python3 -c "import torch; print('cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# ----- run ------------------------------------------------------------------

for d in "${DATASETS[@]}"; do
    ckpt="checkpoints/${d}/best_model.pth"
    if [[ ! -f "${ckpt}" ]]; then
        echo "skipping ${d}: checkpoint not found at ${ckpt}" >&2
        continue
    fi

    echo
    echo "========================================================================"
    echo "Re-evaluating ${d} (num_samples=${NUM_SAMPLES})"
    echo "========================================================================"

    out="${OUT_DIR}/${d}"
    mkdir -p "${out}"

    python3 evaluation/comprehensive_evaluation.py \
        --checkpoint "${ckpt}" \
        --dataset "${d}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_test_classes "${NUM_TEST_CLASSES}" \
        --output_dir "${out}" \
        --device "${DEVICE}" \
        2>&1 | tee "${out}/run.log"
done

# ----- summarise ------------------------------------------------------------

python3 - <<'PY'
"""
Compose a Markdown summary that puts the new (patched) numbers side-by-side
with the v1 paper numbers, so we know at a glance whether anything moved.
"""
import json
import os
from pathlib import Path

# v1 paper numbers from main_tnnls_anonymous.tex (BloodMNIST + DermaMNIST only).
V1 = {
    "bloodmnist": {
        "faithfulness_jacobian": 0.679,
        "faithfulness_gramian":  0.671,
        "faithfulness_gradcam":  0.568,
        "crossclass_jacobian":   1.000,
        "crossclass_gramian":    1.000,
        "crossclass_gradcam":    0.227,
        "perturb_high_jacobian": 0.006,
        "perturb_low_jacobian":  0.000,
    },
    "dermamnist": {
        "faithfulness_jacobian": 0.169,
        "faithfulness_gramian":  0.156,
        "faithfulness_gradcam":  0.233,
        "crossclass_jacobian":   1.000,
        "crossclass_gramian":    1.000,
        "crossclass_gradcam":    0.086,
        "perturb_high_jacobian": 0.055,
        "perturb_low_jacobian":  0.006,
    },
}

out_root = Path(os.environ.get("OUT_DIR", "results/agg_reeval"))

def gd(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

lines = []
lines.append("# Aggregation re-eval summary\n")
lines.append("Patch: `torch.norm(..., dim=-1)` -> `.mean(dim=-1)` in `controllability/analyzer.py` lines 157-186.\n")
lines.append("Synthetic prediction (`scripts/compare_aggregation.py`): Pearson > 0.999 ranking preservation; magnitudes rescale by ~1/sqrt(D).\n")

for ds, v1 in V1.items():
    summary_path = out_root / ds / "comprehensive_results.json"
    lines.append(f"\n## {ds}\n")
    if not summary_path.exists():
        # Fall back to any *.json in the dir.
        cands = list((out_root / ds).glob("*.json")) if (out_root / ds).exists() else []
        if cands:
            summary_path = cands[0]
        else:
            lines.append(f"(no results found at {summary_path})\n")
            continue
    with open(summary_path) as f:
        res = json.load(f)

    # The evaluation script's exact JSON layout depends on the script's
    # internals; we tolerantly probe several common keys. Update the lookup
    # paths below if the actual JSON differs.
    rows = [
        ("Faithfulness | Jacobian", v1["faithfulness_jacobian"],
         gd(res, "faithfulness", "jacobian", "score") or gd(res, "faithfulness", "Jacobian")),
        ("Faithfulness | Gramian", v1["faithfulness_gramian"],
         gd(res, "faithfulness", "gramian", "score") or gd(res, "faithfulness", "Gramian")),
        ("Faithfulness | Grad-CAM", v1["faithfulness_gradcam"],
         gd(res, "faithfulness", "gradcam", "score") or gd(res, "faithfulness", "Grad-CAM")),
        ("Cross-class corr | Jacobian", v1["crossclass_jacobian"],
         gd(res, "cross_class_consistency", "jacobian", "mean_correlation") or
         gd(res, "cross_class_consistency", "Jacobian")),
        ("Cross-class corr | Gramian", v1["crossclass_gramian"],
         gd(res, "cross_class_consistency", "gramian", "mean_correlation") or
         gd(res, "cross_class_consistency", "Gramian")),
        ("Cross-class corr | Grad-CAM", v1["crossclass_gradcam"],
         gd(res, "cross_class_consistency", "gradcam", "mean_correlation") or
         gd(res, "cross_class_consistency", "Grad-CAM")),
    ]
    lines.append("| Metric | v1 (paper) | v2 (patched) | delta |")
    lines.append("|---|---|---|---|")
    for name, v, w in rows:
        if w is None:
            lines.append(f"| {name} | {v:.3f} | (not in JSON) | --- |")
        else:
            try:
                ww = float(w)
                lines.append(f"| {name} | {v:.3f} | {ww:.3f} | {ww - v:+.3f} |")
            except Exception:
                lines.append(f"| {name} | {v:.3f} | {w} | --- |")

summary = out_root / "SUMMARY.md"
summary.write_text("\n".join(lines))
print(f"\nwrote {summary}\n")
print("\n".join(lines))
PY

echo
echo "Done. Results in: ${OUT_DIR}"
echo "Open ${OUT_DIR}/SUMMARY.md for the side-by-side comparison."
