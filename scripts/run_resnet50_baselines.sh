#!/usr/bin/env bash
#
# run_resnet50_baselines.sh
# -------------------------
# Phase 10 — Train ResNet-50 on the three newly-added non-medical datasets
# (CIFAR-100, FashionMNIST, EuroSAT) under the SAME training protocol used
# for VMamba-Tiny, so Table~I's ResNet-50 column reflects matched-protocol
# baselines rather than literature pickups.
#
# The four MedMNIST datasets keep their existing ResNet-50 values
# (Yang et al. 2023, MedMNIST v2 paper), which were also trained from
# scratch under a comparable protocol; the new three add matching values.
#
# Run from the xvmamba/ directory with MambaMedEnv conda env active.
#
# Usage:
#     ./scripts/run_resnet50_baselines.sh
#
# Override knobs:
#     DATASETS="cifar100 fashionmnist eurosat"   # default
#     EPOCHS=50         BATCH_SIZE=32            LR=1.0e-4
#     SEEDS="42"                                 # default (single seed)
#     OUT_BASE=./checkpoints/resnet50_baseline
#
# Estimated runtime on RTX 3090 (single seed, 3 datasets):
#     CIFAR-100      ~6 h
#     FashionMNIST   ~3 h
#     EuroSAT        ~2 h
# Total: ≈10–12 GPU-hours.
# Override with SEEDS="42 137 2024" if multi-seed coverage is desired.

set -euo pipefail

DATASETS="${DATASETS:-cifar100 fashionmnist eurosat}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-./data}"
DEVICE="${DEVICE:-cuda}"
OUT_BASE="${OUT_BASE:-./checkpoints/resnet50_baseline}"
LOG_DIR="${LOG_DIR:-./logs/resnet50}"

mkdir -p "${LOG_DIR}"
python3 -c "import torch, torchvision; assert torch.cuda.is_available(), 'CUDA required'"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Datasets: ${DATASETS}    Seeds: ${SEEDS}"

# Inline trainer — reuses our data loaders. Trains torchvision.models.resnet50
# from scratch (no ImageNet pretraining) to match how VMamba-Tiny was trained.
TRAIN_PY=$(cat <<'PYEND'
import argparse, os, sys, time
import torch, torch.nn as nn, torch.optim as optim
import torchvision.models as tvm

sys.path.insert(0, '.')
from data import DatasetType, get_dataloader, get_dataset_info

def set_seed(s):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    info = get_dataset_info(DatasetType(args.dataset))
    num_classes = info["num_classes"]
    print(f"  dataset={args.dataset}  num_classes={num_classes}")

    train_loader, val_loader, test_loader = get_dataloader(
        dataset_type=DatasetType(args.dataset),
        batch_size=args.batch_size, image_size=224,
        num_workers=args.num_workers, data_root=args.data_root,
    )

    model = tvm.resnet50(weights=None, num_classes=num_classes).to(args.device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_acc = 0.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        total, correct, ls = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward(); opt.step()
            ls += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = 100.0 * correct / total
        # val
        model.eval(); total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = 100.0 * correct / total
        sched.step()
        dt = time.time() - t0
        print(f"  ep {ep+1:02d}/{args.epochs}  train_acc={train_acc:.2f}%  val_acc={val_acc:.2f}%  ({dt:.1f}s)")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(),
                        "best_val_acc": best_val_acc,
                        "args": vars(args)},
                       os.path.join(args.output_dir, "best_model.pth"))
    # test
    ck = torch.load(os.path.join(args.output_dir, "best_model.pth"),
                    map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    test_acc = 100.0 * correct / total
    print(f"  TEST_ACC: {test_acc:.2f}%   best_val_acc: {best_val_acc:.2f}%")
    torch.save({"test_acc": test_acc, "best_val_acc": best_val_acc, "args": vars(args)},
               os.path.join(args.output_dir, "training_results.pth"))

if __name__ == "__main__":
    main()
PYEND
)

for seed in ${SEEDS}; do
    for ds in ${DATASETS}; do
        out="${OUT_BASE}/seed_${seed}/${ds}"
        if [[ -f "${out}/training_results.pth" ]]; then
            echo "[skip] ${ds} seed ${seed}: already trained"; continue
        fi
        mkdir -p "${out}"
        echo
        echo "============================================================"
        echo "  TRAIN  resnet50  ${ds}  seed=${seed}  epochs=${EPOCHS}"
        echo "============================================================"
        python3 -c "${TRAIN_PY}" \
            --dataset "${ds}" \
            --seed "${seed}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --lr "${LR}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --num_workers "${NUM_WORKERS}" \
            --data_root "${DATA_ROOT}" \
            --device "${DEVICE}" \
            --output_dir "${out}" \
            2>&1 | tee "${LOG_DIR}/resnet50_${ds}_seed${seed}.log"
    done
done

# ---- Aggregate test accuracies and print Table I additions ----
echo
echo "============================================================"
echo "  ResNet-50 baseline summary (3-seed mean ± std)"
echo "============================================================"
python3 - <<PYAGG
import torch, glob
from pathlib import Path
from statistics import mean, pstdev
import json
BASE = Path("${OUT_BASE}")
DATASETS = "${DATASETS}".split()
SEEDS = ${SEEDS//\"/}
summary = {}
for ds in DATASETS:
    accs = []
    for s in SEEDS:
        f = BASE / f"seed_{s}" / ds / "training_results.pth"
        if f.exists():
            r = torch.load(f, map_location="cpu", weights_only=False)
            accs.append(r["test_acc"])
    if accs:
        mu = mean(accs); sd = pstdev(accs) if len(accs) > 1 else 0
        summary[ds] = {"mean": mu, "std": sd, "n": len(accs), "per_seed": accs}
        print(f"  {ds:<14s}: {mu:.2f} ± {sd:.2f}%  (n={len(accs)})  per-seed: {[round(a,2) for a in accs]}")
(BASE / "resnet50_summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(f"\n  wrote {BASE/'resnet50_summary.json'}")
PYAGG

echo
echo "Done. To fold into Table I, run:"
echo "  python scripts/apply_resnet50_to_manuscript.py"
