"""
Aggregation ablation + per-direction consistency analysis.

These are eval-only studies on an existing checkpoint that re-use the
cached SSM parameters captured during one analysis-mode forward pass.

Phase 3.1 (R3 #1) -- Aggregation ablation
=========================================
The current manuscript aggregates per-channel/direction/block influence
maps by uniform arithmetic mean. R3 asked us to compare alternatives.
For each strategy below we recompute the spatial saliency map from the
same cached parameters, then measure faithfulness (insertion AUC minus
deletion AUC) on the same test images.

Strategies:

  Channel  aggregation: mean (default), max, top-k mean, weighted by |c|.
  Direction aggregation: mean (default), max, per-direction (no merge).
  Layer-depth aggregation: mean (default), shallow-only (stages 0-1),
                           deep-only (stages 2-3), depth-weighted.

Phase 3.2 (R4 Q4) -- Per-direction consistency
==============================================
For each image, compute the four per-direction maps separately, then
report pairwise Pearson correlations across the 4*3/2 = 6 pairs.
Patterns we expect to confirm: forward/backward pairs along the same
axis (H or V) should correlate near +1; orthogonal pairs (H vs V)
should be lower but still positive.

Run:
    python evaluation/run_aggregation_ablation.py \\
        --checkpoint ./checkpoints/bloodmnist/best_model.pth \\
        --dataset bloodmnist \\
        --num_samples 50 \\
        --output_dir ./results/aggregation_ablation/bloodmnist
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import DatasetType, get_dataloader
from evaluation.comprehensive_evaluation import (
    load_model, FaithfulnessEvaluator,
)
from controllability.analyzer import (
    JacobianControllability,
    GramianControllability,
    ControllabilityResult,
    FullControllabilityAnalysis,
)
from models.ss2d import ScanDirection


# ----- aggregation primitives ------------------------------------------------

def _aggregate_channel(per_channel: torch.Tensor, mode: str, top_k: int = 8,
                       weights: torch.Tensor | None = None) -> torch.Tensor:
    """Combine a `[D]` per-channel scalar into a single scalar.

    Modes: "mean", "max", "topk" (top-k mean), "weighted" (sum normalised by
    `weights.sum()`).
    """
    if mode == "mean":
        return per_channel.mean(dim=-1)
    if mode == "max":
        return per_channel.max(dim=-1).values
    if mode == "topk":
        k = min(top_k, per_channel.shape[-1])
        return per_channel.topk(k, dim=-1).values.mean(dim=-1)
    if mode == "weighted":
        w = weights if weights is not None else torch.ones_like(per_channel)
        return (per_channel * w).sum(dim=-1) / (w.sum(dim=-1) + 1e-8)
    raise ValueError(f"unknown channel aggregation: {mode}")


def _aggregate_layer(per_block_maps: list[torch.Tensor], mode: str,
                     depths: list[int]) -> torch.Tensor:
    """Combine a list of `[H, W]` per-block maps into a single `[H, W]` map.

    `depths[i]` is the stage index (0-3 for VMamba, 0 for Vim) of block `i`.
    """
    if not per_block_maps:
        return None
    # Upsample all to the largest resolution.
    max_h = max(m.shape[0] for m in per_block_maps)
    max_w = max(m.shape[1] for m in per_block_maps)
    ups = []
    for m in per_block_maps:
        if m.shape != (max_h, max_w):
            m = F.interpolate(
                m.unsqueeze(0).unsqueeze(0).float(),
                size=(max_h, max_w), mode="bilinear", align_corners=False,
            ).squeeze()
        ups.append(m)
    stacked = torch.stack(ups, dim=0)        # [N, H, W]

    if mode == "mean":
        return stacked.mean(dim=0)
    if mode == "max":
        return stacked.max(dim=0).values
    if mode == "shallow":
        keep = [i for i, d in enumerate(depths) if d <= 1]
        return stacked[keep].mean(dim=0) if keep else stacked.mean(dim=0)
    if mode == "deep":
        keep = [i for i, d in enumerate(depths) if d >= 2]
        return stacked[keep].mean(dim=0) if keep else stacked.mean(dim=0)
    if mode == "depth_weighted":
        # Linear weight by stage index (deeper -> heavier).
        ws = torch.tensor([d + 1 for d in depths], dtype=stacked.dtype).view(-1, 1, 1)
        return (stacked * ws).sum(dim=0) / ws.sum()
    raise ValueError(f"unknown layer aggregation: {mode}")


# ----- direction-keyed map computation ---------------------------------------

def _per_direction_map(cache, method: str, channel_mode: str,
                       top_k: int) -> dict[str, torch.Tensor]:
    """Compute a `[H, W]` map per direction key for a single block's cache.

    Performance note: the position-to-index mapping is applied with a
    single tensor gather (`seq[pos.long()]`) instead of a Python double
    loop. The previous loop took ~30 min per image on a 56x56 grid; the
    gather brings this down to a fraction of a second.
    """
    out = {}
    for direction, ssm in cache.direction_caches.items():
        if ssm.A_bar is None:
            continue
        if method == "jacobian":
            total, direct, prop = JacobianControllability.compute_influence_1d(
                ssm.A_bar, ssm.B_bar, ssm.C,
            )
        elif method == "gramian":
            total = GramianControllability.compute_influence_1d(
                ssm.A_bar, ssm.B_bar, ssm.C,
            )
        else:
            raise ValueError(method)
        # `total` is `[batch, length]`. Mean over batch -> [length].
        seq = total.mean(dim=0)                    # [length]
        H, W = cache.height, cache.width
        pos = cache.position_to_index.get(direction)
        if pos is None:
            # Fall back to row-major reshape.
            map2d = seq.view(H, W)
        else:
            # Vectorised gather: pos has shape [H, W] of integer sequence
            # indices; seq[pos] returns a [H, W] tensor in one GPU op.
            map2d = seq[pos.long()]                # [H, W]
        out[direction.value if hasattr(direction, "value") else str(direction)] = map2d.cpu()
    return out


def _per_block_aggregated(analysis: FullControllabilityAnalysis, method: str,
                          direction_mode: str, channel_mode: str,
                          top_k: int) -> tuple[list[torch.Tensor], list[int]]:
    """Returns (list of `[H,W]` per-block maps, list of stage indices)."""
    maps, depths = [], []
    for stage_idx, stage_blocks in enumerate(analysis.stage_caches):
        for cache in stage_blocks:
            per_dir = _per_direction_map(cache, method, channel_mode, top_k)
            if not per_dir:
                continue
            if direction_mode == "mean":
                stk = torch.stack(list(per_dir.values()))
                blk_map = stk.mean(dim=0)
            elif direction_mode == "max":
                stk = torch.stack(list(per_dir.values()))
                blk_map = stk.max(dim=0).values
            else:
                raise ValueError(direction_mode)
            maps.append(blk_map)
            depths.append(stage_idx)
    return maps, depths


def _final_map_from_strategy(analysis: FullControllabilityAnalysis,
                              method: str,
                              channel_mode: str,
                              direction_mode: str,
                              layer_mode: str,
                              target_hw: tuple[int, int],
                              top_k: int = 8) -> torch.Tensor:
    """Compute the final aggregated `[H, W]` saliency for the input image,
    upsampled to `target_hw`, using the requested strategy."""
    per_block, depths = _per_block_aggregated(analysis, method,
                                              direction_mode, channel_mode, top_k)
    final_lo = _aggregate_layer(per_block, layer_mode, depths)
    if final_lo is None:
        return torch.zeros(target_hw)
    final = F.interpolate(
        final_lo.unsqueeze(0).unsqueeze(0).float(),
        size=target_hw, mode="bilinear", align_corners=False,
    ).squeeze()
    # Min-max normalise.
    mn, mx = final.min(), final.max()
    if (mx - mn).abs() < 1e-8:
        return torch.zeros_like(final).cpu()
    return ((final - mn) / (mx - mn + 1e-8)).cpu()


# ----- main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregation ablation + per-direction consistency")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="jacobian", choices=["jacobian", "gramian"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir or f"./results/aggregation_ablation/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AGGREGATION ABLATION + PER-DIRECTION CONSISTENCY")
    print("=" * 72)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  Output:      {out}")
    print(f"  Device:      {device}")
    print(f"  Method:      {args.method}")

    model, config = load_model(args.checkpoint, device)
    # We toggle analysis mode per-image: on for the single forward that
    # captures the SSM caches, off for the ~280 faithfulness forwards
    # (deletion + insertion AUC at 21 steps each, across 7 strategies).
    # Leaving analysis mode on for the perturbation forwards adds the full
    # per-block cache capture overhead and inflates per-image time by ~30x.

    dataset_type = DatasetType(args.dataset)
    _, _, test_loader = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )

    fevaluator = FaithfulnessEvaluator(model, device)

    # Strategies grid. Each entry is (direction_mode, layer_mode).
    # Note: channel aggregation is performed inside
    # `JacobianControllability.compute_influence_1d` (which returns a
    # per-position scalar already channel-averaged after the 2026-05-14
    # patch). Exposing per-channel intermediates would require touching
    # the analyzer; we leave that as future work and ablate at the
    # direction and depth levels here.
    strategies: dict[str, tuple[str, str]] = {
        "mean (current default)":   ("mean", "mean"),
        "max-direction":            ("max", "mean"),
        "shallow-only":             ("mean", "shallow"),
        "deep-only":                ("mean", "deep"),
        "depth-weighted":           ("mean", "depth_weighted"),
        "max-direction + deep":     ("max", "deep"),
        "max-direction + shallow":  ("max", "shallow"),
    }

    # --- 1) per-image: compute analysis once, run every strategy on it ----
    print("\n" + "-" * 72)
    print("Phase 3.1 — Faithfulness per aggregation strategy")
    print("-" * 72)
    faith = {name: {"del": [], "ins": []} for name in strategies}
    per_direction_corr_pairs: dict[str, list[float]] = {}

    sample_count = 0
    for images, labels in tqdm(test_loader,
                               total=min(args.num_samples, len(test_loader)),
                               desc="Sweep"):
        if sample_count >= args.num_samples:
            break
        img = images[0:1].to(device)

        # 1) Capture caches with analysis mode ON for exactly one forward.
        if hasattr(model, "enable_analysis_mode"):
            model.enable_analysis_mode(store_states=False)
        with torch.no_grad():
            _, analysis = model(img, return_analysis=True)
        pred = int(analysis.predicted_class.item())
        H, W = img.shape[2], img.shape[3]

        # 2) Extract per-block per-direction maps WHILE analysis mode is
        #    still active. The model's per-block SS2DCache references
        #    may become stale after `disable_analysis_mode`, so we read
        #    out the maps eagerly and detach to CPU here.
        block_data: list[tuple[int, dict[str, torch.Tensor]]] = []
        for stage_idx, stage_blocks in enumerate(analysis.stage_caches):
            for cache in stage_blocks:
                per_dir = _per_direction_map(
                    cache, args.method, channel_mode="mean", top_k=8,
                )
                if per_dir:
                    block_data.append((stage_idx, per_dir))

        # 3) Disable analysis mode now that the maps are safely extracted.
        #    The 7 * 42 = ~294 faithfulness forwards run with analysis off,
        #    which is the path the existing comprehensive_evaluation also
        #    takes when computing deletion/insertion AUC.
        if hasattr(model, "disable_analysis_mode"):
            model.disable_analysis_mode()

        # Defensive: if for any reason no blocks produced direction maps,
        # surface that loudly rather than silently skipping all 7
        # strategies (the bug from the previous run).
        if not block_data:
            raise RuntimeError(
                "No per-direction maps were extracted from the analysis "
                "output. The model's analysis-mode caches may not be "
                "populated; check that `enable_analysis_mode(store_states="
                "False)` actually toggles the block caches on this model."
            )

        # --- For each strategy, only the per-block aggregation differs ---
        for name, (dr, lr) in strategies.items():
            per_block_maps: list[torch.Tensor] = []
            depths: list[int] = []
            for stage_idx, per_dir in block_data:
                stk = torch.stack(list(per_dir.values()))
                blk_map = stk.mean(dim=0) if dr == "mean" else stk.max(dim=0).values
                per_block_maps.append(blk_map)
                depths.append(stage_idx)
            final_lo = _aggregate_layer(per_block_maps, lr, depths)
            if final_lo is None:
                continue
            final = F.interpolate(
                final_lo.unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze()
            mn, mx = final.min(), final.max()
            if (mx - mn).abs() < 1e-8:
                sal = torch.zeros_like(final).cpu()
            else:
                sal = ((final - mn) / (mx - mn + 1e-8)).cpu()
            try:
                del_auc = fevaluator.compute_deletion(img, sal, pred)
                ins_auc = fevaluator.compute_insertion(img, sal, pred)
                faith[name]["del"].append(del_auc)
                faith[name]["ins"].append(ins_auc)
            except Exception as e:
                # On the FIRST image only, print the exception so we know
                # what's happening. Silently swallowing exceptions hid the
                # disable-before-extract bug in the previous iteration.
                if sample_count == 0:
                    print(f"  faithfulness exception for strategy {name!r}: {e}")
                continue

        # Per-direction pairwise correlation on the last-stage last block.
        if block_data:
            _, per_dir = block_data[-1]
            keys = sorted(per_dir.keys())
            for a, b in itertools.combinations(keys, 2):
                v1 = per_dir[a].flatten().numpy()
                v2 = per_dir[b].flatten().numpy()
                if v1.std() < 1e-8 or v2.std() < 1e-8:
                    continue
                rho = float(np.corrcoef(v1, v2)[0, 1])
                per_direction_corr_pairs.setdefault(f"{a} vs {b}", []).append(rho)

        sample_count += 1

    # --- 2) summarise faithfulness sweep ---------------------------------
    print(f"\n{'Strategy':<40} {'Del':>8} {'Ins':>8} {'Score':>8}")
    faith_summary = {}
    for name in strategies:
        d, i = faith[name]["del"], faith[name]["ins"]
        if d:
            dm, im = float(np.mean(d)), float(np.mean(i))
            faith_summary[name] = {"deletion": dm, "insertion": im,
                                    "score": im - dm, "n": len(d)}
            print(f"{name:<40} {dm:>8.4f} {im:>8.4f} {im - dm:>+8.4f}")
        else:
            faith_summary[name] = {"deletion": float("nan"), "insertion": float("nan"),
                                    "score": float("nan"), "n": 0}

    # --- 3) summarise per-direction consistency --------------------------
    print("\n" + "-" * 72)
    print("Phase 3.2 — Per-direction pairwise consistency (last-stage block)")
    print("-" * 72)
    print(f"{'Direction pair':<28} {'Mean rho':>10} {'Std':>8} {'n':>6}")
    direction_summary = {}
    for pair, vals in sorted(per_direction_corr_pairs.items()):
        m = float(np.mean(vals))
        s = float(np.std(vals))
        direction_summary[pair] = {"mean": m, "std": s, "n": len(vals)}
        print(f"{pair:<28} {m:>10.4f} {s:>8.4f} {len(vals):>6d}")

    # --- save -----------------------------------------------------------
    payload = {
        "method": args.method,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "aggregation_ablation": faith_summary,
        "per_direction_pairwise_correlation": direction_summary,
        "strategies": {name: list(t) for name, t in strategies.items()},
    }
    (out / "aggregation_ablation.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out / 'aggregation_ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
