"""
Vim (Vision Mamba) block primitives — bidirectional 1D selective SSM.

Reference architecture: Zhu et al., "Vision Mamba: Efficient Visual
Representation Learning with Bidirectional State Space Model" (ICML 2024,
arXiv:2401.09417). Compared to VMamba's 4-direction cross-scan, Vim
processes a single flattened patch sequence with two independent SSMs
(forward and backward) and merges their outputs.

Implementation notes:

- We reuse the existing `FastSelectiveSSM` / `SelectiveSSM` modules so the
  underlying mamba-ssm CUDA kernels (when available) handle the heavy
  lifting. Each Vim block instantiates **two** separate SSM modules, one
  for each scan direction, so the forward and backward branches have
  independent learnable parameters as in the published Vim paper.
- The block exposes an `SS2DCache` populated with the
  `ScanDirection.FORWARD_HORIZONTAL` and `BACKWARD_HORIZONTAL` keys, so
  the existing `ControllabilityAnalyzer` (which iterates over a
  direction-keyed cache) works without modification.
- For a 1D sequence of length `L = H * W` flattened from an `H x W` patch
  grid in row-major order, the position-to-index mappings are:
    FORWARD:  pos_to_idx[i, j] = i * W + j
    BACKWARD: pos_to_idx[i, j] = (L - 1) - (i * W + j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass

from .selective_ssm import SelectiveSSM, SSMCache, MAMBA_CUDA_AVAILABLE
from .ss2d import SS2DCache, ScanDirection

if MAMBA_CUDA_AVAILABLE:
    # FastSelectiveSSM wraps the official mamba-ssm CUDA kernels.
    from .selective_ssm import FastSelectiveSSM


def _make_ssm(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    """Construct an SSM module — fast path if mamba-ssm is available, slow
    Python fallback otherwise. Two instances per Vim block (forward, backward)
    give Vim its independent bidirectional parameters."""
    if MAMBA_CUDA_AVAILABLE:
        return FastSelectiveSSM(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
        )
    return SelectiveSSM(
        d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
    )


class BidirectionalSSM1D(nn.Module):
    """
    Bidirectional 1D selective SSM as used in Vim.

    Takes a flat sequence `x` of shape `[batch, L, d_model]`, processes it
    through two independent SSMs (forward and reversed), and merges the
    outputs by summation (Vim default). The two SSMs have their own
    parameters and are trained jointly.

    Args:
        d_model:   Model dimension.
        d_state:   SSM state dimension `N`.
        d_conv:    Local convolution width inside each SSM.
        expand:    Inner-dimension expansion factor.
        merge_mode: How to merge forward and backward outputs.
                   `"sum"` (Vim default), `"mean"`, or `"concat"` (then
                   projected back to `d_model`).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        merge_mode: str = "sum",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.merge_mode = merge_mode

        # Two independent SSMs.
        self.ssm_forward = _make_ssm(d_model, d_state, d_conv, expand)
        self.ssm_backward = _make_ssm(d_model, d_state, d_conv, expand)

        # Optional concat-then-project for non-sum merge.
        self.merge_proj = (
            nn.Linear(2 * d_model, d_model) if merge_mode == "concat" else None
        )

        # Cache exposed to the controllability analyzer.
        self._cache = SS2DCache()
        self._store_for_analysis = False

        # Spatial dims to fill into the cache (set by the parent Vim block
        # which knows the patch grid).
        self._grid_h = 0
        self._grid_w = 0

    # ------------------------------------------------------------------
    # Analysis API (matches SS2D's surface so analyzer works unchanged).
    # ------------------------------------------------------------------
    def enable_analysis_mode(self, store_states: bool = False) -> None:
        self._store_for_analysis = True
        self.ssm_forward.enable_analysis_mode(store_states)
        self.ssm_backward.enable_analysis_mode(store_states)

    def disable_analysis_mode(self) -> None:
        self._store_for_analysis = False
        self.ssm_forward.disable_analysis_mode()
        self.ssm_backward.disable_analysis_mode()

    def get_cache(self) -> SS2DCache:
        return self._cache

    def set_grid(self, height: int, width: int) -> None:
        """Tell the block what the underlying patch grid is, so we can build
        position-to-index mappings for the analyzer."""
        self._grid_h = height
        self._grid_w = width

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, use_sequential: bool = False) -> torch.Tensor:
        """
        Args:
            x: Tensor `[batch, L, d_model]`.
        Returns:
            Tensor `[batch, L, d_model]`.
        """
        # Forward scan.
        y_fwd = self.ssm_forward(x, use_sequential=use_sequential)

        # Backward scan: reverse sequence, scan, reverse back.
        x_rev = torch.flip(x, dims=[1])
        y_bwd_rev = self.ssm_backward(x_rev, use_sequential=use_sequential)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])

        # Cache: capture (A, B, C) for both directions.
        if self._store_for_analysis:
            self._populate_cache(x.device)

        # Merge.
        if self.merge_mode == "sum":
            y = y_fwd + y_bwd
        elif self.merge_mode == "mean":
            y = 0.5 * (y_fwd + y_bwd)
        elif self.merge_mode == "concat":
            y = torch.cat([y_fwd, y_bwd], dim=-1)
            y = self.merge_proj(y)
        else:
            raise ValueError(f"unknown merge_mode: {self.merge_mode}")

        return y

    # ------------------------------------------------------------------
    def _populate_cache(self, device: torch.device) -> None:
        """Copy forward/backward SSM parameter caches into our `SS2DCache`
        and fill in the position-to-index mappings."""
        H, W = self._grid_h, self._grid_w
        L = H * W if H * W > 0 else None

        def _clone(c: SSMCache) -> SSMCache:
            return SSMCache(
                A_bar=c.A_bar.clone() if c.A_bar is not None else None,
                B_bar=c.B_bar.clone() if c.B_bar is not None else None,
                C=c.C.clone() if c.C is not None else None,
                delta=c.delta.clone() if c.delta is not None else None,
                states=c.states.clone() if c.states is not None else None,
            )

        fwd_cache = _clone(self.ssm_forward.get_cache())
        bwd_cache = _clone(self.ssm_backward.get_cache())

        self._cache.direction_caches[ScanDirection.FORWARD_HORIZONTAL] = fwd_cache
        self._cache.direction_caches[ScanDirection.BACKWARD_HORIZONTAL] = bwd_cache

        if H > 0 and W > 0:
            self._cache.height = H
            self._cache.width = W
            # Forward: row-major. pos_to_idx[i, j] = i*W + j.
            idx = torch.arange(L, device=device).reshape(H, W)
            self._cache.position_to_index[ScanDirection.FORWARD_HORIZONTAL] = idx
            # Backward: reverse row-major. pos_to_idx[i, j] = (L-1) - (i*W + j).
            self._cache.position_to_index[ScanDirection.BACKWARD_HORIZONTAL] = (L - 1) - idx


class VimBlock(nn.Module):
    """
    Single Vim transformer-style block.

    Pre-norm residual structure (matches the published Vim paper):

        x -> LN -> BidirectionalSSM1D -> drop -> + -> y_1
        |_________________________________________|
                                                       y_1 -> LN -> MLP -> drop -> + -> output
                                                                                   |
                                                                          y_1 ____|

    A small MLP after the SSM is optional in the Vim paper; we include it
    by default at `mlp_ratio=4` because (a) it matches the standard
    transformer/ViT recipe and (b) it improves training stability for the
    plain (non-hierarchical) architecture.

    Args:
        d_model:    Token dimension.
        d_state:    SSM state dimension.
        d_conv:     Local convolution width.
        expand:     Inner-dimension expansion factor.
        mlp_ratio:  MLP hidden = `int(d_model * mlp_ratio)`. Set to 0 to
                    disable the MLP entirely (more faithful to plain Vim).
        drop_rate:  Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(d_model)
        self.bissm = BidirectionalSSM1D(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
            merge_mode="sum",
        )
        self.drop1 = nn.Dropout(drop_rate)

        if mlp_ratio > 0:
            self.norm2 = nn.LayerNorm(d_model)
            mlp_hidden = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_hidden),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(mlp_hidden, d_model),
                nn.Dropout(drop_rate),
            )
        else:
            self.norm2 = None
            self.mlp = None

    # ------------------------------------------------------------------
    # Analysis API.
    # ------------------------------------------------------------------
    def enable_analysis_mode(self, store_states: bool = False) -> None:
        self.bissm.enable_analysis_mode(store_states)

    def disable_analysis_mode(self) -> None:
        self.bissm.disable_analysis_mode()

    def get_cache(self) -> SS2DCache:
        return self.bissm.get_cache()

    def set_grid(self, height: int, width: int) -> None:
        self.bissm.set_grid(height, width)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, use_sequential: bool = False) -> torch.Tensor:
        """
        Args:
            x: Tensor `[batch, L, d_model]`.
        Returns:
            Tensor `[batch, L, d_model]`.
        """
        x = x + self.drop1(self.bissm(self.norm1(x), use_sequential=use_sequential))
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x


# Optional self-test --------------------------------------------------------
if __name__ == "__main__":
    print("Testing Vim block primitives...")
    block = VimBlock(d_model=192, d_state=16, expand=2, mlp_ratio=4.0)
    block.set_grid(14, 14)
    block.enable_analysis_mode()

    x = torch.randn(2, 14 * 14, 192)
    y = block(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Output matches input shape: {y.shape == x.shape}")

    cache = block.get_cache()
    print(f"  Cache directions: {[d.value for d in cache.direction_caches.keys()]}")
    for d, c in cache.direction_caches.items():
        if c.A_bar is not None:
            print(f"    {d.value}: A_bar {c.A_bar.shape}, B_bar {c.B_bar.shape}, C {c.C.shape}")
    print(f"  pos_to_idx shape: "
          f"{cache.position_to_index[ScanDirection.FORWARD_HORIZONTAL].shape}")
