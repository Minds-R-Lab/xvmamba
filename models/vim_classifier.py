"""
Vim (Vision Mamba) classifier with controllability-analysis support.

Reference architecture: Zhu et al., ``Vision Mamba: Efficient Visual
Representation Learning with Bidirectional State Space Model'' (ICML 2024).

Architecture (plain, non-hierarchical, faithful to Vim's published design):

    image (B, C, H, W)
        -> 2D convolution patch embed (kernel=patch_size, stride=patch_size)
        -> flatten to token sequence (B, L, d_model) with L = (H/patch_size)^2
        -> add learnable positional embedding
        -> N x VimBlock (each: LN -> BidirectionalSSM1D -> drop -> + -> [LN -> MLP -> +])
        -> final LayerNorm
        -> mean pooling over tokens
        -> Linear classification head

We expose `VMambaAnalysisOutput`-compatible output (`stage_caches`,
`spatial_dims`) so the existing `ControllabilityAnalyzer` works on Vim
checkpoints without modification. Since Vim has a single resolution
throughout, we report it as ``one stage with N blocks''.

Vim-Tiny configuration (matches the paper's Vim-Ti):
    image_size = 224, patch_size = 16, d_model = 192,
    depth = 24 blocks (paper uses 24; we default to 12 for faster training
    on a single GPU and explicitly note this in the manuscript).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .ss2d import SS2DCache
from .vmamba_classifier import VMambaAnalysisOutput
from .vim_blocks import VimBlock


@dataclass
class VimConfig:
    """Configuration for the Vim classifier."""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Model.
    d_model: int = 192
    depth: int = 12          # paper Vim-Ti uses 24; 12 fits a single 3090 better
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # Head.
    num_classes: int = 1000

    # Training stability.
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0   # we keep stochastic depth optional
    mlp_ratio: float = 4.0        # set to 0 for ``pure'' Vim without MLP

    def __post_init__(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})."
            )
        self.grid_h = self.image_size // self.patch_size
        self.grid_w = self.image_size // self.patch_size
        self.num_patches = self.grid_h * self.grid_w


class VimPatchEmbed(nn.Module):
    """1D patch embedding for Vim (ViT-style, no class token)."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid

        # Single conv with stride=patch_size: produces non-overlapping patches.
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `[B, C, H, W]`.
        Returns:
            `[B, L, embed_dim]` with `L = (H/patch_size)^2`.
        """
        B, C, H, W = x.shape
        x = self.proj(x)              # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, L, embed_dim], row-major.
        return x


class VimClassifier(nn.Module):
    """Vim image classifier with controllability-analysis support.

    Implementation notes:
    - No CLS token; we use mean-pool over tokens before the head (matches
      the ``without-CLS'' Vim variant in the paper and parallels VMamba's
      global-average pooling so the comparison with VMamba is clean).
    - Each `VimBlock` exposes an `SS2DCache` with `FORWARD_HORIZONTAL` and
      `BACKWARD_HORIZONTAL` directions, suitable for the existing
      `ControllabilityAnalyzer`.
    - `forward(x, return_analysis=True)` returns
      `(logits, VMambaAnalysisOutput)` so the eval pipeline can stay
      identical to VMamba.
    """

    def __init__(self, config: VimConfig):
        super().__init__()
        self.config = config

        self.patch_embed = VimPatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.d_model,
        )

        # Learnable positional embedding (one per token).
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(config.drop_rate)

        # Stack of Vim blocks. Each block knows the patch grid so its
        # bidirectional SSM cache can produce the correct position-to-index
        # mapping for the analyzer.
        self.blocks = nn.ModuleList([
            VimBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                mlp_ratio=config.mlp_ratio,
                drop_rate=config.drop_rate,
            )
            for _ in range(config.depth)
        ])
        for blk in self.blocks:
            blk.set_grid(config.grid_h, config.grid_w)

        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)

        # Weight init.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Analysis API (mirrors VMambaClassifier).
    # ------------------------------------------------------------------
    def enable_analysis_mode(
        self,
        store_states: bool = False,
        store_features: bool = False,
    ) -> None:
        # The Vim model is plain (single resolution); no inter-stage
        # features to store beyond the final norm output. `store_features`
        # is accepted for API compatibility but is a no-op here.
        del store_features
        for blk in self.blocks:
            blk.enable_analysis_mode(store_states)

    def disable_analysis_mode(self) -> None:
        for blk in self.blocks:
            blk.disable_analysis_mode()

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_analysis: bool = False,
    ):
        """
        Args:
            x: image `[B, C, H, W]`.
            return_analysis: if True, also return a `VMambaAnalysisOutput`.

        Returns:
            `logits [B, num_classes]` (and optionally an analysis output).
        """
        analysis_mode = return_analysis or any(
            blk.bissm._store_for_analysis for blk in self.blocks
        )

        # Patch embed + positional embedding.
        x = self.patch_embed(x)                # [B, L, d_model]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if return_analysis:
            analysis = VMambaAnalysisOutput()
            analysis.spatial_dims.append((self.config.grid_h, self.config.grid_w))
            # We report Vim as a single stage with N blocks.
            stage_caches: List[SS2DCache] = []

        # Forward through blocks.
        for blk in self.blocks:
            x = blk(x)
            if return_analysis:
                stage_caches.append(blk.get_cache())

        # Final norm, pool, head.
        x = self.norm(x)               # [B, L, d_model]
        x = x.mean(dim=1)              # [B, d_model] — global average pool
        logits = self.head(x)          # [B, num_classes]

        if return_analysis:
            analysis.logits = logits
            analysis.probabilities = torch.softmax(logits, dim=-1)
            analysis.predicted_class = logits.argmax(dim=-1)
            analysis.stage_caches.append(stage_caches)  # 1 stage with N blocks
            return logits, analysis

        return logits

    # ------------------------------------------------------------------
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience constructors -----------------------------------------------
def create_vim_tiny(
    num_classes: int = 1000,
    image_size: int = 224,
    in_channels: int = 3,
    depth: int = 12,
    patch_size: int = 16,
) -> VimClassifier:
    """Vim-Tiny configuration. The published Vim-Ti uses depth=24; we
    default to 12 for single-GPU training. Pass `depth=24` to match the
    paper exactly."""
    config = VimConfig(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        d_model=192,
        depth=depth,
        d_state=16,
        expand=2,
        num_classes=num_classes,
    )
    return VimClassifier(config)


def create_vim_medical(
    num_classes: int,
    image_size: int = 224,
    in_channels: int = 3,
    patch_size: int = 16,
    depth: int = 12,
) -> VimClassifier:
    """Vim configuration mirroring the create_vmamba_medical preset --
    smaller depth/d_model for medical-imaging-scale data."""
    config = VimConfig(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        d_model=192,
        depth=depth,
        d_state=16,
        expand=2,
        num_classes=num_classes,
    )
    return VimClassifier(config)


# Optional self-test --------------------------------------------------------
if __name__ == "__main__":
    print("Testing Vim Classifier...")
    model = create_vim_tiny(num_classes=10, image_size=224, depth=4)
    print(f"  Parameters: {model.get_num_params():,}")

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"  logits shape: {logits.shape}")
    assert logits.shape == (2, 10), "logits shape mismatch"

    model.enable_analysis_mode()
    logits, analysis = model(x, return_analysis=True)
    print(f"  spatial_dims: {analysis.spatial_dims}")
    print(f"  number of stages: {len(analysis.stage_caches)}")
    print(f"  blocks in stage 0: {len(analysis.stage_caches[0])}")
    cache0 = analysis.stage_caches[0][0]
    print(f"  block 0 cached dirs: {[d.value for d in cache0.direction_caches.keys()]}")
    print("  OK.")
