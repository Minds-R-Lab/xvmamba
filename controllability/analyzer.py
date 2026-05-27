"""
Controllability-Based Interpretability for Vision State Space Models

This module implements the two controllability analysis methods described
in the X-VMamba paper:

1. Jacobian Method (Section 3.2): Measures influence on aggregated output
   - More general, works with any SSM architecture
   - Computes: sum over j of ||∂y_j/∂u_k||_F
   
2. Gramian Method (Section 3.3): Uses analytical controllability Gramian
   - Faster (closed-form solution)
   - Requires diagonal A matrix (standard in Mamba)
   - Computes: C² * (B² / (1 - A²))

Both methods produce influence scores that quantify how much each input
patch controls the model's internal state dynamics.

Reference: companion paper (under double-anonymous review)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import math

# Import our model components (type hints only, actual import at runtime)
# from .models.ss2d import SS2DCache, ScanDirection, ScanPatterns
# from .models.vmamba_classifier import VMambaAnalysisOutput


class ControllabilityMethod(Enum):
    """Available controllability computation methods."""
    JACOBIAN = "jacobian"
    GRAMIAN = "gramian"


@dataclass
class ControllabilityResult:
    """
    Result of controllability analysis for a single layer/block.
    
    Contains:
    - influence_map: 2D spatial map of influence scores [H, W]
    - per_direction_maps: Influence maps for each scan direction
    - direct_influence: The immediate influence term ||C_k * B_k||
    - propagated_influence: The long-term influence through state transitions
    - metadata: Additional information about the computation
    """
    influence_map: torch.Tensor  # [H, W]
    per_direction_maps: Dict[str, torch.Tensor] = None  # direction -> [H, W]
    direct_influence: torch.Tensor = None  # [H, W]
    propagated_influence: torch.Tensor = None  # [H, W]
    layer_idx: int = -1
    block_idx: int = -1
    method: str = ""
    

@dataclass 
class FullControllabilityAnalysis:
    """
    Complete controllability analysis across all layers.
    
    Structure:
    - layer_results[stage_idx][block_idx] = ControllabilityResult
    - aggregated_map: Single influence map aggregated across all layers
    """
    layer_results: List[List[ControllabilityResult]]
    aggregated_map: torch.Tensor  # [H, W] - original input resolution
    spatial_pyramid: List[torch.Tensor]  # Influence maps at each spatial scale


class JacobianControllability:
    """
    Jacobian-based controllability analysis.
    
    Computes influence scores by measuring how each input affects
    all subsequent outputs through state propagation.
    
    For a diagonal SSM with h_k = A_k h_{k-1} + b_k x_k and y_k = c_k^T h_k,
    the Jacobian controllability index at position k is:
    
        J_k = sum_n |b_{k,n}| * R_{k,n}
    
    where R_{k,n} = sum_{t=k}^{L} |c_{t,n}| * phi_n(t,k) is the total output
    relevance, and phi_n(t,k) = prod_{j=k+1}^{t} a_{j,n} is the state transition.
    
    This decomposes into:
    
        J_k = sum_n |c_{k,n}| * |b_{k,n}|                      [direct]
            + sum_n Q_{k,n}   * |b_{k,n}|                      [propagated]
    
    where Q is the propagated-only auxiliary vector satisfying:
    
        Q_{L,n} = 0
        Q_{k,n} = a_{k+1,n} * (|c_{k+1,n}| + Q_{k+1,n})
    
    CRITICAL implementation details:
    - Score BEFORE update: ensures Q holds only future influence, preventing
      double-counting of the direct term.
    - Use |c_{t,n}| (abs): we sum magnitudes, not signed values.
    - After scoring position k, the update Q = A_k * (|C_k| + Q) prepares the
      propagator for position k-1 using A_k (the transition from h_{k-1} to h_k).
    """
    
    @staticmethod
    def compute_influence_1d(
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute influence scores for a 1D sequence.
        
        Args:
            A_bar: Discretized state matrix [batch, length, d_inner, d_state]
                   Diagonal entries of the state transition; must satisfy
                   0 < A_bar < 1 for stability (enforced by exp(Delta * a)
                   with a < 0).
            B_bar: Discretized input matrix [batch, length, d_inner, d_state]
            C: Output matrix [batch, length, d_state]
            eps: Small value for numerical stability
            
        Returns:
            total_influence: [batch, length] - J_k = direct + propagated
            direct_influence: [batch, length] - ||C_k ⊙ B_k||
            propagated_influence: [batch, length] - ||Q_k ⊙ B_k||
        """
        batch, length, d_inner, d_state = A_bar.shape
        device = A_bar.device
        dtype = A_bar.dtype
        
        # Initialize outputs
        total_influence = torch.zeros(batch, length, device=device, dtype=dtype)
        direct_influence = torch.zeros(batch, length, device=device, dtype=dtype)
        propagated_influence = torch.zeros(batch, length, device=device, dtype=dtype)
        
        # Q = propagation vector, accumulates future output relevance.
        # Q[b, n, d] represents the propagated influence from all positions
        # after k, for state dimension n and inner channel d.
        # Initialized to zero: the last position has no propagated influence.
        # Shape: [batch, d_state, d_inner]
        Q = torch.zeros(batch, d_state, d_inner, device=device, dtype=dtype)
        
        # Backward iteration
        for k in range(length - 1, -1, -1):
            # Get parameters at position k
            A_k = A_bar[:, k]    # [batch, d_inner, d_state]
            B_k = B_bar[:, k]    # [batch, d_inner, d_state]
            C_k = C[:, k]        # [batch, d_state]
            C_k_abs = C_k.abs()  # |c_{k,n}|: use magnitudes throughout
            
            # ---- Step 1: Score position k BEFORE updating Q ----
            # This ensures Q holds only the influence from positions k+1..L,
            # preventing double-counting of the direct term.
            #
            # Channel aggregation: arithmetic mean over the inner channel
            # dimension D, matching Eq. (36) of the manuscript:
            #     S_k^{(dir)} = (1/D) sum_d J_k^{(d)}.
            # (Earlier revisions used torch.norm here, an L2 aggregation;
            # this was changed to .mean() on 2026-05-14 to match the paper
            # and resolve a code-vs-paper consistency issue. See
            # LOGS/01_math.md and LOGS/02_experiments.md for the diagnosis
            # and the before/after numbers.)

            # Direct influence: per-channel CB[b, i] = sum_n |c_{k,n}| * |b_{k,i,n}|.
            # CB[b, i] equals the direct-term J_k^{(d=i),direct} of the manuscript;
            # we mean across channels to obtain a single per-position scalar.
            CB = torch.einsum('bn,bin->bi', C_k_abs, B_k.abs())  # [batch, d_inner]
            direct_inf = CB.mean(dim=-1)  # [batch]

            # Propagated influence: per-channel QB[b, i] = sum_n Q[b, n, i] * |b_{k,i,n}|.
            # Q is non-negative by construction (accumulates |C| * positive A),
            # so taking abs of B prevents sign cancellation; we then mean
            # across channels to match the paper's aggregation.
            QB = torch.einsum('bni,bin->bi', Q, B_k.abs())  # [batch, d_inner]
            prop_inf = QB.mean(dim=-1)  # [batch]
            
            # Store results
            direct_influence[:, k] = direct_inf
            propagated_influence[:, k] = prop_inf
            total_influence[:, k] = direct_inf + prop_inf
            
            # ---- Step 2: Update Q for position k-1 ----
            # Q_{k-1, n, d} = A_k[d, n] * (|C_k[n]| + Q_{k, n, d})
            #
            # A_k is the transition FROM h_{k-1} TO h_k (since h_k = A_k h_{k-1} + ...),
            # so it is the correct decay factor for propagating influence backward
            # from position k to position k-1.
            #
            # |C_k| is added because position k's output contributes to the
            # propagated influence when viewed from position k-1.
            if k > 0:
                # Expand |C_k| to match Q's shape: [batch, d_state] -> [batch, d_state, d_inner]
                C_expanded = C_k_abs.unsqueeze(-1).expand(-1, -1, d_inner)
                
                # A_k: [batch, d_inner, d_state] -> [batch, d_state, d_inner]
                A_k_t = A_k.permute(0, 2, 1)
                
                # Q_{k-1} = A_k * (|C_k| + Q_k)
                Q = A_k_t * (C_expanded + Q)
        
        return total_influence, direct_influence, propagated_influence
    
    @staticmethod
    def compute_from_cache(
        cache,  # SS2DCache
        aggregate_directions: bool = True,
    ) -> ControllabilityResult:
        """
        Compute controllability from an SS2D cache.
        
        Args:
            cache: SS2DCache containing per-direction SSM parameters
            aggregate_directions: Whether to average across scan directions
            
        Returns:
            ControllabilityResult with influence maps
        """
        height = cache.height
        width = cache.width
        
        # Import here to avoid circular imports
        from models.ss2d import ScanDirection, ScanPatterns
        
        per_direction_maps = {}
        aggregated_map = None
        
        for direction, ssm_cache in cache.direction_caches.items():
            if ssm_cache.A_bar is None:
                continue
            
            # Compute 1D influence scores
            total_inf, direct_inf, prop_inf = JacobianControllability.compute_influence_1d(
                A_bar=ssm_cache.A_bar,
                B_bar=ssm_cache.B_bar,
                C=ssm_cache.C,
            )
            
            # Average over batch dimension
            total_inf = total_inf.mean(dim=0)  # [length]
            
            # Get position mapping for this direction
            pos_mapping = cache.position_to_index.get(direction)
            
            if pos_mapping is not None:
                # Unscan: convert 1D scores back to 2D
                # pos_mapping[i, j] = sequence_index
                influence_2d = torch.zeros(height, width, device=total_inf.device)
                for i in range(height):
                    for j in range(width):
                        seq_idx = pos_mapping[i, j].item()
                        influence_2d[i, j] = total_inf[seq_idx]
            else:
                # Fallback: assume forward horizontal scan
                influence_2d = total_inf.reshape(height, width)
            
            per_direction_maps[direction.value if hasattr(direction, 'value') else str(direction)] = influence_2d
            
            if aggregated_map is None:
                aggregated_map = influence_2d.clone()
            else:
                aggregated_map = aggregated_map + influence_2d
        
        # Average across directions
        if aggregate_directions and len(per_direction_maps) > 0:
            aggregated_map = aggregated_map / len(per_direction_maps)
        
        return ControllabilityResult(
            influence_map=aggregated_map,
            per_direction_maps=per_direction_maps,
            method="jacobian",
        )


class GramianControllability:
    """
    Gramian-based controllability analysis.
    
    Uses the analytical solution to the controllability Gramian for
    diagonal state-space models.
    
    For diagonal A, the Gramian has closed-form solution:
        W_c = B² / (1 - A²)
    
    The influence score weighted by observability is:
        I_i(k) = C_i² * W_c,i(k)
              = C_i² * B_i² / (1 - A_i² + ε)
    
    This is faster than the Jacobian method but requires diagonal A.
    """
    
    @staticmethod
    def compute_gramian_diagonal(
        A_bar: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute the diagonal controllability Gramian factor.
        
        For diagonal A: W_c = B² / (1 - A²)
        This returns the denominator: 1 / (1 - A²)
        
        Args:
            A_bar: Discretized diagonal state matrix [batch, length, d_inner, d_state]
            eps: Numerical stability constant
            
        Returns:
            gramian_factor: [batch, length, d_inner, d_state]
        """
        # W_c = B² / (1 - A²)
        # We compute 1 / (1 - A²) here, multiply by B² later
        A_squared = A_bar ** 2
        denominator = 1.0 - A_squared + eps
        gramian_factor = 1.0 / denominator
        
        return gramian_factor
    
    @staticmethod
    def compute_influence_1d(
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute Gramian-based influence scores for a 1D sequence.
        
        Influence = sum over state dims of: C² * B² / (1 - A²)
        
        Args:
            A_bar: [batch, length, d_inner, d_state]
            B_bar: [batch, length, d_inner, d_state]
            C: [batch, length, d_state]
            eps: Numerical stability
            
        Returns:
            influence: [batch, length]
        """
        batch, length, d_inner, d_state = A_bar.shape
        
        # Compute Gramian factor: 1 / (1 - A²)
        gramian_factor = GramianControllability.compute_gramian_diagonal(A_bar, eps)
        
        # Compute B² (per position, per channel, per state dim)
        B_squared = B_bar ** 2  # [batch, length, d_inner, d_state]
        
        # Controllability: W_c = B² * gramian_factor
        W_c = B_squared * gramian_factor  # [batch, length, d_inner, d_state]
        
        # Weight by observability: C²
        # C: [batch, length, d_state]
        # Expand C to match W_c dimensions
        C_squared = C ** 2  # [batch, length, d_state]
        C_squared = C_squared.unsqueeze(2)  # [batch, length, 1, d_state]
        
        # Influence per state: I = C² * W_c
        I = C_squared * W_c  # [batch, length, d_inner, d_state]
        
        # Sum over state dimensions, average over inner dimensions
        influence = I.sum(dim=-1).mean(dim=-1)  # [batch, length]
        
        return influence
    
    @staticmethod
    def compute_from_cache(
        cache,  # SS2DCache
        aggregate_directions: bool = True,
    ) -> ControllabilityResult:
        """
        Compute Gramian controllability from SS2D cache.
        
        Args:
            cache: SS2DCache with SSM parameters
            aggregate_directions: Average across scan directions
            
        Returns:
            ControllabilityResult
        """
        height = cache.height
        width = cache.width
        
        from models.ss2d import ScanDirection, ScanPatterns
        
        per_direction_maps = {}
        aggregated_map = None
        
        for direction, ssm_cache in cache.direction_caches.items():
            if ssm_cache.A_bar is None:
                continue
            
            # Compute 1D influence
            influence_1d = GramianControllability.compute_influence_1d(
                A_bar=ssm_cache.A_bar,
                B_bar=ssm_cache.B_bar,
                C=ssm_cache.C,
            )
            
            # Average over batch
            influence_1d = influence_1d.mean(dim=0)  # [length]
            
            # Convert to 2D
            pos_mapping = cache.position_to_index.get(direction)
            
            if pos_mapping is not None:
                influence_2d = torch.zeros(height, width, device=influence_1d.device)
                for i in range(height):
                    for j in range(width):
                        seq_idx = pos_mapping[i, j].item()
                        influence_2d[i, j] = influence_1d[seq_idx]
            else:
                influence_2d = influence_1d.reshape(height, width)
            
            dir_key = direction.value if hasattr(direction, 'value') else str(direction)
            per_direction_maps[dir_key] = influence_2d
            
            if aggregated_map is None:
                aggregated_map = influence_2d.clone()
            else:
                aggregated_map = aggregated_map + influence_2d
        
        if aggregate_directions and len(per_direction_maps) > 0:
            aggregated_map = aggregated_map / len(per_direction_maps)
        
        return ControllabilityResult(
            influence_map=aggregated_map,
            per_direction_maps=per_direction_maps,
            method="gramian",
        )


class ControllabilityAnalyzer:
    """
    Main interface for controllability analysis.
    
    This class provides a unified API for analyzing Vision Mamba models
    using either Jacobian or Gramian methods.
    
    Usage:
        analyzer = ControllabilityAnalyzer(method='jacobian')
        
        model.enable_analysis_mode()
        logits, analysis = model(image, return_analysis=True)
        
        results = analyzer.analyze(analysis)
        influence_map = results.aggregated_map
    """
    
    def __init__(
        self,
        method: Union[ControllabilityMethod, str] = ControllabilityMethod.JACOBIAN,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """
        Initialize analyzer.
        
        Args:
            method: 'jacobian' or 'gramian'
            normalize: Whether to normalize influence maps to [0, 1]
            eps: Numerical stability constant
        """
        if isinstance(method, str):
            method = ControllabilityMethod(method)
        
        self.method = method
        self.normalize = normalize
        self.eps = eps
    
    def _normalize_map(self, influence_map: torch.Tensor) -> torch.Tensor:
        """Normalize influence map to [0, 1]."""
        min_val = influence_map.min()
        max_val = influence_map.max()
        
        if max_val - min_val < self.eps:
            return torch.zeros_like(influence_map)
        
        return (influence_map - min_val) / (max_val - min_val + self.eps)
    
    def analyze_block(
        self,
        cache,  # SS2DCache
        stage_idx: int = -1,
        block_idx: int = -1,
    ) -> ControllabilityResult:
        """
        Analyze a single VSSM block.
        
        Args:
            cache: SS2DCache from the block
            stage_idx: Stage index (for metadata)
            block_idx: Block index (for metadata)
            
        Returns:
            ControllabilityResult for this block
        """
        if self.method == ControllabilityMethod.JACOBIAN:
            result = JacobianControllability.compute_from_cache(cache)
        else:
            result = GramianControllability.compute_from_cache(cache)
        
        result.layer_idx = stage_idx
        result.block_idx = block_idx
        
        if self.normalize and result.influence_map is not None:
            result.influence_map = self._normalize_map(result.influence_map)
            
            if result.per_direction_maps is not None:
                for key in result.per_direction_maps:
                    result.per_direction_maps[key] = self._normalize_map(
                        result.per_direction_maps[key]
                    )
        
        return result
    
    def analyze(
        self,
        analysis_output,  # VMambaAnalysisOutput
        aggregate_strategy: str = "mean",
    ) -> FullControllabilityAnalysis:
        """
        Analyze complete model output.
        
        Args:
            analysis_output: VMambaAnalysisOutput from model forward pass
            aggregate_strategy: How to combine layer results ('mean', 'max', 'last')
            
        Returns:
            FullControllabilityAnalysis with all results
        """
        layer_results = []
        spatial_pyramid = []
        
        # Process each stage
        for stage_idx, stage_caches in enumerate(analysis_output.stage_caches):
            stage_results = []
            
            # Process each block in the stage
            for block_idx, cache in enumerate(stage_caches):
                result = self.analyze_block(
                    cache,
                    stage_idx=stage_idx,
                    block_idx=block_idx,
                )
                stage_results.append(result)
            
            layer_results.append(stage_results)
            
            # Store influence map at this spatial scale
            if stage_results and stage_results[-1].influence_map is not None:
                spatial_pyramid.append(stage_results[-1].influence_map)
        
        # Aggregate across all layers
        aggregated_map = self._aggregate_layers(layer_results, aggregate_strategy)
        
        return FullControllabilityAnalysis(
            layer_results=layer_results,
            aggregated_map=aggregated_map,
            spatial_pyramid=spatial_pyramid,
        )
    
    def _aggregate_layers(
        self,
        layer_results: List[List[ControllabilityResult]],
        strategy: str = "mean",
    ) -> torch.Tensor:
        """
        Aggregate influence maps across layers.
        
        Handles different spatial resolutions by upsampling to largest size.
        """
        # Collect all influence maps
        all_maps = []
        max_h, max_w = 0, 0
        
        for stage_results in layer_results:
            for result in stage_results:
                if result.influence_map is not None:
                    h, w = result.influence_map.shape
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
                    all_maps.append(result.influence_map)
        
        if not all_maps:
            return None
        
        # Upsample all maps to maximum resolution
        upsampled_maps = []
        for influence_map in all_maps:
            if influence_map.shape[0] != max_h or influence_map.shape[1] != max_w:
                # Upsample using bilinear interpolation
                upsampled = F.interpolate(
                    influence_map.unsqueeze(0).unsqueeze(0),
                    size=(max_h, max_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze()
            else:
                upsampled = influence_map
            upsampled_maps.append(upsampled)
        
        # Stack and aggregate
        stacked = torch.stack(upsampled_maps, dim=0)
        
        if strategy == "mean":
            aggregated = stacked.mean(dim=0)
        elif strategy == "max":
            aggregated = stacked.max(dim=0)[0]
        elif strategy == "last":
            aggregated = stacked[-1]
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
        
        if self.normalize:
            aggregated = self._normalize_map(aggregated)
        
        return aggregated


# Utility functions for visualization
def overlay_heatmap(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    alpha: float = 0.5,
    colormap: str = 'jet',
) -> torch.Tensor:
    """
    Overlay influence heatmap on original image.
    
    Args:
        image: Original image [C, H, W] or [H, W, C], values in [0, 1]
        heatmap: Influence map [H, W], values in [0, 1]
        alpha: Blending factor
        colormap: Matplotlib colormap name
        
    Returns:
        blended: RGB image with heatmap overlay [H, W, 3]
    """
    # Move everything to CPU for visualization
    image = image.cpu() if image.is_cuda else image
    heatmap = heatmap.cpu() if heatmap.is_cuda else heatmap
    
    # Normalize heatmap to [0, 1]
    if heatmap.max() > 1 or heatmap.min() < 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Resize heatmap to match image if needed
    if image.dim() == 3:
        if image.shape[0] in [1, 3]:  # [C, H, W]
            img_h, img_w = image.shape[1], image.shape[2]
            image = image.permute(1, 2, 0)  # -> [H, W, C]
        else:  # [H, W, C]
            img_h, img_w = image.shape[0], image.shape[1]
    else:
        img_h, img_w = image.shape[0], image.shape[1]
    
    if heatmap.shape[0] != img_h or heatmap.shape[1] != img_w:
        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze()
    
    # Convert heatmap to RGB using simple colormap (red = high, blue = low)
    # This is a simple jet-like colormap implementation
    heatmap_rgb = torch.zeros(img_h, img_w, 3, dtype=image.dtype)
    
    # Red channel: increases with value
    heatmap_rgb[..., 0] = torch.clamp(1.5 - torch.abs(heatmap - 0.75) * 4, 0, 1)
    # Green channel: peak in middle
    heatmap_rgb[..., 1] = torch.clamp(1.5 - torch.abs(heatmap - 0.5) * 4, 0, 1)
    # Blue channel: decreases with value
    heatmap_rgb[..., 2] = torch.clamp(1.5 - torch.abs(heatmap - 0.25) * 4, 0, 1)
    
    # Handle grayscale images
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    
    # Blend
    blended = (1 - alpha) * image + alpha * heatmap_rgb
    blended = torch.clamp(blended, 0, 1)
    
    return blended


def get_top_k_patches(
    influence_map: torch.Tensor,
    k: int = 5,
    patch_size: int = 16,
) -> List[Tuple[int, int, float]]:
    """
    Get the top-k most influential patches.
    
    Args:
        influence_map: 2D influence scores [H, W]
        k: Number of top patches to return
        patch_size: Size of each patch (for coordinate computation)
        
    Returns:
        List of (row_idx, col_idx, influence_score) tuples
    """
    # Flatten and get top-k indices
    flat = influence_map.flatten()
    values, indices = torch.topk(flat, k)
    
    # Convert to 2D coordinates
    height, width = influence_map.shape
    results = []
    for val, idx in zip(values, indices):
        row = idx.item() // width
        col = idx.item() % width
        results.append((row, col, val.item()))
    
    return results


# Test the implementation
if __name__ == "__main__":
    print("Testing Controllability Analysis...")
    print("=" * 60)
    
    # Create synthetic data that mimics SSM cache structure
    batch, length, d_inner, d_state = 2, 16, 128, 16
    
    # Synthetic SSM parameters
    A_bar = torch.sigmoid(torch.randn(batch, length, d_inner, d_state)) * 0.99  # Stable
    B_bar = torch.randn(batch, length, d_inner, d_state) * 0.1
    C = torch.randn(batch, length, d_state)
    
    # Test Jacobian method
    print("\nTesting Jacobian method...")
    total_j, direct_j, prop_j = JacobianControllability.compute_influence_1d(A_bar, B_bar, C)
    print(f"  Total influence shape: {total_j.shape}")
    print(f"  Direct influence range: [{direct_j.min():.4f}, {direct_j.max():.4f}]")
    print(f"  Propagated influence range: [{prop_j.min():.4f}, {prop_j.max():.4f}]")
    
    # Verify boundary case: at last position, propagated should be 0
    print(f"\n  Boundary check (k=L-1):")
    print(f"    propagated[last] = {prop_j[:, -1].mean():.6f}  (should be 0.0)")
    print(f"    direct[last]     = {direct_j[:, -1].mean():.6f}  (should be > 0)")
    
    # Test Gramian method
    print("\nTesting Gramian method...")
    influence_g = GramianControllability.compute_influence_1d(A_bar, B_bar, C)
    print(f"  Influence shape: {influence_g.shape}")
    print(f"  Influence range: [{influence_g.min():.4f}, {influence_g.max():.4f}]")
    
    # Compare methods
    print("\nComparing methods...")
    total_j_mean = total_j.mean(dim=0)
    influence_g_mean = influence_g.mean(dim=0)
    correlation = torch.corrcoef(torch.stack([total_j_mean, influence_g_mean]))[0, 1]
    print(f"  Correlation between methods: {correlation:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")