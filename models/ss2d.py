"""
2D Selective Scan (SS2D) Module for Vision Mamba

This module implements the four-directional scanning strategy used in VMamba
to convert 2D image patches into 1D sequences while preserving spatial relationships.

Key Design Decisions:
1. Explicit tracking of patch-to-sequence mapping for each direction
2. Support for both forward pass and inverse mapping (for visualization)
3. Separate storage of per-direction SSM parameters for analysis
4. Uses fast CUDA kernels when mamba-ssm is available

The four scan directions are:
1. Forward horizontal (left-to-right, top-to-bottom)
2. Backward horizontal (right-to-left, bottom-to-top)
3. Forward vertical/transposed (top-to-bottom, left-to-right)
4. Backward vertical/transposed (bottom-to-top, right-to-left)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from .selective_ssm import SelectiveSSM, SSMCache, MAMBA_CUDA_AVAILABLE

# Import fast SSM if available
if MAMBA_CUDA_AVAILABLE:
    from .selective_ssm import FastSelectiveSSM


class ScanDirection(Enum):
    """Enumeration of the four scan directions."""
    FORWARD_HORIZONTAL = "fwd_h"      # Left-to-right, top-to-bottom
    BACKWARD_HORIZONTAL = "bwd_h"     # Right-to-left, bottom-to-top  
    FORWARD_VERTICAL = "fwd_v"        # Top-to-bottom, left-to-right
    BACKWARD_VERTICAL = "bwd_v"       # Bottom-to-top, right-to-left


@dataclass
class SS2DCache:
    """
    Cache for SS2D analysis containing per-direction SSM parameters.
    
    This allows us to analyze how each scanning direction contributes
    to the model's understanding of spatial relationships.
    """
    # Per-direction SSM caches
    direction_caches: Dict[ScanDirection, SSMCache] = field(default_factory=dict)
    
    # Patch grid dimensions
    height: int = 0
    width: int = 0
    
    # Mapping from 2D position to 1D sequence index for each direction
    # shape: [height, width] -> sequence_index
    position_to_index: Dict[ScanDirection, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        """Clear all cached values."""
        for cache in self.direction_caches.values():
            cache.clear()
        self.direction_caches.clear()
        self.position_to_index.clear()


class ScanPatterns:
    """
    Utility class for generating and managing scan patterns.
    
    This class provides methods to:
    1. Generate scan orders for each direction
    2. Create index mappings between 2D and 1D
    3. Rearrange tensors according to scan patterns
    """
    
    @staticmethod
    def get_scan_indices(
        height: int, 
        width: int, 
        direction: ScanDirection,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate scan indices for a given direction.
        
        Args:
            height: Number of rows in patch grid
            width: Number of columns in patch grid
            direction: Scanning direction
            device: Device for tensor creation
            
        Returns:
            forward_indices: Indices to convert 2D -> 1D [length]
            inverse_indices: Indices to convert 1D -> 2D [length]
        """
        length = height * width
        
        if direction == ScanDirection.FORWARD_HORIZONTAL:
            # Standard row-major order: (0,0), (0,1), ..., (0,W-1), (1,0), ...
            forward_indices = torch.arange(length)
            
        elif direction == ScanDirection.BACKWARD_HORIZONTAL:
            # Reverse of forward horizontal
            forward_indices = torch.arange(length - 1, -1, -1)
            
        elif direction == ScanDirection.FORWARD_VERTICAL:
            # Column-major order: (0,0), (1,0), ..., (H-1,0), (0,1), ...
            indices_2d = torch.arange(length).reshape(height, width)
            forward_indices = indices_2d.T.flatten()
            
        elif direction == ScanDirection.BACKWARD_VERTICAL:
            # Reverse of forward vertical
            indices_2d = torch.arange(length).reshape(height, width)
            forward_indices = indices_2d.T.flatten().flip(0)
        
        else:
            raise ValueError(f"Unknown scan direction: {direction}")
        
        # Compute inverse indices
        inverse_indices = torch.argsort(forward_indices)
        
        if device is not None:
            forward_indices = forward_indices.to(device)
            inverse_indices = inverse_indices.to(device)
        
        return forward_indices, inverse_indices
    
    @staticmethod
    def create_position_mapping(
        height: int,
        width: int,
        direction: ScanDirection,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Create a 2D tensor mapping (i, j) position to sequence index.
        
        Args:
            height: Number of rows
            width: Number of columns
            direction: Scanning direction
            device: Device for tensor
            
        Returns:
            mapping: [height, width] tensor where mapping[i,j] = sequence_index
        """
        forward_indices, _ = ScanPatterns.get_scan_indices(
            height, width, direction, device
        )
        
        # Create inverse mapping: sequence_index -> 2D position
        mapping = torch.zeros(height, width, dtype=torch.long, device=device)
        
        for seq_idx, flat_idx in enumerate(forward_indices):
            i = flat_idx // width
            j = flat_idx % width
            mapping[i, j] = seq_idx
        
        return mapping
    
    @staticmethod
    def scan_2d_to_1d(
        x: torch.Tensor,
        direction: ScanDirection
    ) -> torch.Tensor:
        """
        Convert 2D patch grid to 1D sequence according to scan direction.
        
        Args:
            x: Input tensor [batch, height, width, channels]
            direction: Scanning direction
            
        Returns:
            x_seq: Scanned sequence [batch, length, channels]
        """
        batch, height, width, channels = x.shape
        device = x.device
        
        forward_indices, _ = ScanPatterns.get_scan_indices(
            height, width, direction, device
        )
        
        # Flatten spatial dimensions
        x_flat = x.reshape(batch, height * width, channels)
        
        # Reorder according to scan direction
        x_seq = x_flat[:, forward_indices, :]
        
        return x_seq
    
    @staticmethod
    def unscan_1d_to_2d(
        x_seq: torch.Tensor,
        height: int,
        width: int,
        direction: ScanDirection
    ) -> torch.Tensor:
        """
        Convert 1D sequence back to 2D patch grid (inverse of scan_2d_to_1d).
        
        Args:
            x_seq: Sequence tensor [batch, length, channels]
            height: Original height
            width: Original width
            direction: Scanning direction used
            
        Returns:
            x: 2D tensor [batch, height, width, channels]
        """
        batch, length, channels = x_seq.shape
        device = x_seq.device
        
        _, inverse_indices = ScanPatterns.get_scan_indices(
            height, width, direction, device
        )
        
        # Reorder back to original spatial order
        x_flat = x_seq[:, inverse_indices, :]
        
        # Reshape to 2D
        x = x_flat.reshape(batch, height, width, channels)
        
        return x


class SS2D(nn.Module):
    """
    2D Selective Scan module for Vision Mamba.
    
    This module processes 2D patch grids by:
    1. Scanning the patches in four directions
    2. Processing each direction through a separate SSM
    3. Merging the outputs from all directions
    
    The multi-directional scanning allows the model to capture
    spatial relationships in multiple orientations.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Local convolution width
        expand: Expansion factor
        merge_mode: How to combine direction outputs ('sum', 'mean', 'concat')
        use_fast_path: Use CUDA kernels if available (default True)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        merge_mode: str = "mean",
        use_fast_path: bool = True,
        **ssm_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.merge_mode = merge_mode
        self.use_fast_path = use_fast_path and MAMBA_CUDA_AVAILABLE
        
        # Direction-specific layer norms (optional, can help with training)
        self.direction_norms = nn.ModuleDict({
            d.value: nn.LayerNorm(d_model) 
            for d in ScanDirection
        })
        
        # Choose SSM implementation
        if self.use_fast_path:
            # Use fast CUDA implementation
            self.ssm = FastSelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Use pure Python implementation (slow but works everywhere)
            self.ssm = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                **ssm_kwargs
            )
        
        # If concat mode, need projection back to d_model
        if merge_mode == "concat":
            self.merge_proj = nn.Linear(d_model * 4, d_model)
        else:
            self.merge_proj = None
        
        # Cache for analysis
        self._cache = SS2DCache()
        self._store_for_analysis = False
        
        # Store scan patterns (will be computed on first forward)
        self._scan_patterns_cached = False
        self._cached_height = None
        self._cached_width = None
    
    def enable_analysis_mode(self, store_states: bool = False):
        """Enable storage of intermediate values for analysis."""
        self._store_for_analysis = True
        self.ssm.enable_analysis_mode(store_states)
    
    def disable_analysis_mode(self):
        """Disable storage to save memory."""
        self._store_for_analysis = False
        self.ssm.disable_analysis_mode()
        self._cache.clear()
    
    def get_cache(self) -> SS2DCache:
        """Get cached values for analysis."""
        return self._cache
    
    def _process_direction(
        self,
        x: torch.Tensor,
        direction: ScanDirection,
        use_sequential: bool = False
    ) -> torch.Tensor:
        """
        Process patches through SSM for one scan direction.
        
        Args:
            x: Input [batch, height, width, d_model]
            direction: Scanning direction
            use_sequential: Use sequential scan for state storage
            
        Returns:
            y: Output [batch, height, width, d_model]
        """
        batch, height, width, _ = x.shape
        
        # Apply direction-specific normalization
        x_norm = self.direction_norms[direction.value](x)
        
        # Scan 2D -> 1D
        x_seq = ScanPatterns.scan_2d_to_1d(x_norm, direction)
        
        # Process through SSM
        y_seq = self.ssm(x_seq, use_sequential=use_sequential)
        
        # Store cache if in analysis mode
        if self._store_for_analysis:
            self._cache.direction_caches[direction] = SSMCache(
                A_bar=self.ssm.get_cache().A_bar.clone() if self.ssm.get_cache().A_bar is not None else None,
                B_bar=self.ssm.get_cache().B_bar.clone() if self.ssm.get_cache().B_bar is not None else None,
                C=self.ssm.get_cache().C.clone() if self.ssm.get_cache().C is not None else None,
                delta=self.ssm.get_cache().delta.clone() if self.ssm.get_cache().delta is not None else None,
                states=self.ssm.get_cache().states.clone() if self.ssm.get_cache().states is not None else None,
            )
            
            # Store position mapping
            self._cache.position_to_index[direction] = ScanPatterns.create_position_mapping(
                height, width, direction, x.device
            )
        
        # Unscan 1D -> 2D
        y = ScanPatterns.unscan_1d_to_2d(y_seq, height, width, direction)
        
        return y
    
    def forward(
        self,
        x: torch.Tensor,
        use_sequential: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through SS2D.
        
        Args:
            x: Input tensor [batch, height, width, d_model]
            use_sequential: Use sequential scan (slower but stores states)
            
        Returns:
            y: Output tensor [batch, height, width, d_model]
        """
        batch, height, width, d_model = x.shape
        
        # Store dimensions in cache
        if self._store_for_analysis:
            self._cache.height = height
            self._cache.width = width
        
        # Process all four directions
        outputs = []
        for direction in ScanDirection:
            y_dir = self._process_direction(x, direction, use_sequential)
            outputs.append(y_dir)
        
        # Merge outputs from all directions
        if self.merge_mode == "sum":
            y = sum(outputs)
        elif self.merge_mode == "mean":
            y = sum(outputs) / len(outputs)
        elif self.merge_mode == "concat":
            y = torch.cat(outputs, dim=-1)  # [batch, H, W, 4*d_model]
            y = self.merge_proj(y)  # [batch, H, W, d_model]
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")
        
        return y


class VSSMBlock(nn.Module):
    """
    Vision State Space Model Block.
    
    Complete block with:
    - SS2D (2D selective scan)
    - LayerNorm
    - MLP
    - Residual connections
    
    Structure:
        x -> LN -> SS2D -> + -> LN -> MLP -> + -> output
        |__________________|   |_______________|
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        **ssm_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # SS2D branch
        self.norm1 = nn.LayerNorm(d_model)
        self.ss2d = SS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **ssm_kwargs
        )
        self.drop1 = nn.Dropout(drop_rate)
        
        # MLP branch
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(drop_rate),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        use_sequential: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, height, width, d_model]
            use_sequential: Use sequential scan for analysis
            
        Returns:
            Output [batch, height, width, d_model]
        """
        # SS2D branch with residual
        x = x + self.drop1(self.ss2d(self.norm1(x), use_sequential))
        
        # MLP branch with residual
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def enable_analysis_mode(self, store_states: bool = False):
        self.ss2d.enable_analysis_mode(store_states)
    
    def disable_analysis_mode(self):
        self.ss2d.disable_analysis_mode()
    
    def get_cache(self) -> SS2DCache:
        return self.ss2d.get_cache()


# Test the implementation
if __name__ == "__main__":
    # Test scan patterns
    print("Testing scan patterns...")
    height, width = 4, 4
    
    for direction in ScanDirection:
        mapping = ScanPatterns.create_position_mapping(height, width, direction)
        print(f"\n{direction.value}:")
        print(mapping)
    
    # Test SS2D
    print("\n\nTesting SS2D...")
    batch_size = 2
    d_model = 64
    
    model = SS2D(d_model=d_model, d_state=16)
    model.enable_analysis_mode(store_states=True)
    
    x = torch.randn(batch_size, height, width, d_model)
    y = model(x, use_sequential=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    cache = model.get_cache()
    print(f"\nCached directions: {list(cache.direction_caches.keys())}")
    
    for direction, ssm_cache in cache.direction_caches.items():
        print(f"\n{direction.value}:")
        print(f"  A_bar shape: {ssm_cache.A_bar.shape}")
        print(f"  B_bar shape: {ssm_cache.B_bar.shape}")
        print(f"  C shape: {ssm_cache.C.shape}")
