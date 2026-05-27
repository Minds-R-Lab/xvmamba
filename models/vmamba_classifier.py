"""
Vision Mamba Classifier with Full Interpretability Support

This module implements a complete Vision Mamba model for image classification
with built-in support for controllability analysis at every layer.

Architecture:
    Input Image -> Patch Embedding -> [VSSM Block] x N -> Global Pool -> Classifier

Key Features:
1. Hierarchical structure with configurable depth
2. Per-layer access to SSM parameters for controllability analysis
3. Support for both training and analysis modes
4. Configurable for different image sizes and patch sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .ss2d import SS2D, VSSMBlock, SS2DCache, ScanDirection


@dataclass
class VMambaConfig:
    """Configuration for Vision Mamba model."""
    
    # Image parameters
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    
    # Model parameters
    d_model: int = 96              # Base model dimension
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])  # Blocks per stage
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])  # Dims per stage
    
    # SSM parameters
    d_state: int = 16              # State dimension
    d_conv: int = 4                # Convolution width
    expand: int = 2                # Expansion factor
    
    # Classification
    num_classes: int = 1000
    
    # Training
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1    # Stochastic depth
    
    # Analysis
    store_intermediates: bool = False
    
    def __post_init__(self):
        # Compute derived values
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_height = self.image_size // self.patch_size
        self.patch_width = self.image_size // self.patch_size
        self.num_stages = len(self.depths)


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    Uses a single convolution with stride equal to patch size
    to efficiently extract non-overlapping patches.
    
    Args:
        image_size: Input image size (assumed square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_height = image_size // patch_size
        self.patch_width = image_size // patch_size
        
        # Patch extraction via convolution
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input image [batch, channels, height, width]
            
        Returns:
            patches: [batch, H_patches, W_patches, embed_dim]
        """
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P]
        x = self.proj(x)
        
        # [B, embed_dim, H/P, W/P] -> [B, H/P, W/P, embed_dim]
        x = x.permute(0, 2, 3, 1)
        
        x = self.norm(x)
        
        return x


class PatchMerging(nn.Module):
    """
    Merge patches to reduce spatial resolution and increase channels.
    
    Combines 2x2 patches into 1 patch, similar to pooling but learned.
    This is used between stages to create hierarchical features.
    
    Handles odd dimensions by padding before merging.
    
    Args:
        input_dim: Input channel dimension
        output_dim: Output channel dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear projection from 4*input_dim to output_dim
        self.reduction = nn.Linear(4 * input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge 2x2 patches.
        
        Args:
            x: Input [batch, H, W, C]
            
        Returns:
            Output [batch, ceil(H/2), ceil(W/2), output_dim]
        """
        batch, height, width, channels = x.shape
        
        # Pad if dimensions are odd
        pad_h = height % 2
        pad_w = width % 2
        
        if pad_h or pad_w:
            # Pad with zeros on the right and bottom
            # x shape: [B, H, W, C] -> need to pad H and W dimensions
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # (C_left, C_right, W_left, W_right, H_left, H_right)
            height = height + pad_h
            width = width + pad_w
        
        # Reshape to combine 2x2 patches
        # [B, H, W, C] -> [B, H/2, 2, W/2, 2, C] -> [B, H/2, W/2, 4*C]
        x = x.reshape(batch, height // 2, 2, width // 2, 2, channels)
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, H/2, W/2, 2, 2, C]
        x = x.reshape(batch, height // 2, width // 2, 4 * channels)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class VSSMStage(nn.Module):
    """
    A stage of VSSM blocks with optional downsampling.
    
    Each stage contains multiple VSSM blocks operating at the same resolution,
    followed by optional patch merging for downsampling.
    
    Args:
        depth: Number of VSSM blocks in this stage
        d_model: Model dimension
        d_state: SSM state dimension
        downsample: Whether to downsample at the end
        output_dim: Output dimension after downsampling
    """
    
    def __init__(
        self,
        depth: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        downsample: bool = True,
        output_dim: Optional[int] = None,
        drop_rate: float = 0.0,
        drop_path_rates: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.depth = depth
        self.d_model = d_model
        self.downsample = downsample
        
        # Create VSSM blocks
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth
        
        self.blocks = nn.ModuleList([
            VSSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_rate=drop_rate,
            )
            for i in range(depth)
        ])
        
        # Optional downsampling
        if downsample and output_dim is not None:
            self.merge = PatchMerging(d_model, output_dim)
        else:
            self.merge = None
    
    def forward(
        self,
        x: torch.Tensor,
        use_sequential: bool = False
    ) -> torch.Tensor:
        """
        Forward through all blocks in stage.
        
        Args:
            x: Input [batch, H, W, d_model]
            use_sequential: Use sequential scan for analysis
            
        Returns:
            Output tensor
        """
        for block in self.blocks:
            x = block(x, use_sequential=use_sequential)
        
        if self.merge is not None:
            x = self.merge(x)
        
        return x
    
    def enable_analysis_mode(self, store_states: bool = False):
        for block in self.blocks:
            block.enable_analysis_mode(store_states)
    
    def disable_analysis_mode(self):
        for block in self.blocks:
            block.disable_analysis_mode()
    
    def get_caches(self) -> List[SS2DCache]:
        """Get caches from all blocks in this stage."""
        return [block.get_cache() for block in self.blocks]


@dataclass
class VMambaAnalysisOutput:
    """
    Complete analysis output from Vision Mamba.
    
    Contains all intermediate values needed for controllability analysis.
    """
    # Model predictions
    logits: torch.Tensor = None
    probabilities: torch.Tensor = None
    predicted_class: torch.Tensor = None
    
    # Per-stage, per-block caches
    # Structure: stage_caches[stage_idx][block_idx] = SS2DCache
    stage_caches: List[List[SS2DCache]] = field(default_factory=list)
    
    # Spatial dimensions at each stage
    spatial_dims: List[Tuple[int, int]] = field(default_factory=list)
    
    # Feature maps at each stage (optional)
    features: List[torch.Tensor] = field(default_factory=list)


class VMambaClassifier(nn.Module):
    """
    Vision Mamba Classifier with Full Interpretability Support.
    
    This is the main model class that:
    1. Takes an input image
    2. Processes it through hierarchical VSSM stages
    3. Produces classification output
    4. Provides access to all intermediate values for analysis
    
    Args:
        config: VMambaConfig with model parameters
    """
    
    def __init__(self, config: VMambaConfig):
        super().__init__()
        
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.dims[0],
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        
        # Compute drop path rates (linearly increasing)
        total_blocks = sum(config.depths)
        dpr = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()
        block_idx = 0
        
        for stage_idx in range(config.num_stages):
            stage_depth = config.depths[stage_idx]
            stage_dim = config.dims[stage_idx]
            
            # Determine if this stage should downsample
            is_last = (stage_idx == config.num_stages - 1)
            output_dim = config.dims[stage_idx + 1] if not is_last else None
            
            stage = VSSMStage(
                depth=stage_depth,
                d_model=stage_dim,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                downsample=not is_last,
                output_dim=output_dim,
                drop_rate=config.drop_rate,
                drop_path_rates=dpr[block_idx:block_idx + stage_depth],
            )
            
            self.stages.append(stage)
            block_idx += stage_depth
        
        # Classification head
        self.norm = nn.LayerNorm(config.dims[-1])
        self.head = nn.Linear(config.dims[-1], config.num_classes)
        
        # Analysis mode
        self._analysis_mode = False
        self._store_features = False
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def enable_analysis_mode(self, store_states: bool = False, store_features: bool = False):
        """
        Enable analysis mode to store intermediate values.
        
        Args:
            store_states: Store hidden states at each position (memory intensive)
            store_features: Store feature maps at each stage
        """
        self._analysis_mode = True
        self._store_features = store_features
        
        for stage in self.stages:
            stage.enable_analysis_mode(store_states)
    
    def disable_analysis_mode(self):
        """Disable analysis mode to save memory during training."""
        self._analysis_mode = False
        self._store_features = False
        
        for stage in self.stages:
            stage.disable_analysis_mode()
    
    def forward(
        self,
        x: torch.Tensor,
        return_analysis: bool = False,
        use_sequential: bool = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, VMambaAnalysisOutput]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image [batch, channels, height, width]
            return_analysis: Whether to return analysis output
            use_sequential: Force sequential scan (auto-determined if None)
            
        Returns:
            logits: Classification logits [batch, num_classes]
            analysis: VMambaAnalysisOutput (if return_analysis=True)
        """
        # Determine scan mode
        if use_sequential is None:
            use_sequential = self._analysis_mode
        
        # Prepare analysis output if needed
        if return_analysis or self._analysis_mode:
            analysis = VMambaAnalysisOutput()
        else:
            analysis = None
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, H/P, W/P, D]
        
        # Track spatial dimensions
        if analysis is not None:
            analysis.spatial_dims.append((x.shape[1], x.shape[2]))
        
        # Process through stages
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, use_sequential=use_sequential)
            
            if analysis is not None:
                # Store spatial dimensions
                analysis.spatial_dims.append((x.shape[1], x.shape[2]))
                
                # Store caches
                analysis.stage_caches.append(stage.get_caches())
                
                # Store features if requested
                if self._store_features:
                    analysis.features.append(x.clone())
        
        # Global average pooling
        x = x.mean(dim=(1, 2))  # [B, D]
        
        # Classification
        x = self.norm(x)
        logits = self.head(x)
        
        if analysis is not None:
            analysis.logits = logits
            analysis.probabilities = F.softmax(logits, dim=-1)
            analysis.predicted_class = logits.argmax(dim=-1)
        
        if return_analysis:
            return logits, analysis
        else:
            return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience function to create common configurations
def create_vmamba_tiny(num_classes: int = 1000, image_size: int = 224) -> VMambaClassifier:
    """Create VMamba-Tiny model."""
    config = VMambaConfig(
        image_size=image_size,
        patch_size=16,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        d_state=16,
        num_classes=num_classes,
    )
    return VMambaClassifier(config)


def create_vmamba_small(num_classes: int = 1000, image_size: int = 224) -> VMambaClassifier:
    """Create VMamba-Small model."""
    config = VMambaConfig(
        image_size=image_size,
        patch_size=16,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 18, 2],
        d_state=16,
        num_classes=num_classes,
    )
    return VMambaClassifier(config)


def create_vmamba_base(num_classes: int = 1000, image_size: int = 224) -> VMambaClassifier:
    """Create VMamba-Base model."""
    config = VMambaConfig(
        image_size=image_size,
        patch_size=16,
        dims=[128, 256, 512, 1024],
        depths=[2, 2, 18, 2],
        d_state=16,
        num_classes=num_classes,
    )
    return VMambaClassifier(config)


# For medical imaging with smaller images
def create_vmamba_medical(
    num_classes: int,
    image_size: int = 224,
    in_channels: int = 3,
    patch_size: int = 4,  # Smaller patches for medical images
) -> VMambaClassifier:
    """
    Create VMamba model optimized for medical imaging.
    
    Uses smaller patches to preserve fine details.
    """
    config = VMambaConfig(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        dims=[64, 128, 256, 512],
        depths=[2, 2, 4, 2],
        d_state=16,
        num_classes=num_classes,
    )
    return VMambaClassifier(config)


# Test
if __name__ == "__main__":
    print("Testing VMamba Classifier...")
    
    # Create model
    model = create_vmamba_tiny(num_classes=10, image_size=224)
    print(f"Total parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test with analysis mode
    print("\nTesting analysis mode...")
    model.enable_analysis_mode(store_states=True, store_features=True)
    
    logits, analysis = model(x, return_analysis=True)
    
    print(f"\nAnalysis output:")
    print(f"  Logits shape: {analysis.logits.shape}")
    print(f"  Predicted classes: {analysis.predicted_class.tolist()}")
    print(f"  Spatial dims per stage: {analysis.spatial_dims}")
    print(f"  Number of stages with caches: {len(analysis.stage_caches)}")
    
    for stage_idx, stage_caches in enumerate(analysis.stage_caches):
        print(f"\n  Stage {stage_idx}:")
        print(f"    Number of blocks: {len(stage_caches)}")
        if stage_caches:
            cache = stage_caches[0]
            print(f"    Cached directions: {list(cache.direction_caches.keys())}")
            for direction, ssm_cache in cache.direction_caches.items():
                if ssm_cache.A_bar is not None:
                    print(f"      {direction.value}: A_bar shape = {ssm_cache.A_bar.shape}")
    
    # Test medical imaging configuration
    print("\n\nTesting medical imaging configuration...")
    medical_model = create_vmamba_medical(num_classes=2, image_size=224, in_channels=1)
    print(f"Medical model parameters: {medical_model.get_num_params():,}")
    
    x_medical = torch.randn(2, 1, 224, 224)
    logits_medical = medical_model(x_medical)
    print(f"Medical input shape: {x_medical.shape}")
    print(f"Medical output shape: {logits_medical.shape}")
