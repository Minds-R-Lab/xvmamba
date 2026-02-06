"""
Selective State Space Model (SSM) with Full Parameter Transparency

This module implements the core Mamba SSM layer with explicit access to all
internal parameters (A, B, C, Δ) needed for controllability analysis.

Key Design Decisions:
1. A matrix is diagonal (standard in Mamba) - enables Gramian method
2. All intermediate values are stored for interpretability
3. Uses official mamba-ssm CUDA kernels when available for speed
4. Falls back to Python implementation for analysis mode

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Try to import official mamba-ssm for CUDA acceleration
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_CUDA_AVAILABLE = True
    print("✓ mamba-ssm CUDA kernels available - using fast implementation")
except ImportError:
    MAMBA_CUDA_AVAILABLE = False
    print("⚠ mamba-ssm not available - using slow Python implementation")


@dataclass
class SSMCache:
    """
    Container for storing intermediate SSM values for interpretability.
    
    These values are needed for controllability analysis:
    - A_bar: Discretized state matrix (diagonal, per-position)
    - B_bar: Discretized input matrix (per-position)
    - C: Output projection matrix (per-position)
    - delta: Discretization step (per-position)
    - states: Hidden states at each position (optional, memory intensive)
    """
    A_bar: torch.Tensor = None      # [batch, length, state_dim]
    B_bar: torch.Tensor = None      # [batch, length, state_dim, input_dim] or simplified
    C: torch.Tensor = None          # [batch, length, state_dim]
    delta: torch.Tensor = None      # [batch, length, inner_dim]
    states: torch.Tensor = None     # [batch, length, state_dim] (optional)
    
    def clear(self):
        """Clear cached values to free memory."""
        self.A_bar = None
        self.B_bar = None
        self.C = None
        self.delta = None
        self.states = None


class FastSelectiveSSM(nn.Module):
    """
    Fast Selective SSM using official mamba-ssm CUDA kernels.
    
    This wraps the official Mamba implementation for speed while
    providing hooks for parameter extraction during analysis.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default 16)
        d_conv: Convolution width (default 4)
        expand: Expansion factor (default 2)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        if MAMBA_CUDA_AVAILABLE:
            # Use official Mamba implementation
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            raise RuntimeError("FastSelectiveSSM requires mamba-ssm package")
        
        # Cache for analysis
        self._cache = SSMCache()
        self._store_for_analysis = False
    
    def enable_analysis_mode(self, store_states: bool = False):
        """Enable parameter extraction for controllability analysis."""
        self._store_for_analysis = True
        self._store_states = store_states
    
    def disable_analysis_mode(self):
        """Disable analysis mode for faster training."""
        self._store_for_analysis = False
        self._cache.clear()
    
    def get_cache(self) -> SSMCache:
        """Get cached SSM parameters."""
        return self._cache
    
    def _extract_parameters(self, x: torch.Tensor):
        """
        Extract SSM parameters for analysis.
        
        This runs a separate forward pass to capture A_bar, B_bar, C, delta.
        Only called when analysis mode is enabled.
        """
        batch, length, _ = x.shape
        
        # Access internal Mamba parameters
        mamba = self.mamba
        
        # Input projection
        xz = mamba.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x_conv = x_inner.transpose(1, 2)
        x_conv = mamba.conv1d(x_conv)[:, :, :length]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM parameters projection
        x_dbl = mamba.x_proj(x_conv)
        
        # Split into delta, B, C
        dt = x_dbl[:, :, :mamba.dt_rank]
        B = x_dbl[:, :, mamba.dt_rank:mamba.dt_rank + self.d_state]
        C = x_dbl[:, :, mamba.dt_rank + self.d_state:]
        
        # Delta projection and softplus
        dt = mamba.dt_proj(dt)
        dt = F.softplus(dt)
        
        # Get A (log form stored in mamba)
        A = -torch.exp(mamba.A_log.float())  # [d_inner, d_state]
        
        # Discretize A_bar = exp(delta * A)
        # dt: [batch, length, d_inner]
        # A: [d_inner, d_state]
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(dt_A)  # [batch, length, d_inner, d_state]
        
        # Discretize B_bar = delta * B
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)  # [batch, length, d_inner, d_state]
        
        # Store in cache
        self._cache.A_bar = A_bar.detach()
        self._cache.B_bar = B_bar.detach()
        self._cache.C = C.detach()
        self._cache.delta = dt.detach()
    
    def forward(self, x: torch.Tensor, use_sequential: bool = False) -> torch.Tensor:
        """
        Forward pass using fast CUDA kernels.
        
        Args:
            x: Input [batch, length, d_model]
            use_sequential: Ignored (always uses fast kernels), but triggers analysis
            
        Returns:
            y: Output [batch, length, d_model]
        """
        # Extract parameters for analysis if needed
        if self._store_for_analysis or use_sequential:
            self._extract_parameters(x)
        
        # Use fast CUDA forward
        return self.mamba(x)


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model layer with interpretability support.
    
    This implements the core Mamba selective scan mechanism where the SSM
    parameters (B, C, Δ) are input-dependent, allowing content-aware reasoning.
    
    The state equation is:
        h_t = Ā_t * h_{t-1} + B̄_t * x_t
        y_t = C_t * h_t
    
    Where:
        Ā_t = exp(Δ_t * A)  -- discretized state matrix
        B̄_t = Δ_t * B_t     -- discretized input matrix (simplified ZOH)
    
    Args:
        d_model: Model dimension (input/output dimension)
        d_state: SSM state dimension (N in the paper, typically 16)
        d_conv: Local convolution width (typically 4)
        expand: Expansion factor for inner dimension (typically 2)
        dt_min: Minimum discretization step
        dt_max: Maximum discretization step
        dt_init: Initialization method for dt ('random' or 'constant')
        dt_scale: Scale factor for dt initialization
        store_states: Whether to store all hidden states (memory intensive)
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        store_states: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.store_states = store_states
        
        # Input projection: projects input to inner dimension
        # Creates: x (for SSM), z (for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Local convolution for capturing local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise convolution
        )
        
        # SSM parameter projections (input-dependent)
        # x_proj: projects x to (Δ, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, 
            self.d_state + self.d_state + 1,  # dt_rank=1 for simplicity, B, C
            bias=False
        )
        
        # Δ (delta) projection and parameters
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias for proper range
        dt_init_std = dt_scale / math.sqrt(self.d_inner)
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize bias to achieve dt in [dt_min, dt_max] after softplus
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter: diagonal, learned, initialized for stability
        # A is stored as log(-A) for numerical stability during exp
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(self.d_inner, -1))
        
        # D parameter: skip connection (direct input-to-output)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Cache for interpretability
        self._cache = SSMCache()
        self._store_for_analysis = False
    
    def enable_analysis_mode(self, store_states: bool = False):
        """Enable storage of intermediate values for controllability analysis."""
        self._store_for_analysis = True
        self.store_states = store_states
    
    def disable_analysis_mode(self):
        """Disable storage to save memory during normal training."""
        self._store_for_analysis = False
        self.store_states = False
        self._cache.clear()
    
    def get_cache(self) -> SSMCache:
        """Retrieve cached values for analysis."""
        return self._cache
    
    def _compute_ssm_parameters(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent SSM parameters.
        
        Args:
            x: Input tensor [batch, length, d_inner]
            
        Returns:
            A_bar: Discretized diagonal state matrix [batch, length, d_inner, d_state]
            B_bar: Discretized input matrix [batch, length, d_state]
            C: Output matrix [batch, length, d_state]
            delta: Discretization step [batch, length, d_inner]
        """
        batch, length, _ = x.shape
        
        # Project x to get dt_input, B, C
        x_dbl = self.x_proj(x)  # [batch, length, dt_rank + 2*d_state]
        
        # Split into components
        dt_input = x_dbl[..., :1]  # [batch, length, 1]
        B = x_dbl[..., 1:1+self.d_state]  # [batch, length, d_state]
        C = x_dbl[..., 1+self.d_state:]   # [batch, length, d_state]
        
        # Compute Δ (delta) - the discretization step
        delta = self.dt_proj(dt_input)  # [batch, length, d_inner]
        delta = F.softplus(delta)  # Ensure positive
        
        # Get A (diagonal, shared across batch and positions)
        A = -torch.exp(self.A_log)  # [d_inner, d_state], negative for stability
        
        # Discretize: A_bar = exp(Δ * A)
        # delta: [batch, length, d_inner]
        # A: [d_inner, d_state]
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [batch, length, d_inner, d_state]
        A_bar = torch.exp(delta_A)  # [batch, length, d_inner, d_state]
        
        # Discretize: B_bar = Δ * B (simplified ZOH approximation)
        # This is a simplification; full ZOH would be (A_bar - I) * A^{-1} * B
        # But for diagonal A, this simplification works well in practice
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)  # [batch, length, d_inner, d_state]
        
        return A_bar, B_bar, C, delta
    
    def _selective_scan_sequential(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential selective scan - slower but clearer for analysis.
        
        Implements:
            h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            y_t = C_t @ h_t
        
        Args:
            x: Input [batch, length, d_inner]
            A_bar: Discretized state matrix [batch, length, d_inner, d_state]
            B_bar: Discretized input matrix [batch, length, d_inner, d_state]
            C: Output matrix [batch, length, d_state]
            
        Returns:
            y: Output [batch, length, d_inner]
        """
        batch, length, d_inner = x.shape
        d_state = A_bar.shape[-1]
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        states = [] if self.store_states else None
        
        for t in range(length):
            # State update: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            # A_bar_t: [batch, d_inner, d_state]
            # h: [batch, d_inner, d_state]
            # B_bar_t: [batch, d_inner, d_state]
            # x_t: [batch, d_inner]
            
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            
            # Output: y_t = sum over state_dim (C_t * h_t)
            # C_t: [batch, d_state]
            # h: [batch, d_inner, d_state]
            y_t = torch.einsum('bd,bid->bi', C[:, t], h)  # [batch, d_inner]
            
            outputs.append(y_t)
            
            if self.store_states:
                states.append(h.clone())
        
        y = torch.stack(outputs, dim=1)  # [batch, length, d_inner]
        
        if self.store_states and states:
            self._cache.states = torch.stack(states, dim=1)  # [batch, length, d_inner, d_state]
        
        return y
    
    def _selective_scan_parallel(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel selective scan using associative scan (for training efficiency).
        
        This uses the parallel scan algorithm which has O(log L) depth.
        For simplicity, we use a chunked approach here.
        
        In production, you'd use the CUDA kernels from mamba-ssm package.
        """
        # For now, fall back to sequential (can be optimized later)
        return self._selective_scan_sequential(x, A_bar, B_bar, C)
    
    def forward(
        self, 
        x: torch.Tensor,
        use_sequential: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the selective SSM.
        
        Args:
            x: Input tensor [batch, length, d_model]
            use_sequential: Force sequential scan (slower but stores states)
            
        Returns:
            y: Output tensor [batch, length, d_model]
        """
        batch, length, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # [batch, length, 2 * d_inner]
        x_inner, z = xz.chunk(2, dim=-1)  # Each: [batch, length, d_inner]
        
        # Local convolution
        x_conv = x_inner.transpose(1, 2)  # [batch, d_inner, length]
        x_conv = self.conv1d(x_conv)[:, :, :length]  # [batch, d_inner, length]
        x_conv = x_conv.transpose(1, 2)  # [batch, length, d_inner]
        x_conv = F.silu(x_conv)
        
        # Compute SSM parameters
        A_bar, B_bar, C, delta = self._compute_ssm_parameters(x_conv)
        
        # Store for analysis if enabled
        if self._store_for_analysis:
            self._cache.A_bar = A_bar.detach()
            self._cache.B_bar = B_bar.detach()
            self._cache.C = C.detach()
            self._cache.delta = delta.detach()
        
        # Run selective scan
        if use_sequential or self._store_for_analysis:
            y = self._selective_scan_sequential(x_conv, A_bar, B_bar, C)
        else:
            y = self._selective_scan_parallel(x_conv, A_bar, B_bar, C)
        
        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        return y


class SelectiveSSMBlock(nn.Module):
    """
    Complete SSM block with normalization and residual connection.
    
    Structure:
        x -> LayerNorm -> SelectiveSSM -> + -> output
        |_________________________________|
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **ssm_kwargs
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **ssm_kwargs
        )
    
    def forward(self, x: torch.Tensor, use_sequential: bool = False) -> torch.Tensor:
        """Forward with pre-norm and residual."""
        return x + self.ssm(self.norm(x), use_sequential=use_sequential)
    
    def enable_analysis_mode(self, store_states: bool = False):
        self.ssm.enable_analysis_mode(store_states)
    
    def disable_analysis_mode(self):
        self.ssm.disable_analysis_mode()
    
    def get_cache(self) -> SSMCache:
        return self.ssm.get_cache()


# Test the implementation
if __name__ == "__main__":
    # Quick test
    batch_size = 2
    seq_len = 16
    d_model = 64
    
    model = SelectiveSSM(d_model=d_model, d_state=16)
    model.enable_analysis_mode(store_states=True)
    
    x = torch.randn(batch_size, seq_len, d_model)
    y = model(x, use_sequential=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    cache = model.get_cache()
    print(f"A_bar shape: {cache.A_bar.shape}")
    print(f"B_bar shape: {cache.B_bar.shape}")
    print(f"C shape: {cache.C.shape}")
    print(f"delta shape: {cache.delta.shape}")
    print(f"states shape: {cache.states.shape}")
    
    # Verify A_bar values are in (0, 1) for stability
    print(f"A_bar range: [{cache.A_bar.min():.4f}, {cache.A_bar.max():.4f}]")
