"""
Comprehensive Controllability Evaluation for Vision Mamba

This script implements four evaluation paradigms that demonstrate the unique
value of controllability analysis beyond simple saliency comparison:

1. PERTURBATION INVARIANCE TEST
   - Perturb high-controllability pixels → prediction SHOULD change
   - Perturb low-controllability pixels → prediction should NOT change
   - Validates that controllability correctly identifies influential regions

2. CROSS-CLASS CONSISTENCY TEST
   - Controllability is structural (depends on A, B, C matrices)
   - Maps should be nearly identical regardless of target class
   - Grad-CAM maps will differ significantly per class
   - Demonstrates the unique "task-agnostic" property

3. ARCHITECTURE ANALYSIS
   - Per-stage controllability statistics
   - Information flow analysis across layers
   - Reveals architectural insights about VMamba's design

4. STATE MAGNITUDE CORRELATION
   - High controllability should correlate with large state magnitudes
   - Validates the theoretical connection to control theory

Plus: Standard faithfulness evaluation (deletion/insertion AUC)

Usage:
    python comprehensive_evaluation.py \
        --checkpoint ./checkpoints/pneumoniamnist/best_model.pth \
        --dataset pneumoniamnist \
        --num_samples 50

Author: X-VMamba Validation Pipeline
"""

import argparse
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VMambaClassifier, VMambaConfig
from models.vim_classifier import VimClassifier, VimConfig
from controllability import ControllabilityAnalyzer
from data import DatasetType, get_dataloader, get_dataset_info

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# Publication Style Configuration
# ============================================================================

METHOD_COLORS = {
    'Jacobian': '#d62728',
    'Gramian': '#1f77b4',
    'Grad-CAM': '#2ca02c',
    'InputGrad': '#9467bd',
    'Random': '#7f7f7f',
}


def apply_pub_style():
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
    })



def _to_picklable(obj):
    """Recursively convert defaultdict (with lambda default_factory) and
    any nested defaultdicts into plain dicts so that torch.save / pickle
    can serialise them. Leaves other types untouched."""
    from collections import defaultdict as _dd
    if isinstance(obj, _dd):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: _to_picklable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_picklable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_picklable(v) for v in obj)
    return obj


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path, device):
    """Load a checkpoint, dispatching on the saved `model_arch` flag.

    Backwards compatible: checkpoints saved before the Vim wiring don't
    have `model_arch` in their args dict; we default to "vmamba".
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})
    arch = str(args.get('model_arch', 'vmamba')).lower()

    if arch == 'vmamba':
        config = VMambaConfig(
            image_size=args.get('image_size', 224),
            patch_size=args.get('patch_size', 4),
            in_channels=args.get('in_channels', 3),
            dims=args.get('dims', [32, 64, 128, 256]),
            depths=args.get('depths', [2, 2, 4, 2]),
            d_state=args.get('d_state', 16),
            num_classes=args.get('num_classes', 7),
        )
        model = VMambaClassifier(config)
    elif arch == 'vim':
        config = VimConfig(
            image_size=args.get('image_size', 224),
            patch_size=args.get('patch_size', 16),
            in_channels=args.get('in_channels', 3),
            d_model=args.get('vim_d_model', 192),
            depth=args.get('vim_depth', 12),
            d_state=args.get('d_state', 16),
            num_classes=args.get('num_classes', 10),
            drop_rate=args.get('drop_rate', 0.0),
            mlp_ratio=args.get('vim_mlp_ratio', 4.0),
        )
        model = VimClassifier(config)
    else:
        raise ValueError(f"unknown model_arch in checkpoint: {arch!r}")

    # Adapter for checkpoints saved with mamba-ssm CUDA path.
    # When the original training used FastSelectiveSSM (mamba-ssm CUDA backend),
    # the SSM parameters were stored under a `.mamba.` prefix:
    #     stages.X.blocks.Y.ss2d.ssm.mamba.A_log
    # When we evaluate on CPU (no mamba-ssm), the slow Python path flattens those:
    #     stages.X.blocks.Y.ss2d.ssm.A_log
    # We strip the `.mamba.` segment from each key when the model expects flat keys.
    state = checkpoint['model_state_dict']
    model_keys = set(model.state_dict().keys())
    if any(k.endswith('.mamba.A_log') for k in state.keys()) and \
       not any(k.endswith('.mamba.A_log') for k in model_keys):
        # Strip `.ssm.mamba.` (VMamba) or `.ssm_forward.mamba.` /
        # `.ssm_backward.mamba.` (Vim) so flat state dicts load on
        # either the CUDA or the slow-Python path.
        new_state = {}
        for k, v in state.items():
            k = k.replace('.ssm.mamba.', '.ssm.')
            k = k.replace('.ssm_forward.mamba.', '.ssm_forward.')
            k = k.replace('.ssm_backward.mamba.', '.ssm_backward.')
            new_state[k] = v
        state = new_state
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model, config


# ============================================================================
# Saliency Methods (Structural Controllability - NO class conditioning)
# ============================================================================

class StructuralControllability:
    """
    Pure structural controllability - NO gradient weighting.
    
    This measures the intrinsic influence of each position on SSM dynamics,
    independent of any specific class or prediction.
    """
    
    def __init__(self, model, device, method='jacobian', layer_weighting='exponential'):
        self.model = model
        self.device = device
        self.method = method
        self.layer_weighting = layer_weighting
        self.analyzer = ControllabilityAnalyzer(method=method, normalize=True)
    
    def generate(self, image, target_class=None):
        """Generate structural controllability map (target_class is IGNORED)."""
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]
        
        self.model.enable_analysis_mode(store_states=False)
        with torch.no_grad():
            _, analysis = self.model(image, return_analysis=True)
        
        ctrl_map = self._weighted_aggregate(analysis, target_size=(h, w))
        self.model.disable_analysis_mode()
        
        return self._normalize(ctrl_map).cpu()
    
    def _weighted_aggregate(self, analysis_output, target_size):
        layer_maps = []
        
        for stage_idx, stage_caches in enumerate(analysis_output.stage_caches):
            for block_idx, cache in enumerate(stage_caches):
                result = self.analyzer.analyze_block(
                    cache, stage_idx=stage_idx, block_idx=block_idx,
                )
                imap = result.influence_map
                if imap is None:
                    continue
                
                if imap.shape[0] != target_size[0] or imap.shape[1] != target_size[1]:
                    imap = F.interpolate(
                        imap.unsqueeze(0).unsqueeze(0),
                        size=target_size, mode='bilinear', align_corners=False,
                    ).squeeze()
                
                layer_maps.append(imap)
        
        if not layer_maps:
            return torch.ones(target_size, device=self.device) * 0.5
        
        stacked = torch.stack(layer_maps, dim=0)
        weights = self._layer_weights(len(layer_maps), stacked.device)
        aggregated = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
        return aggregated
    
    def _layer_weights(self, n, device):
        if self.layer_weighting == 'exponential':
            raw = torch.exp(torch.linspace(0, 2, n, device=device))
        elif self.layer_weighting == 'linear':
            raw = torch.arange(1, n + 1, dtype=torch.float32, device=device)
        else:
            raw = torch.ones(n, device=device)
        return raw / raw.sum()
    
    @staticmethod
    def _normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin)


class GradCAMSaliency:
    """Grad-CAM baseline - CLASS-SPECIFIC attribution."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()
        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Auto-locate the last feature-map block so the same hook works on
        # both VMamba (hierarchical `stages[-1].blocks[-1]`) and Vim
        # (plain `blocks[-1]`).
        if hasattr(self.model, "stages") and len(self.model.stages) > 0:
            last_block = self.model.stages[-1].blocks[-1]
        elif hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            last_block = self.model.blocks[-1]
        else:
            raise AttributeError(
                "Could not locate a last feature-map block on this model "
                "(no `stages` or `blocks` attribute)."
            )
        last_block.register_forward_hook(fwd_hook)
        last_block.register_full_backward_hook(bwd_hook)
    
    def generate(self, image, target_class):
        """Generate Grad-CAM for SPECIFIC target class."""
        self.model.zero_grad()
        image = image.to(self.device).requires_grad_(True)
        
        output = self.model(image)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is not None and self.activations is not None:
            # The activations and gradients shapes depend on the architecture:
            #   VMamba (channels-last 2D): [B, H, W, C]
            #   Vim    (flat sequence):    [B, L, C], with L = H_grid * W_grid
            # We dispatch on rank so Grad-CAM works on both.
            if self.activations.ndim == 4:
                # [B, H, W, C]: average gradients over spatial dims.
                weights = self.gradients.mean(dim=(1, 2), keepdim=True)   # [B, 1, 1, C]
                cam = (weights * self.activations).sum(dim=-1)             # [B, H, W]
                cam = F.relu(cam).squeeze(0)                                # [H, W]
            elif self.activations.ndim == 3:
                # [B, L, C]: average over token positions.
                weights = self.gradients.mean(dim=1, keepdim=True)          # [B, 1, C]
                cam_seq = (weights * self.activations).sum(dim=-1)          # [B, L]
                cam_seq = F.relu(cam_seq).squeeze(0)                        # [L]
                # Reshape sequence back to a (h_grid, w_grid) map.
                L = cam_seq.shape[0]
                side = int(round(L ** 0.5))
                if side * side != L:
                    raise ValueError(
                        f"Grad-CAM: cannot reshape sequence length {L} to a "
                        f"square grid; got side={side}."
                    )
                cam = cam_seq.view(side, side)
            else:
                raise ValueError(
                    f"Unexpected activations rank for Grad-CAM: "
                    f"{tuple(self.activations.shape)}"
                )

            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            h, w = image.shape[2], image.shape[3]
            if cam.shape[0] != h or cam.shape[1] != w:
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=(h, w), mode='bilinear', align_corners=False
                ).squeeze()

            return cam.cpu()
        
        return torch.rand(image.shape[2], image.shape[3])


class RandomSaliency:
    """Random baseline."""
    def generate(self, image, target_class=None):
        return torch.rand(image.shape[2], image.shape[3])


# ============================================================================
# TEST 1: PERTURBATION INVARIANCE
# ============================================================================

def perturbation_invariance_test(model, dataloader, device, num_samples=50,
                                  perturb_percent=0.1, perturbation_strength=0.5,
                                  min_orig_conf=0.0):
    """
    Test whether controllability correctly identifies influential regions.
    
    Protocol:
    1. For each image, compute controllability maps
    2. Perturb top-K% pixels (high controllability) → measure confidence drop
    3. Perturb bottom-K% pixels (low controllability) → measure confidence drop
    4. Ratio (high_drop / low_drop) should be >> 1 if controllability is correct
    
    We also run the same test for Grad-CAM and Random as baselines.
    """
    print("\n" + "=" * 70)
    print("TEST 1: PERTURBATION INVARIANCE")
    print("=" * 70)
    print(f"  Perturbing top/bottom {perturb_percent*100:.0f}% of pixels")
    print(f"  Perturbation strength: {perturbation_strength}")
    
    ctrl_j = StructuralControllability(model, device, 'jacobian')
    ctrl_g = StructuralControllability(model, device, 'gramian')
    gradcam = GradCAMSaliency(model, device)
    random_sal = RandomSaliency()
    
    methods = {
        'Jacobian': ctrl_j,
        'Gramian': ctrl_g,
        'Grad-CAM': gradcam,
        'Random': random_sal,
    }
    
    results = {name: {'high_drop': [], 'low_drop': [], 'ratio': [], 'drop_diff': []} 
               for name in methods}
    
    sample_count = 0
    for images, labels in tqdm(dataloader, desc="Perturbation Test", 
                                total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break
        
        image = images[0:1].to(device)
        
        # Get original prediction
        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(dim=1).item()
            orig_conf = F.softmax(output, dim=1)[0, pred_class].item()
        
        # Skip if model is very uncertain
        # G3: confidence filter (default 0.0 = no filter); documented in caption.
        if orig_conf < min_orig_conf:
            continue
        
        h, w = image.shape[2], image.shape[3]
        n_pixels = h * w
        k = int(n_pixels * perturb_percent)
        
        for name, method in methods.items():
            try:
                saliency = method.generate(image, pred_class)
                flat = saliency.flatten()
                
                # Get top-k (high saliency) and bottom-k (low saliency) indices
                high_indices = torch.topk(flat, k).indices
                low_indices = torch.topk(flat, k, largest=False).indices
                
                # Create perturbation masks
                high_mask = torch.zeros(h * w, device=device)
                high_mask[high_indices] = 1
                high_mask = high_mask.reshape(1, 1, h, w)
                
                low_mask = torch.zeros(h * w, device=device)
                low_mask[low_indices] = 1
                low_mask = low_mask.reshape(1, 1, h, w)
                
                # Perturb by adding Gaussian noise
                noise = torch.randn_like(image) * perturbation_strength
                
                # Perturb high-saliency regions
                perturbed_high = image + noise * high_mask
                perturbed_high = perturbed_high.clamp(-3, 3)  # Keep in reasonable range
                
                # Perturb low-saliency regions
                perturbed_low = image + noise * low_mask
                perturbed_low = perturbed_low.clamp(-3, 3)
                
                # Measure confidence after perturbation
                with torch.no_grad():
                    out_high = model(perturbed_high)
                    conf_high = F.softmax(out_high, dim=1)[0, pred_class].item()
                    
                    out_low = model(perturbed_low)
                    conf_low = F.softmax(out_low, dim=1)[0, pred_class].item()
                
                high_drop = orig_conf - conf_high
                low_drop = orig_conf - conf_low
                
                results[name]['high_drop'].append(high_drop)
                results[name]['low_drop'].append(low_drop)
                # B1: drop_diff is bounded and interpretable; ratio retained
                # for backwards compatibility but is unstable near low_drop=0.
                results[name]['drop_diff'].append(high_drop - low_drop)

                ratio = high_drop / (low_drop + 1e-6) if low_drop > 0 else high_drop / 1e-6
                results[name]['ratio'].append(ratio)
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        sample_count += 1
    
    # Aggregate and print results
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'High Drop':<12} {'Low Drop':<12} {'Ratio':<12} {'Interpretation'}")
    print("-" * 70)
    
    summary = {}
    for name in methods:
        if results[name]['high_drop']:
            high_mean = np.mean(results[name]['high_drop'])
            high_std = np.std(results[name]['high_drop'])
            low_mean = np.mean(results[name]['low_drop'])
            low_std = np.std(results[name]['low_drop'])
            ratio_mean = np.mean(results[name]['ratio'])
            
            diff_mean = float(np.mean(results[name]['drop_diff']))
            diff_std  = float(np.std(results[name]['drop_diff']))
            summary[name] = {
                'high_drop_mean': high_mean,
                'high_drop_std': high_std,
                'low_drop_mean': low_mean,
                'low_drop_std': low_std,
                'drop_diff_mean': diff_mean,
                'drop_diff_std':  diff_std,
                'ratio_mean': ratio_mean,  # deprecated -- unstable near low_drop=0
            }
            
            # Interpretation
            if ratio_mean > 2.0:
                interp = "✓ Excellent"
            elif ratio_mean > 1.5:
                interp = "✓ Good"
            elif ratio_mean > 1.0:
                interp = "~ Weak"
            else:
                interp = "✗ Failed"
            
            print(f"{name:<15} {high_mean:.4f}±{high_std:.3f}  {low_mean:.4f}±{low_std:.3f}  "
                  f"{ratio_mean:.2f}         {interp}")
    
    print("-" * 70)
    print("Interpretation: Ratio > 1 means high-saliency regions are more influential.")
    print("                Ratio >> 1 validates that the method identifies truly important regions.")
    
    return summary, results


# ============================================================================
# TEST 2: CROSS-CLASS CONSISTENCY
# ============================================================================

def cross_class_consistency_test(model, dataloader, device, num_classes, 
                                  num_samples=30, num_test_classes=5):
    """
    Test whether controllability is class-agnostic (structural).
    
    Protocol:
    1. For each image, compute controllability maps "targeting" different classes
    2. Since controllability is STRUCTURAL, maps should be nearly identical
    3. Compute pairwise correlation between maps for different classes
    4. Compare to Grad-CAM which WILL produce different maps per class
    
    Expected result:
    - Controllability: correlation ≈ 1.0 (maps are identical)
    - Grad-CAM: correlation << 1.0 (maps depend on target class)
    """
    print("\n" + "=" * 70)
    print("TEST 2: CROSS-CLASS CONSISTENCY")
    print("=" * 70)
    print(f"  Testing with {num_test_classes} different target classes per image")
    
    ctrl_j = StructuralControllability(model, device, 'jacobian')
    ctrl_g = StructuralControllability(model, device, 'gramian')
    gradcam = GradCAMSaliency(model, device)
    
    # B2: target classes are chosen per-image (top-K predictions) below;
    #     the previous range [0..K-1] queried implausible targets.
    
    results = {
        'Jacobian': [],
        'Gramian': [],
        'Grad-CAM': [],
    }
    
    sample_count = 0
    for images, labels in tqdm(dataloader, desc="Cross-Class Test",
                                total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break
        
        image = images[0:1].to(device)

        # B2: top-K predicted classes for this image.
        with torch.no_grad():
            _logits = model(image)
        _topk = _logits[0].topk(min(num_test_classes, num_classes)).indices.tolist()
        test_classes_per_image = _topk[: num_test_classes]

        # Compute maps for different "target" classes
        jacobian_maps = []
        gramian_maps = []
        gradcam_maps = []

        for target_class in test_classes_per_image:
            try:
                # Controllability is structural - should NOT depend on target_class
                j_map = ctrl_j.generate(image, target_class)
                g_map = ctrl_g.generate(image, target_class)
                
                # Grad-CAM IS class-specific - SHOULD depend on target_class
                gc_map = gradcam.generate(image, target_class)
                
                jacobian_maps.append(j_map.flatten())
                gramian_maps.append(g_map.flatten())
                gradcam_maps.append(gc_map.flatten())
                
            except Exception as e:
                continue
        
        if len(jacobian_maps) < 2:
            continue
        
        # Compute pairwise correlations
        def compute_mean_correlation(maps):
            correlations = []
            for i in range(len(maps)):
                for j in range(i + 1, len(maps)):
                    corr = torch.corrcoef(torch.stack([maps[i], maps[j]]))[0, 1].item()
                    if not np.isnan(corr):
                        correlations.append(corr)
            return np.mean(correlations) if correlations else 0.0
        
        results['Jacobian'].append(compute_mean_correlation(jacobian_maps))
        results['Gramian'].append(compute_mean_correlation(gramian_maps))
        results['Grad-CAM'].append(compute_mean_correlation(gradcam_maps))
        
        sample_count += 1
    
    # Print results
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Mean Correlation':<20} {'Std':<12} {'Interpretation'}")
    print("-" * 70)
    
    summary = {}
    for name, correlations in results.items():
        if correlations:
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            
            summary[name] = {
                'mean_correlation': mean_corr,
                'std_correlation': std_corr,
            }
            
            if mean_corr > 0.95:
                interp = "✓ Class-agnostic (structural)"
            elif mean_corr > 0.8:
                interp = "~ Mostly structural"
            else:
                interp = "✗ Class-dependent"
            
            print(f"{name:<15} {mean_corr:.4f}               {std_corr:.4f}        {interp}")
    
    print("-" * 70)
    print("Interpretation: Controllability should have correlation ≈ 1.0 (class-agnostic).")
    print("                Grad-CAM should have lower correlation (class-dependent).")
    
    return summary, results


# ============================================================================
# TEST 3: ARCHITECTURE ANALYSIS
# ============================================================================

def architecture_analysis(model, dataloader, device, num_samples=20):
    """
    Analyze controllability patterns across the model architecture.
    
    Questions answered:
    1. Which stage has highest mean controllability?
    2. Does controllability increase or decrease with depth?
    3. How focused (sparse) are the maps at different stages?
    4. What is the entropy distribution per stage?
    
    This provides architectural insights about VMamba's information flow.
    """
    print("\n" + "=" * 70)
    print("TEST 3: ARCHITECTURE ANALYSIS")
    print("=" * 70)
    
    analyzer_j = ControllabilityAnalyzer(method='jacobian', normalize=True)
    analyzer_g = ControllabilityAnalyzer(method='gramian', normalize=True)
    
    # Per-stage, per-block statistics
    stage_stats = defaultdict(lambda: defaultdict(lambda: {
        'mean_j': [], 'max_j': [], 'entropy_j': [], 'sparsity_j': [],
        'mean_g': [], 'max_g': [], 'entropy_g': [], 'sparsity_g': [],
    }))
    
    sample_count = 0
    for images, labels in tqdm(dataloader, desc="Architecture Analysis",
                                total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break
        
        image = images[0:1].to(device)
        
        model.enable_analysis_mode(store_states=False)
        with torch.no_grad():
            _, analysis = model(image, return_analysis=True)
        
        for stage_idx, stage_caches in enumerate(analysis.stage_caches):
            for block_idx, cache in enumerate(stage_caches):
                # Jacobian analysis
                try:
                    result_j = analyzer_j.analyze_block(cache, stage_idx, block_idx)
                    if result_j.influence_map is not None:
                        imap = result_j.influence_map
                        stats = stage_stats[stage_idx][block_idx]
                        
                        stats['mean_j'].append(imap.mean().item())
                        stats['max_j'].append(imap.max().item())
                        stats['sparsity_j'].append((imap > 0.5).float().mean().item())
                        
                        flat = imap.flatten()
                        flat = flat / (flat.sum() + 1e-8)
                        entropy = -torch.sum(flat * torch.log(flat + 1e-8)).item()
                        stats['entropy_j'].append(entropy)
                except:
                    pass
                
                # Gramian analysis
                try:
                    result_g = analyzer_g.analyze_block(cache, stage_idx, block_idx)
                    if result_g.influence_map is not None:
                        imap = result_g.influence_map
                        stats = stage_stats[stage_idx][block_idx]
                        
                        stats['mean_g'].append(imap.mean().item())
                        stats['max_g'].append(imap.max().item())
                        stats['sparsity_g'].append((imap > 0.5).float().mean().item())
                        
                        flat = imap.flatten()
                        flat = flat / (flat.sum() + 1e-8)
                        entropy = -torch.sum(flat * torch.log(flat + 1e-8)).item()
                        stats['entropy_g'].append(entropy)
                except:
                    pass
        
        model.disable_analysis_mode()
        sample_count += 1
    
    # Aggregate per-stage statistics
    print("\n" + "-" * 70)
    print("PER-STAGE CONTROLLABILITY PROFILE")
    print("-" * 70)
    print(f"{'Stage':<8} {'Block':<8} {'Mean(J)':<10} {'Mean(G)':<10} {'Entropy(J)':<12} {'Sparsity(J)'}")
    print("-" * 70)
    
    summary = {}
    for stage_idx in sorted(stage_stats.keys()):
        stage_summary = {'blocks': {}}
        for block_idx in sorted(stage_stats[stage_idx].keys()):
            stats = stage_stats[stage_idx][block_idx]
            
            mean_j = np.mean(stats['mean_j']) if stats['mean_j'] else 0
            mean_g = np.mean(stats['mean_g']) if stats['mean_g'] else 0
            entropy_j = np.mean(stats['entropy_j']) if stats['entropy_j'] else 0
            sparsity_j = np.mean(stats['sparsity_j']) if stats['sparsity_j'] else 0
            
            stage_summary['blocks'][block_idx] = {
                'mean_jacobian': mean_j,
                'mean_gramian': mean_g,
                'entropy_jacobian': entropy_j,
                'sparsity_jacobian': sparsity_j,
            }
            
            print(f"  {stage_idx:<6} {block_idx:<8} {mean_j:<10.4f} {mean_g:<10.4f} "
                  f"{entropy_j:<12.2f} {sparsity_j:.4f}")
        
        summary[stage_idx] = stage_summary
    
    # Stage-level summary
    print("\n" + "-" * 70)
    print("STAGE-LEVEL SUMMARY")
    print("-" * 70)
    
    for stage_idx in sorted(stage_stats.keys()):
        all_mean_j = []
        all_entropy_j = []
        for block_idx in stage_stats[stage_idx]:
            stats = stage_stats[stage_idx][block_idx]
            all_mean_j.extend(stats['mean_j'])
            all_entropy_j.extend(stats['entropy_j'])
        
        if all_mean_j:
            resolution = 56 // (2 ** stage_idx)  # Approximate spatial resolution
            print(f"Stage {stage_idx} ({resolution}x{resolution}): "
                  f"Mean Ctrl={np.mean(all_mean_j):.4f}, "
                  f"Mean Entropy={np.mean(all_entropy_j):.2f}")
    
    print("-" * 70)
    print("Interpretation: Higher mean controllability = more information flow.")
    print("                Lower entropy = more focused/sparse attention.")
    
    return summary, stage_stats


# ============================================================================
# TEST 4: STATE MAGNITUDE CORRELATION
# ============================================================================

def state_magnitude_correlation_test(model, dataloader, device, num_samples=20):
    """
    Test if high controllability correlates with large state magnitudes.
    
    Theory: Controllability measures how much input position k CAN influence
    the SSM state. If this is correct, positions with high controllability
    should correspond to positions where the hidden state ||h_k|| is large
    (when there's actual signal at that position).
    
    Protocol:
    1. Forward pass with store_states=True
    2. For each block, compute controllability per position
    3. Compute ||h_k|| (state magnitude) per position
    4. Measure correlation between controllability and state magnitude
    """
    print("\n" + "=" * 70)
    print("TEST 4: STATE MAGNITUDE CORRELATION")
    print("=" * 70)
    print("  Testing correlation between controllability and hidden state magnitude")
    
    analyzer_j = ControllabilityAnalyzer(method='jacobian', normalize=True)
    
    correlations_per_stage = defaultdict(list)
    
    sample_count = 0
    for images, labels in tqdm(dataloader, desc="State Correlation Test",
                                total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break
        
        image = images[0:1].to(device)
        
        # Forward pass with state storage
        model.enable_analysis_mode(store_states=True)
        with torch.no_grad():
            _, analysis = model(image, return_analysis=True)
        
        for stage_idx, stage_caches in enumerate(analysis.stage_caches):
            for block_idx, cache in enumerate(stage_caches):
                try:
                    # Get controllability map
                    result = analyzer_j.analyze_block(cache, stage_idx, block_idx)
                    if result.influence_map is None:
                        continue
                    
                    ctrl_map = result.influence_map.flatten()
                    
                    # Try to get state magnitudes from cache
                    # The states are stored in the direction_caches
                    state_mags = []
                    for direction, ssm_cache in cache.direction_caches.items():
                        if hasattr(ssm_cache, 'states') and ssm_cache.states is not None:
                            # states: [batch, length, d_inner, d_state]
                            # Compute norm over state dimensions
                            s = ssm_cache.states
                            mag = s.norm(dim=-1).mean(dim=-1).mean(dim=0)  # [length]
                            state_mags.append(mag)
                    
                    if state_mags:
                        # Average state magnitude across directions
                        avg_state_mag = torch.stack(state_mags, dim=0).mean(dim=0)
                        
                        # Resize to match controllability map if needed
                        if len(avg_state_mag) != len(ctrl_map):
                            # The spatial dimensions might not match exactly
                            # Skip this sample for now
                            continue
                        
                        # Compute correlation
                        corr = torch.corrcoef(
                            torch.stack([ctrl_map, avg_state_mag.to(ctrl_map.device)])
                        )[0, 1].item()
                        
                        if not np.isnan(corr):
                            correlations_per_stage[stage_idx].append(corr)
                
                except Exception as e:
                    continue
        
        model.disable_analysis_mode()
        sample_count += 1
    
    # Print results
    print("\n" + "-" * 70)
    print(f"{'Stage':<10} {'Mean Correlation':<20} {'Std':<12} {'N samples'}")
    print("-" * 70)
    
    summary = {}
    if correlations_per_stage:
        for stage_idx in sorted(correlations_per_stage.keys()):
            corrs = correlations_per_stage[stage_idx]
            if corrs:
                mean_corr = np.mean(corrs)
                std_corr = np.std(corrs)
                summary[stage_idx] = {
                    'mean_correlation': mean_corr,
                    'std_correlation': std_corr,
                    'n_samples': len(corrs),
                }
                print(f"Stage {stage_idx:<4} {mean_corr:.4f}               {std_corr:.4f}        {len(corrs)}")
    else:
        print("  Note: State storage may not be enabled or states are not accessible.")
        print("  This test requires modifications to the model to store hidden states.")
        print("  Skipping state magnitude correlation test.")
    
    print("-" * 70)
    print("Interpretation: Positive correlation validates that controllability")
    print("                correctly identifies positions with active state dynamics.")
    
    return summary, correlations_per_stage


# ============================================================================
# FAITHFULNESS EVALUATION (Standard benchmark)
# ============================================================================

class FaithfulnessEvaluator:
    """Standard deletion/insertion faithfulness evaluation."""
    
    def __init__(self, model, device, num_steps=20):
        self.model = model
        self.device = device
        self.num_steps = num_steps
    
    def compute_deletion(self, image, saliency, target_class):
        image = image.to(self.device)
        saliency = saliency.to(self.device)
        
        h, w = image.shape[2], image.shape[3]
        n_pixels = h * w
        
        # G2: jitter to break ties deterministically.
        saliency_flat = (
            saliency.flatten()
            + 1e-9 * torch.randn_like(saliency.flatten())
        )
        sorted_indices = torch.argsort(saliency_flat, descending=True)
        # G1: per-channel baseline (shape [1, C, 1, 1] broadcasts).
        baseline = image.mean(dim=[2, 3], keepdim=True)
        
        scores = []
        for step in range(self.num_steps + 1):
            n_mask = int((step / self.num_steps) * n_pixels)
            mask = torch.ones(h * w, device=self.device)
            if n_mask > 0:
                mask[sorted_indices[:n_mask]] = 0
            mask = mask.reshape(1, 1, h, w)
            masked = image * mask + baseline * (1 - mask)
            
            with torch.no_grad():
                output = self.model(masked)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                scores.append(prob)
        
        return float(np.trapz(scores, np.linspace(0, 1, len(scores))))
    
    def compute_insertion(self, image, saliency, target_class):
        image = image.to(self.device)
        saliency = saliency.to(self.device)
        
        h, w = image.shape[2], image.shape[3]
        n_pixels = h * w
        
        saliency_flat = (
            saliency.flatten()
            + 1e-9 * torch.randn_like(saliency.flatten())
        )
        sorted_indices = torch.argsort(saliency_flat, descending=True)
        # G1: per-channel baseline, broadcast to full image shape.
        baseline = image.mean(dim=[2, 3], keepdim=True).expand_as(image)
        
        scores = []
        for step in range(self.num_steps + 1):
            n_reveal = int((step / self.num_steps) * n_pixels)
            mask = torch.zeros(h * w, device=self.device)
            if n_reveal > 0:
                mask[sorted_indices[:n_reveal]] = 1
            mask = mask.reshape(1, 1, h, w)
            revealed = baseline * (1 - mask) + image * mask
            
            with torch.no_grad():
                output = self.model(revealed)
                prob = F.softmax(output, dim=1)[0, target_class].item()
                scores.append(prob)
        
        return float(np.trapz(scores, np.linspace(0, 1, len(scores))))


def faithfulness_evaluation(model, dataloader, device, num_samples=50):
    """Standard faithfulness benchmark."""
    print("\n" + "=" * 70)
    print("FAITHFULNESS EVALUATION (Standard Benchmark)")
    print("=" * 70)
    
    ctrl_j = StructuralControllability(model, device, 'jacobian')
    ctrl_g = StructuralControllability(model, device, 'gramian')
    gradcam = GradCAMSaliency(model, device)
    random_sal = RandomSaliency()
    
    methods = {
        'Jacobian': ctrl_j,
        'Gramian': ctrl_g,
        'Grad-CAM': gradcam,
        'Random': random_sal,
    }
    
    evaluator = FaithfulnessEvaluator(model, device)
    
    results = {name: {'del': [], 'ins': []} for name in methods}
    
    sample_count = 0
    for images, labels in tqdm(dataloader, desc="Faithfulness",
                                total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break
        
        image = images[0:1].to(device)
        
        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(dim=1).item()
        
        for name, method in methods.items():
            try:
                saliency = method.generate(image, pred_class)
                del_auc = evaluator.compute_deletion(image, saliency, pred_class)
                ins_auc = evaluator.compute_insertion(image, saliency, pred_class)
                results[name]['del'].append(del_auc)
                results[name]['ins'].append(ins_auc)
            except Exception as e:
                continue
        
        sample_count += 1
    
    # Print results
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Deletion AUC':<15} {'Insertion AUC':<15} {'Score (I-D)'}")
    print("-" * 70)
    
    summary = {}
    for name in methods:
        if results[name]['del']:
            del_mean = np.mean(results[name]['del'])
            ins_mean = np.mean(results[name]['ins'])
            score = ins_mean - del_mean
            
            summary[name] = {
                'deletion': del_mean,
                'insertion': ins_mean,
                'score': score,
            }
            
            print(f"{name:<15} {del_mean:.4f}          {ins_mean:.4f}          {score:.4f}")
    
    print("-" * 70)
    
    # Rank by score
    ranked = sorted(summary.items(), key=lambda x: x[1]['score'], reverse=True)
    print(f"\nBest method: {ranked[0][0]} (score={ranked[0][1]['score']:.4f})")
    
    return summary, results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_perturbation_results(results, output_dir):
    """Plot perturbation invariance test results."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    methods = list(results.keys())
    high_drops = [results[m]['high_drop_mean'] for m in methods]
    low_drops = [results[m]['low_drop_mean'] for m in methods]
    ratios = [results[m]['ratio_mean'] for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = [METHOD_COLORS.get(m, '#333') for m in methods]
    
    # High drop
    axes[0].bar(methods, high_drops, color=colors)
    axes[0].set_ylabel('Confidence Drop')
    axes[0].set_title('(a) Drop when perturbing HIGH-saliency regions')
    axes[0].tick_params(axis='x', rotation=30)
    
    # Low drop
    axes[1].bar(methods, low_drops, color=colors)
    axes[1].set_ylabel('Confidence Drop')
    axes[1].set_title('(b) Drop when perturbing LOW-saliency regions')
    axes[1].tick_params(axis='x', rotation=30)
    
    # Ratio
    axes[2].bar(methods, ratios, color=colors)
    axes[2].axhline(1.0, color='red', linestyle='--', label='Ratio=1 (no difference)')
    axes[2].set_ylabel('Ratio (High/Low)')
    axes[2].set_title('(c) Ratio: Higher = better at identifying important regions')
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perturbation_invariance.png')
    plt.close()
    print(f"  ✓ Saved perturbation_invariance.png")


def plot_cross_class_results(results, output_dir):
    """Plot cross-class consistency results."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    methods = list(results.keys())
    correlations = [results[m]['mean_correlation'] for m in methods]
    stds = [results[m]['std_correlation'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#333') for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, correlations, yerr=stds, capsize=5, color=colors)
    
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect consistency')
    ax.axhline(0.0, color='red', linestyle='--', alpha=0.5, label='No consistency')
    
    ax.set_ylabel('Cross-Class Correlation')
    ax.set_title('Cross-Class Consistency Test\n(Higher = more class-agnostic/structural)')
    ax.set_ylim(-0.2, 1.2)
    ax.legend()
    
    for bar, corr in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{corr:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_class_consistency.png')
    plt.close()
    print(f"  ✓ Saved cross_class_consistency.png")


def plot_architecture_analysis(stage_stats, output_dir):
    """Plot architecture analysis results."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Collect per-stage means
    stages = sorted(stage_stats.keys())
    mean_ctrl = []
    mean_entropy = []
    
    for stage_idx in stages:
        all_mean = []
        all_entropy = []
        for block_idx in stage_stats[stage_idx]:
            stats = stage_stats[stage_idx][block_idx]
            all_mean.extend(stats['mean_j'])
            all_entropy.extend(stats['entropy_j'])
        mean_ctrl.append(np.mean(all_mean) if all_mean else 0)
        mean_entropy.append(np.mean(all_entropy) if all_entropy else 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Mean controllability per stage
    axes[0].bar(stages, mean_ctrl, color='#d62728')
    axes[0].set_xlabel('Stage')
    axes[0].set_ylabel('Mean Controllability')
    axes[0].set_title('(a) Controllability Across Stages\n(Higher = more information flow)')
    
    # Entropy per stage
    axes[1].bar(stages, mean_entropy, color='#1f77b4')
    axes[1].set_xlabel('Stage')
    axes[1].set_ylabel('Mean Entropy')
    axes[1].set_title('(b) Entropy Across Stages\n(Lower = more focused attention)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'architecture_analysis.png')
    plt.close()
    print(f"  ✓ Saved architecture_analysis.png")


def plot_faithfulness_comparison(summary, output_dir):
    """Plot faithfulness comparison."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    methods = list(summary.keys())
    scores = [summary[m]['score'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#333') for m in methods]
    
    # Sort by score
    order = np.argsort(scores)[::-1]
    methods = [methods[i] for i in order]
    scores = [scores[i] for i in order]
    colors = [colors[i] for i in order]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(methods, scores, color=colors)
    
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Faithfulness Score (Insertion - Deletion AUC)')
    ax.set_title('Faithfulness Evaluation\n(Higher = better)')
    ax.invert_yaxis()
    
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'faithfulness_comparison.png')
    plt.close()
    print(f"  ✓ Saved faithfulness_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Controllability Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bloodmnist")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Test-specific arguments
    parser.add_argument("--perturb_percent", type=float, default=0.1,
                        help="Percentage of pixels to perturb (Test 1)")
    parser.add_argument("--num_test_classes", type=int, default=5,
                        help="Number of classes to test for cross-class consistency (Test 2)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.output_dir is None:
        args.output_dir = f"./comprehensive_results_{args.dataset}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    apply_pub_style()
    
    print("=" * 70)
    print("COMPREHENSIVE CONTROLLABILITY EVALUATION")
    print("=" * 70)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  Output:      {output_dir}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load data
    dataset_type = DatasetType(args.dataset)
    dataset_info = get_dataset_info(dataset_type)
    num_classes = config.num_classes
    
    # ========================================
    # RUN ALL TESTS
    # ========================================
    
    all_results = {}
    
    # Test 1: Perturbation Invariance
    _, _, test_loader1 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )
    perturb_summary, perturb_raw = perturbation_invariance_test(
        model, test_loader1, device, 
        num_samples=args.num_samples,
        perturb_percent=args.perturb_percent,
    )
    all_results['perturbation_invariance'] = perturb_summary
    all_results['perturbation_invariance_raw'] = perturb_raw   # per-sample lists for bootstrap
    
    # Test 2: Cross-Class Consistency
    _, _, test_loader2 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )
    crossclass_summary, crossclass_raw = cross_class_consistency_test(
        model, test_loader2, device, num_classes,
        num_samples=min(args.num_samples, 30),
        num_test_classes=args.num_test_classes,
    )
    all_results['cross_class_consistency'] = crossclass_summary
    all_results['cross_class_consistency_raw'] = crossclass_raw   # per-sample correlations
    
    # Test 3: Architecture Analysis
    _, _, test_loader3 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )
    arch_summary, arch_raw = architecture_analysis(
        model, test_loader3, device,
        num_samples=min(args.num_samples, 20),
    )
    all_results['architecture_analysis'] = arch_summary
    all_results['architecture_analysis_raw'] = arch_raw
    
    # Test 4: State Magnitude Correlation
    _, _, test_loader4 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )
    state_summary, state_raw = state_magnitude_correlation_test(
        model, test_loader4, device,
        num_samples=min(args.num_samples, 20),
    )
    all_results['state_magnitude_correlation'] = state_summary
    
    # Faithfulness Evaluation
    _, _, test_loader5 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root,
    )
    faith_summary, faith_raw = faithfulness_evaluation(
        model, test_loader5, device,
        num_samples=args.num_samples,
    )
    all_results['faithfulness'] = faith_summary
    all_results['faithfulness_raw'] = faith_raw   # per-sample del/ins lists
    
    # ========================================
    # GENERATE FIGURES
    # ========================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    if perturb_summary:
        plot_perturbation_results(perturb_summary, output_dir)
    if crossclass_summary:
        plot_cross_class_results(crossclass_summary, output_dir)
    if arch_raw:
        plot_architecture_analysis(arch_raw, output_dir)
    if faith_summary:
        plot_faithfulness_comparison(faith_summary, output_dir)
    
    # ========================================
    # SAVE COMPREHENSIVE REPORT
    # ========================================
    
    report_path = output_dir / 'comprehensive_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE CONTROLLABILITY EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Samples: {args.num_samples}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TEST 1: PERTURBATION INVARIANCE\n")
        f.write("-" * 70 + "\n")
        f.write("Question: Do high-controllability regions actually influence predictions?\n\n")
        for name, stats in perturb_summary.items():
            f.write(f"  {name}:\n")
            f.write(f"    High-region drop: {stats['high_drop_mean']:.4f} ± {stats['high_drop_std']:.4f}\n")
            f.write(f"    Low-region drop:  {stats['low_drop_mean']:.4f} ± {stats['low_drop_std']:.4f}\n")
            f.write(f"    Ratio:            {stats['ratio_mean']:.2f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TEST 2: CROSS-CLASS CONSISTENCY\n")
        f.write("-" * 70 + "\n")
        f.write("Question: Is controllability structural (class-agnostic)?\n\n")
        for name, stats in crossclass_summary.items():
            f.write(f"  {name}: correlation = {stats['mean_correlation']:.4f} ± {stats['std_correlation']:.4f}\n")
        f.write("\n  Expected: Controllability ≈ 1.0 (structural), Grad-CAM < 1.0 (class-dependent)\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TEST 3: ARCHITECTURE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("Question: How does controllability vary across layers?\n\n")
        # Write stage summaries
        
        f.write("-" * 70 + "\n")
        f.write("TEST 4: STATE MAGNITUDE CORRELATION\n")
        f.write("-" * 70 + "\n")
        f.write("Question: Does high controllability correlate with large state magnitudes?\n\n")
        if state_summary:
            for stage, stats in state_summary.items():
                f.write(f"  Stage {stage}: correlation = {stats['mean_correlation']:.4f}\n")
        else:
            f.write("  (State storage not available - test skipped)\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("FAITHFULNESS EVALUATION\n")
        f.write("-" * 70 + "\n")
        for name, stats in sorted(faith_summary.items(), key=lambda x: x[1]['score'], reverse=True):
            f.write(f"  {name}: Del={stats['deletion']:.4f}, Ins={stats['insertion']:.4f}, Score={stats['score']:.4f}\n")
    
    print(f"\n  ✓ Saved comprehensive_report.txt")
    
    # Save raw results
    torch.save(_to_picklable(all_results), output_dir / 'all_results.pth')
    print(f"  ✓ Saved all_results.pth")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    print("\nKEY FINDINGS:")
    
    # Perturbation test
    if perturb_summary:
        j_ratio = perturb_summary.get('Jacobian', {}).get('ratio_mean', 0)
        gc_ratio = perturb_summary.get('Grad-CAM', {}).get('ratio_mean', 0)
        if j_ratio > gc_ratio:
            print(f"  ✓ Perturbation Test: Jacobian ({j_ratio:.2f}) > Grad-CAM ({gc_ratio:.2f})")
        else:
            print(f"  ~ Perturbation Test: Grad-CAM ({gc_ratio:.2f}) > Jacobian ({j_ratio:.2f})")
    
    # Cross-class consistency
    if crossclass_summary:
        j_corr = crossclass_summary.get('Jacobian', {}).get('mean_correlation', 0)
        gc_corr = crossclass_summary.get('Grad-CAM', {}).get('mean_correlation', 0)
        print(f"  ✓ Cross-Class Test: Jacobian consistency={j_corr:.3f}, Grad-CAM={gc_corr:.3f}")
        if j_corr > gc_corr + 0.1:
            print(f"    → Controllability is MORE structural (class-agnostic) than Grad-CAM")
    
    # Faithfulness
    if faith_summary:
        best = max(faith_summary.items(), key=lambda x: x[1]['score'])
        print(f"  Faithfulness: Best method = {best[0]} (score={best[1]['score']:.4f})")
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()