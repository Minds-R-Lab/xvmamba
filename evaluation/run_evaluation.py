"""
Final Evaluation: Original Controllability vs All Baselines

Uses the mask-based evaluator (which gives cleaner results)
to fairly compare:
  - Jacobian (Ours)
  - Gramian (Ours)  
  - Grad-CAM / InputGrad (baselines)
  - Random (baseline)

Generates publication-grade figures:
  1. Faithfulness curves (deletion + insertion)
  2. Faithfulness bar chart (summary scores)
  3. Deletion-vs-Insertion scatter plot
  4. Per-class faithfulness heatmap
  5. Per-class grouped bar chart
  6. Sparsity & entropy distributions (box plots)
  7. Jacobian–Gramian correlation scatter per class
  8. Combined multi-panel dashboard

Usage:
    python run_final_evaluation.py \
        --checkpoint ./checkpoints/dermamnist/best_model.pth \
        --dataset dermamnist \
        --num_samples 50
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

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VMambaClassifier, VMambaConfig
from controllability import ControllabilityAnalyzer
from data import DatasetType, get_dataloader, get_dataset_info

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# Publication Style Configuration
# ============================================================================

# Consistent colour palette across all figures
METHOD_COLORS = {
    'Jacobian (Ours)': '#d62728',   # red
    'Gramian (Ours)':  '#1f77b4',   # blue
    'Grad-CAM':        '#2ca02c',   # green
    'InputGrad':       '#9467bd',   # purple
    'Random':          '#7f7f7f',   # grey
}

METHOD_MARKERS = {
    'Jacobian (Ours)': 'o',
    'Gramian (Ours)':  's',
    'Grad-CAM':        '^',
    'InputGrad':       'D',
    'Random':          'X',
}

METHOD_LINESTYLES = {
    'Jacobian (Ours)': '-',
    'Gramian (Ours)':  '-',
    'Grad-CAM':        '--',
    'InputGrad':       '--',
    'Random':          ':',
}


def apply_pub_style():
    """Apply publication-quality matplotlib defaults."""
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
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.0,
    })


# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get('args', {})

    image_size = args.get('image_size', 224)
    patch_size = args.get('patch_size', 4)
    in_channels = args.get('in_channels', None)
    dims = args.get('dims', [32, 64, 128, 256])
    depths = args.get('depths', [2, 2, 4, 2])
    d_state = args.get('d_state', 16)
    num_classes = args.get('num_classes', 7)

    if in_channels is None:
        dataset = args.get('dataset', 'dermamnist')
        in_channels = 3 if dataset in ['dermamnist', 'bloodmnist',
                                        'pathmnist', 'retinamnist'] else 1

    config = VMambaConfig(
        image_size=image_size, patch_size=patch_size,
        in_channels=in_channels, dims=dims, depths=depths,
        d_state=d_state, num_classes=num_classes,
    )

    model = VMambaClassifier(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, config


# ============================================================================
# Saliency Methods
# ============================================================================

class ControllabilitySaliency:
    """Class-conditioned controllability saliency with spatial smoothing.

    Pipeline
    --------
    1.  Forward pass in **analysis mode** (no grad) → per-block SSM parameters
        → per-block controllability maps via ``ControllabilityAnalyzer``.
    2.  **Exponential layer weighting**: later blocks receive exponentially more
        weight (semantic features dominate over edges/textures).
    3.  **Gaussian smoothing**: the controllability map is computed at the token
        level (7×7 to 56×56 depending on the stage).  After bilinear upsampling
        to the input resolution (224×224), the map exhibits blocky artifacts
        because each token maps to a large square patch.  A Gaussian blur with
        sigma proportional to the patch size removes block boundaries while
        preserving the spatial structure.  This is standard practice — Grad-CAM
        also benefits from smooth CNN feature maps; our SSM maps need explicit
        smoothing to achieve the same effect.
    4.  (Optional) **Class conditioning**: a second forward pass with gradients
        computes the spatial relevance for the target class.  The structural
        controllability map is element-wise multiplied by this gradient map.
    5.  Normalize to [0, 1].

    Parameters
    ----------
    model : VMambaClassifier
    device : torch.device
    method : str
        ``'jacobian'`` or ``'gramian'``.
    layer_weighting : str
        ``'exponential'`` (default), ``'linear'``, or ``'uniform'``.
    class_conditioned : bool
        When True (default), modulate by gradient of target-class logit.
    smooth_sigma : float
        Sigma for Gaussian smoothing (in pixels at 224×224 scale).
        Default 12 works well for patch_size=4 with 4 downsample stages.
        Set to 0 to disable smoothing.
    """

    def __init__(self, model, device, method='jacobian',
                 layer_weighting='exponential',
                 class_conditioned=True,
                 smooth_sigma=1.0):
        self.model = model
        self.device = device
        self.method = method
        self.layer_weighting = layer_weighting
        self.class_conditioned = class_conditioned
        self.smooth_sigma = smooth_sigma
        self.analyzer = ControllabilityAnalyzer(method=method, normalize=True)

        # Pre-compute Gaussian kernel (if smoothing is enabled)
        self._gauss_kernel = None
        if smooth_sigma > 0:
            self._gauss_kernel = self._make_gaussian_kernel(smooth_sigma)

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    def generate(self, image, target_class=None):
        """Generate a saliency map for *image*.

        Returns
        -------
        saliency : Tensor [H, W], values in [0, 1]
        """
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]

        # ---- Step 1: Structural controllability map ----
        self.model.enable_analysis_mode(store_states=False)
        with torch.no_grad():
            output_a, analysis = self.model(image, return_analysis=True)

        if target_class is None:
            target_class = output_a.argmax(dim=1).item()

        ctrl_map = self._weighted_aggregate(analysis, target_size=(h, w))
        self.model.disable_analysis_mode()

        # ---- Step 2: Gaussian smoothing ----
        ctrl_map = self._smooth(ctrl_map)

        # ---- Step 3 (optional): Class conditioning ----
        if self.class_conditioned:
            grad_map = self._class_gradient_map(image, target_class)
            # Geometric mean balances both signals
            saliency = torch.sqrt(ctrl_map * grad_map + 1e-8)
        else:
            saliency = ctrl_map

        saliency = self._normalize(saliency)
        return saliency.cpu()

    # --------------------------------------------------------------------- #
    # Internals                                                               #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _make_gaussian_kernel(sigma, truncate=4.0):
        """Create a 2D Gaussian kernel as a [1, 1, K, K] conv weight.

        ``truncate`` controls the kernel radius: K = 2 * ceil(truncate * sigma) + 1.
        """
        radius = int(np.ceil(truncate * sigma))
        ks = 2 * radius + 1
        ax = torch.arange(ks, dtype=torch.float32) - radius
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, ks, ks)

    def _smooth(self, x):
        """Apply Gaussian smoothing to a [H, W] map."""
        if self._gauss_kernel is None:
            return x
        kernel = self._gauss_kernel.to(x.device, x.dtype)
        pad = kernel.shape[-1] // 2
        # [H, W] -> [1, 1, H, W] -> conv -> [H, W]
        smoothed = F.conv2d(
            x.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=pad,
        ).squeeze()
        return smoothed

    def _weighted_aggregate(self, analysis_output, target_size):
        """Collect per-block maps, upsample to *target_size*, weight, sum."""
        layer_maps = []

        for stage_idx, stage_caches in enumerate(analysis_output.stage_caches):
            for block_idx, cache in enumerate(stage_caches):
                result = self.analyzer.analyze_block(
                    cache, stage_idx=stage_idx, block_idx=block_idx,
                )
                imap = result.influence_map
                if imap is None:
                    continue

                # Upsample to target resolution
                if imap.shape[0] != target_size[0] or imap.shape[1] != target_size[1]:
                    imap = F.interpolate(
                        imap.unsqueeze(0).unsqueeze(0),
                        size=target_size, mode='bilinear', align_corners=False,
                    ).squeeze()

                layer_maps.append(imap)

        if not layer_maps:
            return torch.ones(target_size, device=self.device) * 0.5

        stacked = torch.stack(layer_maps, dim=0)          # [L, H, W]
        weights = self._layer_weights(len(layer_maps), stacked.device)
        aggregated = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
        return self._normalize(aggregated)

    def _layer_weights(self, n, device):
        """Compute normalised layer weights."""
        if self.layer_weighting == 'exponential':
            raw = torch.exp(torch.linspace(0, 2, n, device=device))
        elif self.layer_weighting == 'linear':
            raw = torch.arange(1, n + 1, dtype=torch.float32, device=device)
        else:  # uniform
            raw = torch.ones(n, device=device)
        return raw / raw.sum()

    def _class_gradient_map(self, image, target_class):
        """Spatial gradient of the target-class logit w.r.t. the input.

        Returns a normalised [H, W] map (also smoothed for consistency).
        """
        self.model.zero_grad()
        img = image.clone().detach().requires_grad_(True)
        output = self.model(img)

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # |dL/dx| averaged over channels → spatial map [H, W]
        grad_map = img.grad.abs().mean(dim=1).squeeze(0)
        # Smooth the gradient map too — raw pixel gradients are noisy
        grad_map = self._smooth(grad_map)
        return self._normalize(grad_map)

    @staticmethod
    def _normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin)


class GradCAMSaliency:
    """Grad-CAM baseline via hook on last stage."""

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

        last_block = self.model.stages[-1].blocks[-1]
        last_block.register_forward_hook(fwd_hook)
        last_block.register_full_backward_hook(bwd_hook)

    def generate(self, image, target_class=None):
        self.model.zero_grad()
        image = image.to(self.device).requires_grad_(True)

        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is not None and self.activations is not None:
            weights = self.gradients.mean(dim=(1, 2), keepdim=True)
            cam = (weights * self.activations).sum(dim=-1)
            cam = F.relu(cam).squeeze(0)

            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            h, w = image.shape[2], image.shape[3]
            if cam.shape[0] != h or cam.shape[1] != w:
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),
                    size=(h, w), mode='bilinear', align_corners=False
                ).squeeze()

            return cam.cpu()
        else:
            return self._fallback(image, target_class)

    def _fallback(self, image, target_class):
        """Fallback to input gradient."""
        self.model.zero_grad()
        image = image.clone().detach().requires_grad_(True)
        output = self.model(image)
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        grad = image.grad.abs().mean(dim=1).squeeze(0)
        if grad.max() > 0:
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        return grad.cpu()


class InputGradSaliency:
    """Input x Gradient saliency."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def generate(self, image, target_class=None):
        self.model.zero_grad()
        image = image.clone().detach().to(self.device).requires_grad_(True)
        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        saliency = (image.grad * image).abs().mean(dim=1).squeeze(0)
        if saliency.max() > 0:
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        return saliency.cpu()


class RandomSaliency:
    """Random baseline."""

    def generate(self, image, target_class=None):
        h, w = image.shape[2], image.shape[3]
        return torch.rand(h, w)


# ============================================================================
# Faithfulness Evaluator  (mask-based – returns per-step scores AND AUC)
# ============================================================================

class FaithfulnessEvaluator:
    """Pixel-perturbation faithfulness evaluator.

    Both ``compute_deletion`` and ``compute_insertion`` now return a tuple of
    ``(auc, step_scores)`` so that per-step confidence curves can be recorded
    for plotting.
    """

    def __init__(self, model, device, num_steps=20):
        self.model = model
        self.device = device
        self.num_steps = num_steps

    # ------------------------------------------------------------------
    def compute_deletion(self, image, saliency, target_class):
        """Return (deletion_auc, list_of_per_step_confidences)."""
        image = image.to(self.device)
        saliency = saliency.to(self.device)

        h, w = image.shape[2], image.shape[3]
        n_pixels = h * w

        saliency_flat = saliency.flatten()
        sorted_indices = torch.argsort(saliency_flat, descending=True)
        baseline = image.mean()

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

        auc = float(np.trapz(scores, np.linspace(0, 1, len(scores))))
        return auc, scores

    # ------------------------------------------------------------------
    def compute_insertion(self, image, saliency, target_class):
        """Return (insertion_auc, list_of_per_step_confidences)."""
        image = image.to(self.device)
        saliency = saliency.to(self.device)

        h, w = image.shape[2], image.shape[3]
        n_pixels = h * w

        saliency_flat = saliency.flatten()
        sorted_indices = torch.argsort(saliency_flat, descending=True)
        baseline = torch.ones_like(image) * image.mean()

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

        auc = float(np.trapz(scores, np.linspace(0, 1, len(scores))))
        return auc, scores

    # ---- Convenience wrappers (backward-compatible scalar-only API) ----
    def compute_deletion_auc(self, image, saliency, target_class):
        auc, _ = self.compute_deletion(image, saliency, target_class)
        return auc

    def compute_insertion_auc(self, image, saliency, target_class):
        auc, _ = self.compute_insertion(image, saliency, target_class)
        return auc


# ============================================================================
# Per-Class Analysis
# ============================================================================

def run_per_class_analysis(model, dataloader, device, class_names, num_samples=100):
    """Per-class controllability analysis.

    Uses **structural-only** (non-class-conditioned) controllability so that
    the sparsity, entropy, and J–G correlation metrics reflect the raw SSM
    dynamics rather than the gradient modulation.
    """
    print("\n" + "=" * 70)
    print("PER-CLASS ANALYSIS")
    print("=" * 70)

    # Structural (non-class-conditioned) analyzers for diagnostics
    analyzer_j = ControllabilitySaliency(model, device, 'jacobian',
                                         class_conditioned=False,
                                         smooth_sigma=1.0)
    analyzer_g = ControllabilitySaliency(model, device, 'gramian',
                                         class_conditioned=False,
                                         smooth_sigma=1.0)

    num_classes = len(class_names)
    class_data = {i: {
        'sparsity_j': [], 'sparsity_g': [],
        'entropy_j': [], 'entropy_g': [],
        'correlation': [],
        'correct': 0, 'total': 0,
        'confidence': [],
    } for i in range(num_classes)}

    sample_count = 0
    for images, labels in tqdm(dataloader, desc="Per-class", total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break

        image = images[0:1].to(device)
        label = labels[0].item()

        try:
            with torch.no_grad():
                output = model(image)
                pred = output.argmax(dim=1).item()
                conf = F.softmax(output, dim=1)[0, pred].item()

            sal_j = analyzer_j.generate(image)
            sal_g = analyzer_g.generate(image)

            # Sparsity
            class_data[label]['sparsity_j'].append((sal_j > 0.5).float().mean().item())
            class_data[label]['sparsity_g'].append((sal_g > 0.5).float().mean().item())

            # Entropy
            flat_j = sal_j.flatten()
            flat_j = flat_j / (flat_j.sum() + 1e-8)
            ent_j = -torch.sum(flat_j * torch.log(flat_j + 1e-8)).item()
            class_data[label]['entropy_j'].append(ent_j)

            flat_g = sal_g.flatten()
            flat_g = flat_g / (flat_g.sum() + 1e-8)
            ent_g = -torch.sum(flat_g * torch.log(flat_g + 1e-8)).item()
            class_data[label]['entropy_g'].append(ent_g)

            # Correlation
            corr = torch.corrcoef(torch.stack([sal_j.flatten(), sal_g.flatten()]))[0, 1].item()
            class_data[label]['correlation'].append(corr)

            class_data[label]['correct'] += (pred == label)
            class_data[label]['total'] += 1
            class_data[label]['confidence'].append(conf)

        except Exception as e:
            print(f"Error: {e}")
            continue

        sample_count += 1

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Class':<20} {'Acc':<8} {'J-Spars':<10} {'G-Spars':<10} {'J-G Corr':<10}")
    print("-" * 70)

    summary = {}
    for i in range(num_classes):
        if class_data[i]['total'] == 0:
            continue
        acc = class_data[i]['correct'] / class_data[i]['total']
        js = np.mean(class_data[i]['sparsity_j'])
        gs = np.mean(class_data[i]['sparsity_g'])
        corr = np.mean(class_data[i]['correlation'])

        summary[class_names[i]] = {
            'accuracy': acc, 'j_sparsity': js, 'g_sparsity': gs,
            'j_entropy': np.mean(class_data[i]['entropy_j']),
            'g_entropy': np.mean(class_data[i]['entropy_g']),
            'jg_correlation': corr,
            'n_samples': class_data[i]['total'],
            'confidence': np.mean(class_data[i]['confidence']),
        }

        print(f"{class_names[i]:<20} {acc:.3f}    {js:.4f}     {gs:.4f}     {corr:.4f}")

    return summary, class_data


# ============================================================================
# Per-Class Faithfulness  (per-class del/ins for each method)
# ============================================================================

def run_per_class_faithfulness(model, dataloader, device, methods,
                               evaluator, class_names, num_samples=50):
    """Evaluate faithfulness per class for all methods.

    Returns dict: {class_name: {method_name: {'del': float, 'ins': float, 'score': float}}}
    """
    print("\n" + "=" * 70)
    print("PER-CLASS FAITHFULNESS")
    print("=" * 70)

    num_classes = len(class_names)
    pc = {i: {name: {'del': [], 'ins': []} for name in methods} for i in range(num_classes)}

    sample_count = 0
    for images, labels in tqdm(dataloader, desc="PC-Faith", total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break

        image = images[0:1].to(device)
        label = labels[0].item()

        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(dim=1).item()

        for name, method in methods.items():
            try:
                saliency = method.generate(image, pred_class)
                d = evaluator.compute_deletion_auc(image, saliency, pred_class)
                i_auc = evaluator.compute_insertion_auc(image, saliency, pred_class)
                pc[label][name]['del'].append(d)
                pc[label][name]['ins'].append(i_auc)
            except Exception:
                pass

        sample_count += 1

    # Aggregate
    pc_summary = {}
    for ci in range(num_classes):
        if not any(pc[ci][n]['del'] for n in methods):
            continue
        cname = class_names[ci]
        pc_summary[cname] = {}
        for name in methods:
            if pc[ci][name]['del']:
                dm = np.mean(pc[ci][name]['del'])
                im = np.mean(pc[ci][name]['ins'])
                pc_summary[cname][name] = {'del': dm, 'ins': im, 'score': im - dm}
            else:
                pc_summary[cname][name] = {'del': 0.0, 'ins': 0.0, 'score': 0.0}

    # Print
    header = f"{'Class':<16}" + "".join(f"{n:<16}" for n in methods)
    print("\n" + header)
    print("-" * len(header))
    for cname, mdata in pc_summary.items():
        row = f"{cname:<16}"
        for n in methods:
            row += f"{mdata[n]['score']:<16.4f}"
        print(row)

    return pc_summary


# ============================================================================
# Figure 1 – Faithfulness Curves
# ============================================================================

def plot_faithfulness_curves(all_curves, output_dir):
    """Two-panel deletion / insertion curves with mean +/- std shading."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.linspace(0, 100, 21)

    # ---------- Deletion ----------
    ax = axes[0]
    for name, curves in all_curves.items():
        if not curves['deletion']:
            continue
        arr = np.array(curves['deletion'])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        c = METHOD_COLORS.get(name, '#000')
        ls = METHOD_LINESTYLES.get(name, '-')
        ax.plot(x, mean, color=c, ls=ls, label=name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel('Pixels Removed (%)')
    ax.set_ylabel('Prediction Confidence')
    ax.set_title('(a)  Deletion Curve')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    # ---------- Insertion ----------
    ax = axes[1]
    for name, curves in all_curves.items():
        if not curves['insertion']:
            continue
        arr = np.array(curves['insertion'])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        c = METHOD_COLORS.get(name, '#000')
        ls = METHOD_LINESTYLES.get(name, '-')
        ax.plot(x, mean, color=c, ls=ls, label=name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlabel('Pixels Revealed (%)')
    ax.set_ylabel('Prediction Confidence')
    ax.set_title('(b)  Insertion Curve')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out = output_dir / 'fig1_faithfulness_curves.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 2 – Faithfulness Bar Chart
# ============================================================================

def plot_summary_bars(summary, output_dir):
    """Horizontal grouped bars: deletion AUC, insertion AUC, score."""
    if not MATPLOTLIB_AVAILABLE:
        return

    names = list(summary.keys())
    scores = np.array([summary[m]['score'] for m in names])
    del_aucs = np.array([summary[m]['deletion'] for m in names])
    ins_aucs = np.array([summary[m]['insertion'] for m in names])
    del_stds = np.array([summary[m]['deletion_std'] for m in names])
    ins_stds = np.array([summary[m]['insertion_std'] for m in names])

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    names = [names[i] for i in order]
    scores = scores[order]
    del_aucs = del_aucs[order]
    ins_aucs = ins_aucs[order]
    del_stds = del_stds[order]
    ins_stds = ins_stds[order]
    colors = [METHOD_COLORS.get(n, '#333') for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Deletion AUC (lower is better)
    ax = axes[0]
    bars = ax.barh(names, del_aucs, xerr=del_stds, color=colors, capsize=3,
                   edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Deletion AUC  (lower is better)')
    ax.set_title('(a)  Deletion AUC')
    ax.invert_yaxis()
    for bar, v in zip(bars, del_aucs):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f'{v:.3f}',
                va='center', fontsize=9)

    # Insertion AUC (higher is better)
    ax = axes[1]
    bars = ax.barh(names, ins_aucs, xerr=ins_stds, color=colors, capsize=3,
                   edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Insertion AUC  (higher is better)')
    ax.set_title('(b)  Insertion AUC')
    ax.invert_yaxis()
    for bar, v in zip(bars, ins_aucs):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f'{v:.3f}',
                va='center', fontsize=9)

    # Overall score
    ax = axes[2]
    bars = ax.barh(names, scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Score = Ins - Del  (higher is better)')
    ax.set_title('(c)  Overall Faithfulness')
    ax.invert_yaxis()
    for bar, v in zip(bars, scores):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2, f'{v:.4f}',
                va='center', fontsize=9)

    plt.tight_layout()
    out = output_dir / 'fig2_faithfulness_bars.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 3 – Deletion-vs-Insertion Scatter
# ============================================================================

def plot_del_ins_scatter(results, output_dir):
    """Each method is a point cloud (sample-level) + a large mean marker."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for name in results:
        if not results[name]['del_auc']:
            continue
        dels = np.array(results[name]['del_auc'])
        inss = np.array(results[name]['ins_auc'])
        c = METHOD_COLORS.get(name, '#333')
        mk = METHOD_MARKERS.get(name, 'o')

        # Individual samples (small, transparent)
        ax.scatter(dels, inss, c=c, marker=mk, s=18, alpha=0.35,
                   edgecolors='none')
        # Mean (large)
        ax.scatter(dels.mean(), inss.mean(), c=c, marker=mk, s=140,
                   edgecolors='black', linewidths=0.8, label=name, zorder=5)

    # Ideal zone annotation
    ax.annotate('Better  ->', xy=(0.02, 0.95), fontsize=9, color='#666',
                xycoords='axes fraction')
    ax.annotate('Better (up)', xy=(0.02, 0.02), fontsize=9, color='#666',
                xycoords='axes fraction', rotation=90)

    # Diagonal reference
    lims = [0, 1]
    ax.plot(lims, lims, ls=':', color='#aaa', lw=0.8, zorder=0)

    ax.set_xlabel('Deletion AUC  (lower is better)')
    ax.set_ylabel('Insertion AUC  (higher is better)')
    ax.set_title('Deletion vs Insertion (Per Sample)')
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc', loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out = output_dir / 'fig3_del_vs_ins_scatter.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 4 – Per-Class Faithfulness Heatmap
# ============================================================================

def plot_per_class_heatmap(pc_summary, output_dir):
    """Heatmap: rows = classes, columns = methods, values = faithfulness score."""
    if not MATPLOTLIB_AVAILABLE or not pc_summary:
        return

    classes = list(pc_summary.keys())
    methods = list(next(iter(pc_summary.values())).keys())
    data = np.array([[pc_summary[c][m]['score'] for m in methods] for c in classes])

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.5),
                                    max(4, len(classes) * 0.65)))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)

    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(methods)):
            val = data[i, j]
            color = 'white' if abs(val - data.mean()) > 0.5 * data.std() else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                    color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Faithfulness Score (Ins - Del)')
    ax.set_title('Per-Class Faithfulness Score by Method')

    plt.tight_layout()
    out = output_dir / 'fig4_per_class_heatmap.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 5 – Per-Class Grouped Bars
# ============================================================================

def plot_per_class_grouped_bars(pc_summary, output_dir):
    """Grouped bar chart: one cluster per class, one bar per method."""
    if not MATPLOTLIB_AVAILABLE or not pc_summary:
        return

    classes = list(pc_summary.keys())
    methods = list(next(iter(pc_summary.values())).keys())
    n_classes = len(classes)
    n_methods = len(methods)

    bar_width = 0.8 / n_methods
    x = np.arange(n_classes)

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.4), 5))

    for j, m in enumerate(methods):
        vals = [pc_summary[c][m]['score'] for c in classes]
        c = METHOD_COLORS.get(m, '#333')
        offset = (j - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width, label=m, color=c,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha='right')
    ax.set_ylabel('Faithfulness Score (Ins - Del)')
    ax.set_title('Per-Class Faithfulness Comparison')
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc',
              fontsize=8, ncol=min(3, n_methods))
    ax.axhline(0, color='black', lw=0.5)

    plt.tight_layout()
    out = output_dir / 'fig5_per_class_bars.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 6 – Sparsity & Entropy Box Plots
# ============================================================================

def plot_sparsity_entropy(class_data, class_names, output_dir):
    """Box plots of Jacobian/Gramian sparsity and entropy distributions."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Gather all values across classes
    j_sparsity, g_sparsity = [], []
    j_entropy, g_entropy = [], []
    for i in class_data:
        j_sparsity.extend(class_data[i]['sparsity_j'])
        g_sparsity.extend(class_data[i]['sparsity_g'])
        j_entropy.extend(class_data[i]['entropy_j'])
        g_entropy.extend(class_data[i]['entropy_g'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Sparsity
    ax = axes[0]
    bp = ax.boxplot(
        [j_sparsity, g_sparsity],
        labels=['Jacobian', 'Gramian'],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='black', linewidth=1.5),
    )
    bp['boxes'][0].set_facecolor(METHOD_COLORS['Jacobian (Ours)'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(METHOD_COLORS['Gramian (Ours)'])
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel('Sparsity (fraction > 0.5)')
    ax.set_title('(a)  Saliency Sparsity')

    # Entropy
    ax = axes[1]
    bp = ax.boxplot(
        [j_entropy, g_entropy],
        labels=['Jacobian', 'Gramian'],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='black', linewidth=1.5),
    )
    bp['boxes'][0].set_facecolor(METHOD_COLORS['Jacobian (Ours)'])
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(METHOD_COLORS['Gramian (Ours)'])
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('(b)  Saliency Entropy')

    plt.tight_layout()
    out = output_dir / 'fig6_sparsity_entropy.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 7 – Jacobian-Gramian Correlation Per Class
# ============================================================================

def plot_jg_correlation(pc_ctrl_summary, output_dir):
    """Horizontal bar chart: J-G correlation coefficient per class."""
    if not MATPLOTLIB_AVAILABLE or not pc_ctrl_summary:
        return

    classes = list(pc_ctrl_summary.keys())
    corrs = [pc_ctrl_summary[c]['jg_correlation'] for c in classes]
    accs = [pc_ctrl_summary[c]['accuracy'] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, len(classes) * 0.55)))

    # Correlation bars
    ax = axes[0]
    y = np.arange(len(classes))
    bar_colors = ['#1f77b4' if c > 0 else '#d62728' for c in corrs]
    ax.barh(y, corrs, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Pearson Correlation (Jacobian vs Gramian)')
    ax.set_title('(a)  J-G Saliency Correlation')
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()

    # Accuracy bars
    ax = axes[1]
    ax.barh(y, accs, color='#2ca02c', edgecolor='white', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Classification Accuracy')
    ax.set_title('(b)  Per-Class Accuracy')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for i, v in enumerate(accs):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    out = output_dir / 'fig7_jg_correlation.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 8 – Combined Dashboard (multi-panel overview)
# ============================================================================

def plot_dashboard(summary, all_curves, pc_summary_faith, pc_ctrl_summary,
                   output_dir):
    """Four-panel dashboard combining key results into one figure."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # --- Panel A: Deletion Curves ---
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 100, 21)
    for name, curves in all_curves.items():
        if not curves['deletion']:
            continue
        arr = np.array(curves['deletion'])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        c = METHOD_COLORS.get(name, '#000')
        ls = METHOD_LINESTYLES.get(name, '-')
        ax.plot(x, mean, color=c, ls=ls, label=name, lw=1.8)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.12)
    ax.set_xlabel('Pixels Removed (%)')
    ax.set_ylabel('Confidence')
    ax.set_title('(a)  Deletion Curves')
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='#ccc')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    # --- Panel B: Insertion Curves ---
    ax = fig.add_subplot(gs[0, 1])
    for name, curves in all_curves.items():
        if not curves['insertion']:
            continue
        arr = np.array(curves['insertion'])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        c = METHOD_COLORS.get(name, '#000')
        ls = METHOD_LINESTYLES.get(name, '-')
        ax.plot(x, mean, color=c, ls=ls, label=name, lw=1.8)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.12)
    ax.set_xlabel('Pixels Revealed (%)')
    ax.set_ylabel('Confidence')
    ax.set_title('(b)  Insertion Curves')
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor='#ccc')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    # --- Panel C: Score Bar Chart ---
    ax = fig.add_subplot(gs[1, 0])
    names = list(summary.keys())
    scores = [summary[m]['score'] for m in names]
    order = np.argsort(scores)[::-1]
    names_sorted = [names[i] for i in order]
    scores_sorted = [scores[i] for i in order]
    colors = [METHOD_COLORS.get(n, '#333') for n in names_sorted]
    bars = ax.barh(names_sorted, scores_sorted, color=colors,
                   edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Faithfulness Score (Ins - Del)')
    ax.set_title('(c)  Overall Faithfulness Ranking')
    ax.invert_yaxis()
    for bar, v in zip(bars, scores_sorted):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=8)

    # --- Panel D: Per-Class Heatmap ---
    ax = fig.add_subplot(gs[1, 1])
    if pc_summary_faith:
        classes = list(pc_summary_faith.keys())
        methods_list = list(next(iter(pc_summary_faith.values())).keys())
        data = np.array([[pc_summary_faith[c][m]['score']
                          for m in methods_list] for c in classes])
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(methods_list)))
        ax.set_xticklabels(methods_list, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(methods_list)):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title('(d)  Per-Class Faithfulness')

    out = output_dir / 'fig8_dashboard.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Utility: Image Denormalization
# ============================================================================

def denormalize_image(tensor, mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225)):
    """Reverse ImageNet normalization for display."""
    img = tensor.detach().cpu().clone().float()
    for c in range(min(img.shape[0], len(mean))):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img.permute(1, 2, 0).numpy()


# ============================================================================
# Data Collection: Layer-wise Controllability
# ============================================================================

def collect_layerwise_data(model, dataloader, device, num_samples=10):
    """Collect per-stage/block controllability statistics."""
    from controllability import ControllabilityAnalyzer as _CA

    analyzer_j = _CA(method='jacobian', normalize=True)
    analyzer_g = _CA(method='gramian', normalize=True)

    layerwise = {'jacobian': {}, 'gramian': {}}

    sample_count = 0
    for images, labels in tqdm(dataloader, desc="Layerwise",
                               total=min(num_samples, len(dataloader))):
        if sample_count >= num_samples:
            break

        image = images[0:1].to(device)

        model.enable_analysis_mode(store_states=False)
        with torch.no_grad():
            _, analysis = model(image, return_analysis=True)

        for mname, analyzer in [('jacobian', analyzer_j),
                                ('gramian', analyzer_g)]:
            try:
                full = analyzer.analyze(analysis)
            except Exception:
                continue

            for stage_idx, stage_results in enumerate(full.layer_results):
                for block_idx, result in enumerate(stage_results):
                    key = (stage_idx, block_idx)
                    if key not in layerwise[mname]:
                        layerwise[mname][key] = {
                            'mean_influence': [],
                            'max_influence': [],
                            'sparsity': [],
                            'entropy': [],
                        }

                    imap = result.influence_map
                    if imap is None:
                        continue
                    stats = layerwise[mname][key]
                    stats['mean_influence'].append(imap.mean().item())
                    stats['max_influence'].append(imap.max().item())
                    stats['sparsity'].append(
                        (imap > 0.5).float().mean().item())
                    flat = imap.flatten()
                    flat = flat / (flat.sum() + 1e-8)
                    ent = -torch.sum(flat * torch.log(flat + 1e-8)).item()
                    stats['entropy'].append(ent)

        model.disable_analysis_mode()
        sample_count += 1

    return layerwise


# ============================================================================
# Data Collection: Qualitative Examples
# ============================================================================

def collect_qualitative_examples(model, methods, dataloader, device,
                                  class_names, num_examples=4):
    """Collect sample images + all saliency maps for visual comparison."""
    examples = []
    seen_classes = set()

    for images, labels in dataloader:
        if len(examples) >= num_examples:
            break

        label = labels[0].item()
        if label in seen_classes and len(examples) < num_examples // 2:
            continue

        image = images[0:1].to(device)

        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(dim=1).item()

        saliency_maps = {}
        for name, method in methods.items():
            try:
                sal = method.generate(image, pred_class)
                saliency_maps[name] = sal.cpu().numpy()
            except Exception:
                pass

        if len(saliency_maps) < 2:
            continue

        examples.append({
            'image': denormalize_image(image[0]),
            'label': label,
            'pred': pred_class,
            'saliency': saliency_maps,
        })
        seen_classes.add(label)

    return examples


# ============================================================================
# Figure 9 – Layer-wise Controllability
# ============================================================================

def plot_layerwise_controllability(layerwise_stats, output_dir):
    """Four-panel chart: mean influence, max influence, sparsity, entropy
    across every stage/block for Jacobian and Gramian."""
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    metrics = ['mean_influence', 'max_influence', 'sparsity', 'entropy']
    titles = ['(a)  Mean Influence', '(b)  Max Influence',
              '(c)  Sparsity', '(d)  Entropy']
    ylabels = ['Mean Influence Score', 'Max Influence Score',
               'Fraction > 0.5', 'Entropy (nats)']

    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        for mname, color, marker in [
            ('jacobian', METHOD_COLORS['Jacobian (Ours)'], 'o'),
            ('gramian', METHOD_COLORS['Gramian (Ours)'], 's'),
        ]:
            keys = sorted(layerwise_stats[mname].keys())
            if not keys:
                continue
            x_labels = [f"S{s}.B{b}" for s, b in keys]
            vals = [np.mean(layerwise_stats[mname][k][metric])
                    for k in keys]
            stds = [np.std(layerwise_stats[mname][k][metric])
                    for k in keys]

            x = np.arange(len(keys))
            label = 'Jacobian' if mname == 'jacobian' else 'Gramian'
            ax.errorbar(x, vals, yerr=stds, marker=marker, capsize=3,
                        color=color, label=label, linewidth=1.5,
                        markersize=5)

        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Layer (Stage.Block)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')

    plt.suptitle('Layer-wise Controllability Analysis', fontsize=14, y=1.01)
    plt.tight_layout()
    out = output_dir / 'fig9_layerwise_controllability.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 10 – Qualitative Saliency Comparison (overlay grid)
# ============================================================================

def plot_qualitative_comparison(examples, class_names, output_dir):
    """Grid: rows = samples, cols = [Input | Jacobian | Gramian | GradCAM |
    InputGrad | Random].  Saliency is overlaid semi-transparently."""
    if not MATPLOTLIB_AVAILABLE or not examples:
        return

    method_order = ['Jacobian (Ours)', 'Gramian (Ours)', 'Grad-CAM',
                    'InputGrad', 'Random']
    present = set()
    for ex in examples:
        present.update(ex['saliency'].keys())
    method_order = [m for m in method_order if m in present]

    n_rows = len(examples)
    n_cols = 1 + len(method_order)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.4, n_rows * 2.4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, ex in enumerate(examples):
        axes[row, 0].imshow(ex['image'])
        cname = (class_names[ex['label']]
                 if ex['label'] < len(class_names)
                 else f"Class {ex['label']}")
        axes[row, 0].set_ylabel(cname, fontsize=9, rotation=90,
                                labelpad=10)

        for col, mname in enumerate(method_order, 1):
            ax = axes[row, col]
            ax.imshow(ex['image'])
            if mname in ex['saliency']:
                sal = ex['saliency'][mname]
                ax.imshow(sal, cmap='jet', alpha=0.50, vmin=0, vmax=1)

    col_titles = ['Input'] + method_order
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Qualitative Saliency Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = output_dir / 'fig10_qualitative_comparison.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Figure 11 – Detailed Saliency Maps (heat-maps with colour bars)
# ============================================================================

def plot_saliency_maps(examples, class_names, output_dir):
    """For the first example: show input + individual saliency heat-maps
    with their own colour bar – no overlay, just the raw map."""
    if not MATPLOTLIB_AVAILABLE or not examples:
        return

    ex = examples[0]
    method_order = ['Jacobian (Ours)', 'Gramian (Ours)', 'Grad-CAM',
                    'InputGrad', 'Random']
    method_order = [m for m in method_order if m in ex['saliency']]

    n_panels = 1 + len(method_order)
    n_cols = min(n_panels, 6)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.8))
    axes = np.array(axes).flatten()

    axes[0].imshow(ex['image'])
    cname = (class_names[ex['label']]
             if ex['label'] < len(class_names)
             else f"Class {ex['label']}")
    pred_name = (class_names[ex['pred']]
                 if ex['pred'] < len(class_names)
                 else f"Class {ex['pred']}")
    axes[0].set_title('Input', fontsize=10, fontweight='bold')
    axes[0].set_xlabel(f'GT: {cname}\nPred: {pred_name}', fontsize=8)

    for i, mname in enumerate(method_order, 1):
        ax = axes[i]
        sal = ex['saliency'][mname]
        im = ax.imshow(sal, cmap='hot', vmin=0, vmax=1)
        ax.set_title(mname, fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    for ax in axes[:n_panels]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Saliency Maps (Single Sample)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / 'fig11_saliency_maps.png'
    plt.savefig(out)
    plt.close()
    print(f"  ✓ Saved {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dermamnist")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_class_cond", action="store_true",
                        help="Disable class conditioning (use raw structural maps)")
    parser.add_argument("--layer_weighting", type=str, default="exponential",
                        choices=["exponential", "linear", "uniform"],
                        help="Layer weighting strategy")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (0 to disable)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Default output dir based on dataset name
    if args.output_dir is None:
        args.output_dir = f"./final_results_{args.dataset}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_pub_style()

    use_class_cond = not args.no_class_cond

    print("=" * 70)
    print("FINAL EVALUATION: Controllability vs Baselines")
    print("=" * 70)
    print(f"  Dataset:           {args.dataset}")
    print(f"  Layer weighting:   {args.layer_weighting}")
    print(f"  Class conditioned: {use_class_cond}")
    print(f"  Smooth sigma:      {args.smooth_sigma}")
    print(f"  Num samples:       {args.num_samples}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load data
    dataset_type = DatasetType(args.dataset)
    dataset_info = get_dataset_info(dataset_type)
    class_names = dataset_info.get('classes',
                                   [f'Class_{i}' for i in range(config.num_classes)])

    _, _, test_loader = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root, npz_path=args.npz_path,
    )

    # ---- Initialize methods ----
    methods = {
        'Jacobian (Ours)': ControllabilitySaliency(
            model, device, method='jacobian',
            layer_weighting=args.layer_weighting,
            class_conditioned=use_class_cond,
            smooth_sigma=args.smooth_sigma,
        ),
        'Gramian (Ours)': ControllabilitySaliency(
            model, device, method='gramian',
            layer_weighting=args.layer_weighting,
            class_conditioned=use_class_cond,
            smooth_sigma=args.smooth_sigma,
        ),
        'Grad-CAM': GradCAMSaliency(model, device),
        'InputGrad': InputGradSaliency(model, device),
        'Random': RandomSaliency(),
    }

    evaluator = FaithfulnessEvaluator(model, device, num_steps=20)

    # ========================================
    # PART 1: Faithfulness  (with per-step curves)
    # ========================================
    print(f"\n{'='*70}")
    print("PART 1: FAITHFULNESS EVALUATION")
    print(f"{'='*70}")

    results = {name: {'del_auc': [], 'ins_auc': []} for name in methods}
    all_curves = {name: {'deletion': [], 'insertion': []} for name in methods}

    sample_count = 0
    pbar = tqdm(test_loader, total=args.num_samples, desc="Faithfulness")

    for images, labels in pbar:
        if sample_count >= args.num_samples:
            break

        image = images[0:1].to(device)

        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(dim=1).item()

        for name, method in methods.items():
            try:
                saliency = method.generate(image, pred_class)

                del_auc, del_scores = evaluator.compute_deletion(
                    image, saliency, pred_class)
                ins_auc, ins_scores = evaluator.compute_insertion(
                    image, saliency, pred_class)

                results[name]['del_auc'].append(del_auc)
                results[name]['ins_auc'].append(ins_auc)

                all_curves[name]['deletion'].append(del_scores)
                all_curves[name]['insertion'].append(ins_scores)

            except Exception as e:
                print(f"Error with {name}: {e}")

        sample_count += 1

        if sample_count % 10 == 0:
            j_del = results['Jacobian (Ours)']['del_auc']
            j_ins = results['Jacobian (Ours)']['ins_auc']
            j_score = (np.mean(j_ins) - np.mean(j_del)) if j_del else 0
            pbar.set_postfix_str(f"J_score={j_score:.4f}")

    # Print faithfulness results
    print("\n" + "=" * 70)
    print("FAITHFULNESS RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Deletion AUC':<18} {'Insertion AUC':<18} {'Score':<10}")
    print("-" * 70)

    summary = {}
    for name in methods:
        if results[name]['del_auc']:
            del_mean = np.mean(results[name]['del_auc'])
            del_std = np.std(results[name]['del_auc'])
            ins_mean = np.mean(results[name]['ins_auc'])
            ins_std = np.std(results[name]['ins_auc'])
            score = ins_mean - del_mean

            summary[name] = {
                'deletion': del_mean, 'deletion_std': del_std,
                'insertion': ins_mean, 'insertion_std': ins_std,
                'score': score
            }

            print(f"{name:<20} {del_mean:.4f} +/- {del_std:.4f}   "
                  f"{ins_mean:.4f} +/- {ins_std:.4f}   {score:.4f}")

    print("-" * 70)

    # Rankings
    ranked = sorted(summary.items(), key=lambda x: x[1]['score'], reverse=True)
    print("\nRANKING:")
    for i, (name, s) in enumerate(ranked):
        marker = "1st" if i == 0 else "2nd" if i == 1 else "3rd" if i == 2 else f"{i+1}th"
        print(f"  {marker}. {name}: {s['score']:.4f}")

    # Improvement over baselines
    j_score = summary.get('Jacobian (Ours)', {}).get('score', 0)
    g_score = summary.get('Gramian (Ours)', {}).get('score', 0)
    our_best = max(j_score, g_score)
    our_best_name = 'Jacobian' if j_score >= g_score else 'Gramian'

    gc_score = summary.get('Grad-CAM', {}).get('score', 0)
    ig_score = summary.get('InputGrad', {}).get('score', 0)
    r_score = summary.get('Random', {}).get('score', 0)

    print(f"\nKEY COMPARISONS:")
    if gc_score != 0:
        print(f"  {our_best_name} vs Grad-CAM:   "
              f"{((our_best - gc_score) / abs(gc_score) * 100):+.1f}%")
    if ig_score != 0:
        print(f"  {our_best_name} vs InputGrad:  "
              f"{((our_best - ig_score) / abs(ig_score) * 100):+.1f}%")
    if r_score != 0:
        print(f"  {our_best_name} vs Random:     "
              f"{((our_best - r_score) / abs(r_score + 1e-8) * 100):+.1f}%")

    # ========================================
    # PART 2: Per-Class Controllability Analysis
    # ========================================
    _, _, test_loader2 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root, npz_path=args.npz_path,
    )

    pc_ctrl_summary, pc_ctrl_data = run_per_class_analysis(
        model, test_loader2, device, class_names, args.num_samples
    )

    # ========================================
    # PART 3: Per-Class Faithfulness
    # ========================================
    _, _, test_loader3 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root, npz_path=args.npz_path,
    )

    pc_faith_summary = run_per_class_faithfulness(
        model, test_loader3, device, methods,
        evaluator, class_names, args.num_samples
    )

    # ========================================
    # PART 4: Layer-wise Controllability
    # ========================================
    _, _, test_loader4 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root, npz_path=args.npz_path,
    )

    layerwise_stats = collect_layerwise_data(
        model, test_loader4, device,
        num_samples=min(args.num_samples, 20)
    )

    # ========================================
    # PART 5: Qualitative Examples
    # ========================================
    _, _, test_loader5 = get_dataloader(
        dataset_type=dataset_type, batch_size=1,
        image_size=config.image_size, num_workers=0,
        data_root=args.data_root, npz_path=args.npz_path,
    )

    qualitative_examples = collect_qualitative_examples(
        model, methods, test_loader5, device, class_names,
        num_examples=min(6, args.num_samples)
    )

    # ========================================
    # ALL PLOTS
    # ========================================
    print(f"\n{'='*70}")
    print("GENERATING PUBLICATION FIGURES")
    print(f"{'='*70}")

    plot_faithfulness_curves(all_curves, output_dir)           # Fig 1
    plot_summary_bars(summary, output_dir)                     # Fig 2
    plot_del_ins_scatter(results, output_dir)                  # Fig 3
    plot_per_class_heatmap(pc_faith_summary, output_dir)       # Fig 4
    plot_per_class_grouped_bars(pc_faith_summary, output_dir)  # Fig 5
    plot_sparsity_entropy(pc_ctrl_data, class_names, output_dir)  # Fig 6
    plot_jg_correlation(pc_ctrl_summary, output_dir)           # Fig 7
    plot_dashboard(summary, all_curves,
                   pc_faith_summary, pc_ctrl_summary, output_dir)  # Fig 8
    plot_layerwise_controllability(layerwise_stats, output_dir)    # Fig 9
    plot_qualitative_comparison(qualitative_examples,
                                class_names, output_dir)           # Fig 10
    plot_saliency_maps(qualitative_examples,
                       class_names, output_dir)                    # Fig 11

    # ========================================
    # Save all results
    # ========================================
    torch.save({
        'faithfulness': summary,
        'per_class_ctrl': pc_ctrl_summary,
        'per_class_faith': pc_faith_summary,
        'layerwise_stats': layerwise_stats,
        'all_curves': {name: {
            'deletion': [s for s in all_curves[name]['deletion']],
            'insertion': [s for s in all_curves[name]['insertion']],
        } for name in all_curves},
        'raw_results': {name: {
            'del_auc': results[name]['del_auc'],
            'ins_auc': results[name]['ins_auc'],
        } for name in results},
        'args': vars(args),
    }, output_dir / 'final_results.pth')

    # Save text summary
    with open(output_dir / 'final_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("X-VMAMBA FINAL EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  Dataset:           {args.dataset}\n")
        f.write(f"  Layer weighting:   {args.layer_weighting}\n")
        f.write(f"  Class conditioned: {use_class_cond}\n")
        f.write(f"  Smooth sigma:      {args.smooth_sigma}\n")
        f.write(f"  Num samples:       {args.num_samples}\n\n")

        f.write("FAITHFULNESS (Insertion AUC - Deletion AUC, higher is better)\n")
        f.write("-" * 70 + "\n")
        for name, s in ranked:
            f.write(f"{name:<20} Score={s['score']:.4f}  "
                    f"Del={s['deletion']:.4f}  Ins={s['insertion']:.4f}\n")

        f.write(f"\nBest method: {ranked[0][0]} ({ranked[0][1]['score']:.4f})\n")

        f.write("\n\nPER-CLASS CONTROLLABILITY ANALYSIS\n")
        f.write("-" * 70 + "\n")
        for cname, stats in pc_ctrl_summary.items():
            f.write(f"{cname}: Acc={stats['accuracy']:.3f}, "
                    f"J-G Corr={stats['jg_correlation']:.3f}, "
                    f"N={stats['n_samples']}\n")

        if pc_faith_summary:
            f.write("\n\nPER-CLASS FAITHFULNESS SCORES\n")
            f.write("-" * 70 + "\n")
            for cname, mdata in pc_faith_summary.items():
                row = f"{cname:<16} "
                for mname, vals in mdata.items():
                    row += f" {mname}={vals['score']:.3f}"
                f.write(row + "\n")

    # Figure listing
    figure_files = sorted(output_dir.glob('fig*.png'))
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to {output_dir}/")
    for ff in figure_files:
        print(f"  - {ff.name}")
    print(f"  - final_summary.txt")
    print(f"  - final_results.pth")

    # Paper-ready summary
    print(f"\n{'='*70}")
    print("PAPER-READY SUMMARY")
    print(f"{'='*70}")
    print(f"""
Table X: Faithfulness Evaluation on {args.dataset} (n={args.num_samples})

Method              Deletion AUC      Insertion AUC      Score
------------------------------------------------------------------""")
    for name, s in ranked:
        print(f"{name:<20} {s['deletion']:.3f} +/- {s['deletion_std']:.3f}      "
              f"{s['insertion']:.3f} +/- {s['insertion_std']:.3f}      {s['score']:.3f}")


if __name__ == "__main__":
    main()