"""
Additional saliency baselines for the TNNLS revision.

Reviewers 2, 3, 4 asked for baselines beyond Grad-CAM. We implement:

  - Score-CAM (Wang et al., CVPRW 2020): gradient-free CAM variant.
  - Integrated Gradients (Sundararajan et al., ICML 2017): gradient
    averaged along a straight-line path from a baseline.
  - RISE (Petsiuk et al., BMVC 2018): perturbation-based occlusion via
    random binary masks.

Each baseline exposes a `generate(image, target_class)` method that returns
a saliency map of shape `[H, W]` matching the input image's spatial
dimensions — the same contract used by `GradCAMSaliency` and
`StructuralControllability` in `comprehensive_evaluation.py`.

Auto-locating the last feature-map layer makes the baselines work for
both VMamba (`stages[-1].blocks[-1]`) and Vim (`blocks[-1]`).
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _locate_last_block(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last feature-map-producing block of the classifier.

    For VMamba: `model.stages[-1].blocks[-1]` (4-stage hierarchical).
    For Vim:    `model.blocks[-1]`            (plain, single resolution).
    Raises ValueError if the architecture is unrecognised.
    """
    if hasattr(model, "stages") and len(model.stages) > 0:
        last_stage = model.stages[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            return last_stage.blocks[-1]
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        return model.blocks[-1]
    raise ValueError(
        "could not auto-locate the last feature-map block on this model; "
        "pass `last_block=` explicitly to the saliency class."
    )


def _ensure_4d(feat: torch.Tensor) -> torch.Tensor:
    """Best-effort canonicalisation to `[B, C, H, W]`.

    VMamba blocks output `[B, H, W, C]` (channels-last); Vim blocks output
    `[B, L, C]` with `L = H*W`. We reshape both to a 4-D conv-style tensor
    so the saliency code below can apply a single algorithm.
    """
    if feat.ndim == 4:
        # Heuristic: if last axis is the largest, assume channels-last.
        if feat.shape[-1] >= feat.shape[1]:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat
    if feat.ndim == 3:
        # `[B, L, C]` — infer (H, W) as integer square root of L.
        B, L, C = feat.shape
        h = int(round(L ** 0.5))
        if h * h != L:
            raise ValueError(
                f"feature shape {tuple(feat.shape)} has non-square length L={L}; "
                "cannot reshape to (H, W). Pass an explicit (H, W) if needed."
            )
        return feat.transpose(1, 2).reshape(B, C, h, h).contiguous()
    raise ValueError(f"unexpected feature tensor rank {feat.ndim}: {tuple(feat.shape)}")


def _normalise(map_: torch.Tensor) -> torch.Tensor:
    """Min-max normalise to [0, 1]; safe against constant maps."""
    mn, mx = map_.min(), map_.max()
    if (mx - mn).abs() < 1e-8:
        return torch.zeros_like(map_)
    return (map_ - mn) / (mx - mn + 1e-8)


# ============================================================================
# Score-CAM (Wang et al., CVPRW 2020)
# ============================================================================
class ScoreCAMSaliency:
    """Gradient-free CAM variant.

    Algorithm:
      1. Hook the last block; on forward pass, capture activations `A`.
      2. For each channel `c` (or each of the top-K channels by activation
         magnitude), build a 2-D mask by min-max normalising `A[:, c]`
         and upsampling to input resolution.
      3. Mask the input (`mask * image + (1 - mask) * baseline`) and
         forward through the model.
      4. Take the softmax score for `target_class` as the weight for that
         channel.
      5. Saliency = ReLU(sum over channels of weight_c * A_c), upsampled
         to input resolution.

    `top_k` bounds the per-image forward-pass cost. The original Score-CAM
    uses all channels; for the high-dim feature maps of our VMamba-Tiny
    (D up to 256) this becomes a several-hundred forward-pass loop per
    image. Setting `top_k=32` keeps the cost comparable to RISE and is a
    standard pragmatic choice in recent CAM literature.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        top_k: int = 32,
        baseline_value: float = 0.0,
        last_block: Optional[torch.nn.Module] = None,
    ):
        self.model = model
        self.device = device
        self.top_k = top_k
        self.baseline_value = baseline_value
        self.last_block = last_block or _locate_last_block(model)
        self._features: Optional[torch.Tensor] = None
        self.last_block.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inputs, output):
        # `output` is the block's output (B,H,W,C) or (B,L,C); we canonicalise
        # to (B,C,H,W) at use-time, not here, to avoid double-permute.
        self._features = output.detach()

    @torch.no_grad()
    def generate(self, image: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Args:
            image: `[1, C, H, W]` on `self.device`.
            target_class: index of the class whose score weights the channels.

        Returns:
            saliency map `[H, W]` on CPU, normalised to [0, 1].
        """
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Score-CAM expects a single image `[1, C, H, W]`")
        image = image.to(self.device)
        H, W = image.shape[2], image.shape[3]

        # 1) capture activations.
        self._features = None
        _ = self.model(image)
        if self._features is None:
            raise RuntimeError("hook did not fire — wrong layer or analysis mode?")
        feat = _ensure_4d(self._features)             # [1, C_feat, h, w]
        feat = feat[0]                                # [C_feat, h, w]
        C_feat, h, w = feat.shape

        # 2) channel selection by activation magnitude.
        if 0 < self.top_k < C_feat:
            mag = feat.flatten(1).abs().mean(dim=1)   # [C_feat]
            chan_idx = torch.topk(mag, self.top_k).indices
        else:
            chan_idx = torch.arange(C_feat, device=feat.device)

        weights = torch.zeros(len(chan_idx), device=feat.device)
        baseline = torch.full_like(image, self.baseline_value)

        # 3-4) for each selected channel, mask the image and forward.
        for i, c in enumerate(chan_idx.tolist()):
            mask_lo = feat[c]                          # [h, w]
            # normalise to [0, 1]
            mn, mx = mask_lo.min(), mask_lo.max()
            if (mx - mn).abs() < 1e-8:
                continue
            mask_lo = (mask_lo - mn) / (mx - mn + 1e-8)
            mask = F.interpolate(
                mask_lo.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            )                                          # [1, 1, H, W]
            masked = mask * image + (1.0 - mask) * baseline
            logits = self.model(masked)
            score = F.softmax(logits, dim=-1)[0, target_class]
            weights[i] = score

        # 5) build CAM.
        cam_lo = torch.zeros(h, w, device=feat.device)
        for i, c in enumerate(chan_idx.tolist()):
            cam_lo = cam_lo + weights[i] * feat[c]
        cam_lo = F.relu(cam_lo)
        cam = F.interpolate(
            cam_lo.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze()
        return _normalise(cam).cpu()


# ============================================================================
# Integrated Gradients (Sundararajan et al., ICML 2017)
# ============================================================================
class IntegratedGradientsSaliency:
    """Sums gradients along a straight-line path from baseline to input.

    Standard formula:
        IG(x)_i = (x_i - baseline_i) * mean_{α∈[0,1]} ∂F(baseline + α(x-baseline))_{class} / ∂x_i

    The returned saliency map is `mean over channels of |IG|`, then
    min-max normalised. The default baseline is the all-zeros image.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        steps: int = 20,
        baseline: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.device = device
        self.steps = steps
        self.baseline = baseline  # `[1, C, H, W]` or None for zeros.

    def generate(self, image: torch.Tensor, target_class: int) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("IG expects a single image `[1, C, H, W]`")
        image = image.to(self.device).detach()
        baseline = (
            self.baseline.to(self.device).detach()
            if self.baseline is not None
            else torch.zeros_like(image)
        )

        # Accumulate gradients at α = (k + 0.5) / steps  (midpoint rule).
        accum = torch.zeros_like(image)
        for k in range(self.steps):
            alpha = (k + 0.5) / self.steps
            x_interp = (baseline + alpha * (image - baseline)).clone().requires_grad_(True)
            self.model.zero_grad()
            logits = self.model(x_interp)
            score = logits[0, target_class]
            grad, = torch.autograd.grad(score, x_interp, retain_graph=False, create_graph=False)
            accum = accum + grad.detach()
        avg_grad = accum / self.steps
        ig = (image - baseline) * avg_grad        # [1, C, H, W]
        sal = ig.abs().mean(dim=1).squeeze(0)     # mean over channels -> [H, W]
        return _normalise(sal).cpu()


# ============================================================================
# RISE (Petsiuk et al., BMVC 2018)
# ============================================================================
class RISESaliency:
    """Random masked-input attribution.

    Algorithm:
      1. Sample N low-resolution binary masks `m_n ∈ {0, 1}^{s × s}` with
         Bernoulli(p) per cell.
      2. Upsample each to `[H, W]` with bilinear interpolation and apply a
         random sub-pixel shift (so mask boundaries don't align with pixels).
      3. Mask the image (`mask * image`) and forward through the model.
      4. Saliency(i, j) ≈ E[score | mask_{ij} = 1] - E[score].
         The standard estimator is `(1/N) Σ_n score_n * mask_n / p`.

    Sample count `num_masks` trades quality for compute; the original paper
    uses 4000–8000. We default to 500 which is enough for the stable
    medical-image inputs we evaluate but is the natural knob to crank up
    for the final paper.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_masks: int = 500,
        mask_grid: int = 7,
        mask_prob: float = 0.5,
        batch_size: int = 16,
    ):
        self.model = model
        self.device = device
        self.num_masks = num_masks
        self.mask_grid = mask_grid
        self.mask_prob = mask_prob
        self.batch_size = batch_size

    def _generate_masks(self, H: int, W: int) -> torch.Tensor:
        """Return `[N, 1, H, W]` masks on `self.device`."""
        N, s = self.num_masks, self.mask_grid
        cell_h, cell_w = H // s + 1, W // s + 1
        # Low-resolution binary masks.
        low = (torch.rand(N, s, s, device=self.device) < self.mask_prob).float()
        # Upsample to (s+1)*cell_h x (s+1)*cell_w then crop with a random offset.
        up_h, up_w = (s + 1) * cell_h, (s + 1) * cell_w
        up = F.interpolate(
            low.unsqueeze(1), size=(up_h, up_w),
            mode="bilinear", align_corners=False,
        )                                              # [N, 1, up_h, up_w]
        # Random crop offsets in [0, cell_h], [0, cell_w].
        offs_h = torch.randint(0, cell_h, (N,), device=self.device)
        offs_w = torch.randint(0, cell_w, (N,), device=self.device)
        masks = torch.empty(N, 1, H, W, device=self.device)
        for n in range(N):
            masks[n, 0] = up[n, 0, offs_h[n]:offs_h[n] + H, offs_w[n]:offs_w[n] + W]
        return masks

    @torch.no_grad()
    def generate(self, image: torch.Tensor, target_class: int) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("RISE expects a single image `[1, C, H, W]`")
        image = image.to(self.device)
        H, W = image.shape[2], image.shape[3]

        masks = self._generate_masks(H, W)             # [N, 1, H, W]
        sal = torch.zeros(H, W, device=self.device)
        for i in range(0, self.num_masks, self.batch_size):
            mb = masks[i:i + self.batch_size]
            # Mask the same image B times.
            masked = mb * image                         # broadcasts [B,1,H,W]*[1,C,H,W]
            logits = self.model(masked)
            scores = F.softmax(logits, dim=-1)[:, target_class]   # [B]
            # Weighted sum of masks.
            sal = sal + (scores.view(-1, 1, 1, 1) * mb).sum(dim=0).squeeze(0)
        sal = sal / (self.num_masks * self.mask_prob)
        return _normalise(sal).cpu()


# ============================================================================
# Convenience: dict of available methods keyed by display name.
# ============================================================================
def build_extra_baselines(
    model: torch.nn.Module,
    device: torch.device,
    *,
    score_cam_top_k: int = 32,
    ig_steps: int = 20,
    rise_num_masks: int = 500,
) -> dict:
    """Construct all three extra baseline objects in one call.

    The keys match the display names used by the existing eval pipeline
    so the new methods can be merged into the `methods` dicts in
    `comprehensive_evaluation.py`.
    """
    return {
        "Score-CAM": ScoreCAMSaliency(model, device, top_k=score_cam_top_k),
        "Integrated Gradients": IntegratedGradientsSaliency(model, device, steps=ig_steps),
        "RISE": RISESaliency(model, device, num_masks=rise_num_masks),
    }


# Self-test (CPU / GPU, depending on availability) --------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from models.vmamba_classifier import create_vmamba_medical

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = create_vmamba_medical(num_classes=8, image_size=224, in_channels=3).to(device).eval()
    image = torch.randn(1, 3, 224, 224, device=device)

    print("Testing Score-CAM (top_k=8)...")
    sc = ScoreCAMSaliency(model, device, top_k=8)
    sal = sc.generate(image, target_class=0)
    print(f"  shape={tuple(sal.shape)}, range=({sal.min():.3f}, {sal.max():.3f})")

    print("Testing Integrated Gradients (steps=5)...")
    ig = IntegratedGradientsSaliency(model, device, steps=5)
    sal = ig.generate(image, target_class=0)
    print(f"  shape={tuple(sal.shape)}, range=({sal.min():.3f}, {sal.max():.3f})")

    print("Testing RISE (N=20)...")
    rise = RISESaliency(model, device, num_masks=20, batch_size=4)
    sal = rise.generate(image, target_class=0)
    print(f"  shape={tuple(sal.shape)}, range=({sal.min():.3f}, {sal.max():.3f})")

    print("All three extra baselines pass the smoke test.")
