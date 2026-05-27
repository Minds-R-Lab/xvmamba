# Controllability Analysis for Vision State Space Models

A structural interpretability framework for Vision Mamba based on control theory.

[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This repository provides the official implementation of **"Controllability Analysis for Vision State Space Models: A Structural Interpretability Framework"**.

We introduce controllability analysis — a control-theoretic framework that quantifies the structural influence of each input position on the internal state dynamics of vision state space models. Unlike gradient-based attribution methods that produce class-specific explanations, controllability indices measure intrinsic model properties that are invariant to the target class.

### Key Features

- **Two Controllability Indices**: Jacobian-based (sensitivity propagation) and Gramian-based (closed-form local frozen-LTI proxy)
- **Class-Agnostic Explanations**: Perfect cross-class consistency (correlation = 1.0)
- **Efficient Computation**: O(LN) complexity via backward recursion
- **Cross-Architecture**: Supports VMamba (cross-scan) and Vim (bidirectional single-sequence scan)
- **Seven Datasets**: Four MedMNIST (BloodMNIST, OCTMNIST, DermaMNIST, PneumoniaMNIST) plus three non-medical (CIFAR-100, FashionMNIST, EuroSAT)
- **Five Saliency Baselines**: Grad-CAM, Integrated Gradients, RISE, Score-CAM, and a Random control
- **Two Faithfulness Metrics**: Insertion/deletion AUC (audit-fixed protocol) and Internal-Attention IoU
- **Statistical Reliability**: Three-seed multi-seed protocol with bootstrap 95% CIs
- **Reproducible Pipeline**: Training, evaluation, and ablation scripts under `scripts/`

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/xvmamba-controllability.git
cd xvmamba-controllability

# Create conda environment
conda create -n xvmamba python=3.10
conda activate xvmamba

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Optional: Install mamba-ssm for fast CUDA kernels
pip install mamba-ssm>=1.2.0
pip install causal-conv1d>=1.2.0
```

---

## Project Structure

```
xvmamba/
├── models/
│   ├── __init__.py
│   ├── selective_ssm.py       # Core SSM implementation
│   ├── ss2d.py                # 2D scanning module
│   └── vmamba_classifier.py   # Complete classifier
├── controllability/
│   ├── __init__.py
│   └── analyzer.py            # Jacobian & Gramian methods
├── data/
│   ├── __init__.py
│   └── datasets.py            # Dataset loaders
├── evaluation/
│   ├── __init__.py
│   ├── comprehensive_evaluation.py  # All 4 tests
│   └── run_evaluation.py      # Faithfulness evaluation
├── scripts/
│   └── train.py               # Training script
├── configs/
│   └── default.yaml           # Default configuration
├── checkpoints/               # Saved model weights
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Train a Model

```bash
# Train on BloodMNIST
python scripts/train.py --dataset bloodmnist --epochs 50

# Train on other datasets
python scripts/train.py --dataset dermamnist
python scripts/train.py --dataset octmnist
python scripts/train.py --dataset pneumoniamnist
```

### 2. Run Comprehensive Evaluation

```bash
python evaluation/comprehensive_evaluation.py \
    --checkpoint ./checkpoints/bloodmnist/best_model.pth \
    --dataset bloodmnist \
    --num_samples 50 \
    --output_dir ./results/bloodmnist

    python evaluation/comprehensive_evaluation.py \
    --checkpoint ./checkpoints/dermamnist/best_model.pth \
    --dataset dermamnist \
    --num_samples 50 \
    --output_dir ./results/dermamnist
```

### 3. Generate Saliency Maps

```python
from models import VMambaClassifier, VMambaConfig
from controllability import ControllabilityAnalyzer
import torch

# Load model
config = VMambaConfig(num_classes=8)
model = VMambaClassifier(config)
model.load_state_dict(torch.load('checkpoints/bloodmnist/best_model.pth')['model_state_dict'])
model.eval()

# Create analyzer
analyzer = ControllabilityAnalyzer(method='jacobian')

# Enable analysis mode
model.enable_analysis_mode()

# Forward pass
image = torch.randn(1, 3, 224, 224)
logits, analysis = model(image, return_analysis=True)

# Compute controllability
results = analyzer.analyze(analysis)
influence_map = results.aggregated_map

# Visualize
import matplotlib.pyplot as plt
plt.imshow(influence_map.cpu().numpy(), cmap='hot')
plt.savefig('controllability_map.png')
```

---

## Datasets

We evaluate on seven image classification benchmarks: four medical (MedMNIST) and three non-medical, covering microscopy, OCT, dermatoscopy, X-ray, natural images, apparel imagery, and satellite remote sensing.

| Dataset | Modality | Classes | Source |
|---------|----------|---------|--------|
| BloodMNIST | Microscopy | 8 | MedMNIST v2 |
| OCTMNIST | OCT | 4 | MedMNIST v2 |
| DermaMNIST | Dermatoscopy | 7 | MedMNIST v2 |
| PneumoniaMNIST | Chest X-ray | 2 | MedMNIST v2 |
| CIFAR-100 | Natural images | 100 | torchvision |
| FashionMNIST | Apparel imagery | 10 | torchvision |
| EuroSAT | Sentinel-2 satellite | 10 | EuroSAT (RGB) |

MedMNIST and torchvision datasets are downloaded automatically on first use. EuroSAT can be obtained from https://github.com/phelber/EuroSAT.

---

## Evaluation Methodology

### 1. Cross-Class Consistency

Tests whether saliency maps are class-agnostic (structural) or class-dependent.

```bash
python evaluation/comprehensive_evaluation.py --checkpoint ./checkpoints/bloodmnist/best_model.pth --dataset bloodmnist
```

**Expected Results:**
- Controllability: correlation ≈ 1.0 (class-agnostic)
- Grad-CAM: correlation ≈ 0.1-0.3 (class-dependent)

### 2. Perturbation Invariance

Tests whether identified regions actually influence predictions.

### 3. Faithfulness (Insertion / Deletion AUC)

Audit-fixed insertion/deletion AUC with a per-channel baseline, tie-jittered pixel ordering, top-K predicted-class targeting, and bootstrap 95% CIs over 50 test images. Reported as insertion AUC minus deletion AUC across all seven datasets.

```bash
bash scripts/run_revised_eval.sh
```

### 4. Internal-Attention IoU

Complementary faithfulness metric measuring spatial agreement between each method's top-K mask and the model's own L2-magnitude attention pattern (top-25% region) at the last block, before the classifier head.

```bash
bash scripts/run_attention_iou_pilot.sh
```

### 5. Architecture Analysis

Per-stage controllability and entropy patterns; per-direction, per-layer scan decomposition.

### 6. Ablations

- **Aggregation ablation** (block/stage/direction weighting): `bash scripts/run_aggregation_ablation.sh`
- **Per-direction consistency** (forward/backward, horizontal/vertical): part of the aggregation script
- **Misclassification stratified analysis**: `bash scripts/run_misclassification_analysis.sh`
- **Extra saliency baselines** (Score-CAM, Integrated Gradients, RISE): `bash scripts/run_extra_baselines.sh`

### 7. Multi-seed Reliability

Three-seed multi-seed protocol (seeds 42, 137, 2024) for every dataset row.

```bash
bash scripts/run_multiseed.sh
bash scripts/run_table_v_full_multiseed.sh
```

---

## Controllability Methods

### Jacobian Controllability Index

Measures input-output sensitivity through the full sequence:

$$\mathcal{J}_k = \sum_{t=k}^{L} \left| \frac{\partial y_t}{\partial x_k} \right|$$

Computed efficiently via backward recursion in O(LN) time.

### Gramian Controllability Index

Closed-form solution exploiting diagonal SSM structure:

$$\mathcal{G}_k = \sum_{n=1}^{N} \frac{c_{k,n}^2 \, \bar{b}_{k,n}^2}{1 - \bar{a}_{k,n}^2}$$

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Patch Size | 4 |
| Stage Depths | [2, 2, 4, 2] |
| Stage Dimensions | [32, 64, 128, 256] |
| State Dimension | 16 |
| Scan Directions | 4 (cross-scan) |

---

## Results

### Cross-Class Consistency

| Dataset | Jacobian | Gramian | Grad-CAM |
|---------|----------|---------|----------|
| BloodMNIST | **1.000** | **1.000** | 0.227 |
| OCTMNIST | **1.000** | **1.000** | 0.091 |
| DermaMNIST | **1.000** | **1.000** | 0.086 |
| PneumoniaMNIST | **1.000** | **1.000** | -0.463 |

### Faithfulness (Insertion - Deletion)

| Dataset | Jacobian | Gramian | Grad-CAM |
|---------|----------|---------|----------|
| BloodMNIST | **0.679** | **0.671** | 0.568 |
| OCTMNIST | 0.116 | 0.106 | **0.232** |
| DermaMNIST | 0.169 | 0.156 | **0.233** |
| PneumoniaMNIST | 0.020 | 0.016 | **0.188** |

---

## Reproducing Paper Results

```bash
# Step 1: Train models on all datasets
for dataset in bloodmnist octmnist dermamnist pneumoniamnist; do
    python scripts/train.py --dataset $dataset --epochs 50
done

# Step 2: Run comprehensive evaluation
for dataset in bloodmnist octmnist dermamnist pneumoniamnist; do
    python evaluation/comprehensive_evaluation.py \
        --checkpoint ./checkpoints/$dataset/best_model.pth \
        --dataset $dataset \
        --num_samples 50 \
        --output_dir ./results/$dataset
done
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{author2024controllability,
  title={Controllability Analysis for Vision State Space Models: A Structural Interpretability Framework},
}
```

---

## Related Work

- [VMamba](https://github.com/MzeroMiko/VMamba) - Visual State Space Model
- [Mamba](https://github.com/state-spaces/mamba) - Linear-Time Sequence Modeling
- [MedMNIST](https://medmnist.com/) - Medical Image Datasets

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- VMamba authors for the visual state space model implementation
- MedMNIST team for the medical imaging benchmarks
- Mamba authors for the efficient SSM kernels
