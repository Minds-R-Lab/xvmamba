"""
Data Loaders for Image Classification Datasets

This module provides data loaders for the datasets used in the X-VMamba paper:

**Medical imaging (MedMNIST family):**
 1. DermaMNIST       – Skin lesion dermatoscopy (7 classes, RGB)
 2. BloodMNIST       – Blood cell microscopy (8 classes, RGB)
 3. PathMNIST        – Colon pathology (9 classes, RGB)
 4. RetinaMNIST      – Retinal fundus / diabetic retinopathy (5 ordinal, RGB)
 5. BreastMNIST      – Breast ultrasound (2 classes, grayscale)
 6. PneumoniaMNIST   – Chest X-ray pneumonia (2 classes, grayscale)
 7. OCTMNIST         – Retinal OCT (4 classes, grayscale)
 8. OrganSMNIST      – Abdominal CT, sagittal (11 classes, grayscale)
 9. OrganAMNIST      – Abdominal CT, axial (11 classes, grayscale)
10. OrganCMNIST      – Abdominal CT, coronal (11 classes, grayscale)
11. TissueMNIST      – Kidney cortex microscopy (8 classes, grayscale)

**Mammography (requires manual download):**
12. CMMD             – Chinese Mammography Database (2 classes, grayscale)

**Natural images:**
13. CIFAR-10         – Natural image classification (10 classes, RGB)

By default the high-resolution (224×224) MedMNIST variants are used so that
no lossy upsampling from 28×28 is required.  The ``size`` parameter can be
changed to 28, 64, or 128 if the smaller versions are preferred.

Usage:
    from data import get_dataloader, DatasetType

    # MedMNIST dataset (downloads 224x224 native by default)
    train_loader, val_loader, test_loader = get_dataloader(
        dataset_type=DatasetType.DERMAMNIST,
        batch_size=32,
        image_size=224,
    )

    # CIFAR-10
    train_loader, val_loader, test_loader = get_dataloader(
        dataset_type=DatasetType.CIFAR10,
        batch_size=32,
        image_size=224,
    )

    # Use pre-downloaded NPZ file
    train_loader, val_loader, test_loader = get_dataloader(
        dataset_type=DatasetType.DERMAMNIST,
        batch_size=32,
        npz_path="data/dermamnist_extended_224.npz",
    )
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Callable
from enum import Enum
import numpy as np

try:
    import medmnist
    from medmnist import INFO
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("Warning: medmnist not installed. Install with: pip install medmnist")

try:
    from torchvision import datasets as tv_datasets
    TORCHVISION_DATASETS_AVAILABLE = True
except ImportError:
    TORCHVISION_DATASETS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ============================================================================
# Dataset Registry
# ============================================================================

class DatasetType(Enum):
    """Supported dataset types."""
    # --- MedMNIST (medical) ---
    DERMAMNIST = "dermamnist"
    BLOODMNIST = "bloodmnist"
    PATHMNIST = "pathmnist"
    RETINAMNIST = "retinamnist"
    BREASTMNIST = "breastmnist"
    PNEUMONIAMNIST = "pneumoniamnist"
    OCTMNIST = "octmnist"
    ORGANSMNIST = "organsmnist"
    ORGANAMNIST = "organamnist"
    ORGANCMNIST = "organcmnist"
    TISSUEMNIST = "tissuemnist"
    # --- Mammography ---
    CMMD = "cmmd"
    # --- Natural images ---
    CIFAR10 = "cifar10"
    # --- Generic ---
    CUSTOM = "custom"


# Which DatasetType values are served by the MedMNIST loader.
_MEDMNIST_TYPES: Dict[DatasetType, str] = {
    DatasetType.DERMAMNIST:     "dermamnist",
    DatasetType.BLOODMNIST:     "bloodmnist",
    DatasetType.PATHMNIST:      "pathmnist",
    DatasetType.RETINAMNIST:    "retinamnist",
    DatasetType.BREASTMNIST:    "breastmnist",
    DatasetType.PNEUMONIAMNIST: "pneumoniamnist",
    DatasetType.OCTMNIST:       "octmnist",
    DatasetType.ORGANSMNIST:    "organsmnist",
    DatasetType.ORGANAMNIST:    "organamnist",
    DatasetType.ORGANCMNIST:    "organcmnist",
    DatasetType.TISSUEMNIST:    "tissuemnist",
}


# ============================================================================
# MedMNIST Size Configuration
# ============================================================================

# Native resolutions available in MedMNIST v3+.
MEDMNIST_VALID_SIZES = {28, 64, 128, 224}

# Default download size – use the largest native resolution so there is no
# lossy upsampling when the model expects 224×224 inputs.
MEDMNIST_DEFAULT_SIZE = 224


def _resolve_medmnist_size(requested_image_size: int) -> int:
    """Pick the best MedMNIST native size for a given ``image_size``.

    Rules:
        * If ``requested_image_size`` exactly matches a valid size, use it.
        * Otherwise pick the smallest valid size that is >= requested, so a
          single down-sample (high quality) is used instead of an up-sample.
        * If ``requested_image_size`` > 224, download 224 and let the
          transform up-sample (rare edge case).
    """
    if requested_image_size in MEDMNIST_VALID_SIZES:
        return requested_image_size

    # Pick smallest valid size >= requested
    larger = sorted(s for s in MEDMNIST_VALID_SIZES if s >= requested_image_size)
    if larger:
        return larger[0]

    # requested > 224 – download the largest available
    return max(MEDMNIST_VALID_SIZES)


# ============================================================================
# Dataset Information
# ============================================================================

DATASET_INFO = {
    # ----- MedMNIST RGB datasets -----
    DatasetType.DERMAMNIST: {
        "name": "DermaMNIST",
        "description": "Skin lesion classification (dermatoscopy images)",
        "num_classes": 7,
        "in_channels": 3,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "actinic keratoses",
            "basal cell carcinoma",
            "benign keratosis",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions",
        ],
    },
    DatasetType.BLOODMNIST: {
        "name": "BloodMNIST",
        "description": "Blood cell classification (microscopy images)",
        "num_classes": 8,
        "in_channels": 3,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature granulocyte",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet",
        ],
    },
    DatasetType.PATHMNIST: {
        "name": "PathMNIST",
        "description": "Colon pathology tissue classification",
        "num_classes": 9,
        "in_channels": 3,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal colon mucosa",
            "cancer-associated stroma",
            "colorectal adenocarcinoma epithelium",
        ],
    },
    DatasetType.RETINAMNIST: {
        "name": "RetinaMNIST",
        "description": "Retinal fundus – diabetic retinopathy grading",
        "num_classes": 5,
        "in_channels": 3,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "ordinal-regression",
        "classes": [
            "no DR",
            "mild NPDR",
            "moderate NPDR",
            "severe NPDR",
            "proliferative DR",
        ],
    },
    # ----- MedMNIST grayscale datasets -----
    DatasetType.BREASTMNIST: {
        "name": "BreastMNIST",
        "description": "Breast ultrasound classification",
        "num_classes": 2,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "binary",
        "classes": ["benign", "malignant"],
    },
    DatasetType.PNEUMONIAMNIST: {
        "name": "PneumoniaMNIST",
        "description": "Chest X-ray pneumonia detection",
        "num_classes": 2,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "binary",
        "classes": ["normal", "pneumonia"],
    },
    DatasetType.OCTMNIST: {
        "name": "OCTMNIST",
        "description": "Retinal OCT classification",
        "num_classes": 4,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": ["CNV", "DME", "drusen", "normal"],
    },
    DatasetType.ORGANSMNIST: {
        "name": "OrganSMNIST",
        "description": "Abdominal CT organ classification (sagittal plane)",
        "num_classes": 11,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "bladder", "femur-left", "femur-right", "heart",
            "kidney-left", "kidney-right", "liver", "lung-left",
            "lung-right", "spleen", "pancreas",
        ],
    },
    DatasetType.ORGANAMNIST: {
        "name": "OrganAMNIST",
        "description": "Abdominal CT organ classification (axial plane)",
        "num_classes": 11,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "bladder", "femur-left", "femur-right", "heart",
            "kidney-left", "kidney-right", "liver", "lung-left",
            "lung-right", "spleen", "pancreas",
        ],
    },
    DatasetType.ORGANCMNIST: {
        "name": "OrganCMNIST",
        "description": "Abdominal CT organ classification (coronal plane)",
        "num_classes": 11,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "bladder", "femur-left", "femur-right", "heart",
            "kidney-left", "kidney-right", "liver", "lung-left",
            "lung-right", "spleen", "pancreas",
        ],
    },
    DatasetType.TISSUEMNIST: {
        "name": "TissueMNIST",
        "description": "Human kidney cortex cell classification (microscopy)",
        "num_classes": 8,
        "in_channels": 1,
        "original_size": 28,
        "available_sizes": [28, 64, 128, 224],
        "default_download_size": MEDMNIST_DEFAULT_SIZE,
        "task": "multi-class",
        "classes": [
            "collecting duct, intercalated",
            "collecting duct, principal",
            "connecting tubule",
            "distal convoluted tubule",
            "glomerular endothelial",
            "interstitial",
            "proximal tubule, convoluted",
            "proximal tubule, straight",
        ],
    },
    # ----- Mammography -----
    DatasetType.CMMD: {
        "name": "CMMD (Chinese Mammography Database)",
        "description": "Mammography classification",
        "num_classes": 2,
        "in_channels": 1,
        "original_size": "variable",
        "task": "binary",
        "classes": ["benign", "malignant"],
        "note": "Requires manual download from TCIA",
    },
    # ----- Natural images -----
    DatasetType.CIFAR10: {
        "name": "CIFAR-10",
        "description": "Natural image classification (10 classes)",
        "num_classes": 10,
        "in_channels": 3,
        "original_size": 32,
        "task": "multi-class",
        "classes": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
    },
}


def get_dataset_info(dataset_type: DatasetType) -> Dict[str, Any]:
    """Get information about a dataset."""
    return DATASET_INFO.get(dataset_type, {})


# ============================================================================
# Transforms
# ============================================================================

def get_train_transforms(
    image_size: int = 224,
    in_channels: int = 3,
    already_resized: bool = False,
) -> transforms.Compose:
    """
    Get training transforms with augmentation.

    Args:
        image_size: Target image size
        in_channels: Number of input channels
        already_resized: If True, skip resize step
    """
    transform_list = []

    if not already_resized:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])

    if in_channels == 3:
        transform_list.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05)
        )

    transform_list.append(transforms.ToTensor())

    # Normalization
    if in_channels == 3:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    else:
        transform_list.append(
            transforms.Normalize(mean=[0.5], std=[0.5])
        )

    return transforms.Compose(transform_list)


def get_val_transforms(
    image_size: int = 224,
    in_channels: int = 3,
    already_resized: bool = False,
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size
        in_channels: Number of input channels
        already_resized: If True, skip resize step
    """
    transform_list = []

    if not already_resized:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.append(transforms.ToTensor())

    if in_channels == 3:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    else:
        transform_list.append(
            transforms.Normalize(mean=[0.5], std=[0.5])
        )

    return transforms.Compose(transform_list)


# ============================================================================
# NPZ Dataset (for pre-downloaded data)
# ============================================================================

class NPZDataset(Dataset):
    """
    Dataset loader for NPZ files containing pre-processed images.

    Expected NPZ structure (common formats):

    Format 1 (separate arrays):
        - train_images: [N, H, W, C] or [N, H, W]
        - train_labels: [N] or [N, 1]
        - val_images / test_images
        - val_labels / test_labels

    Format 2 (MedMNIST style):
        - images: [N, H, W, C]
        - labels: [N] or [N, 1]

    Format 3 (with split info):
        - images: [N, H, W, C]
        - labels: [N]
        - train_idx, val_idx, test_idx

    Args:
        npz_path: Path to the NPZ file
        split: 'train', 'val', or 'test'
        transform: Optional transforms
        val_ratio: Ratio for validation split (if not pre-split)
        test_ratio: Ratio for test split (if not pre-split)
    """

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.npz_path = Path(npz_path)
        self.split = split
        self.transform = transform

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Load NPZ file
        print(f"Loading {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)

        # Print available keys for debugging
        print(f"  Available keys: {list(data.keys())}")

        # Try to load images and labels based on different formats
        images, labels = self._load_data(data, split, val_ratio, test_ratio, seed)

        self.images = images
        self.labels = labels

        print(f"  {split} set: {len(self.images)} samples")
        print(f"  Image shape: {self.images[0].shape if len(self.images) > 0 else 'N/A'}")

        # Determine number of channels
        if len(self.images) > 0:
            if self.images[0].ndim == 2:
                self.in_channels = 1
            else:
                self.in_channels = (self.images[0].shape[-1]
                                    if self.images[0].shape[-1] in [1, 3, 4]
                                    else 1)
        else:
            self.in_channels = 3

    def _load_data(
        self,
        data: np.lib.npyio.NpzFile,
        split: str,
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load images and labels from NPZ file."""

        keys = list(data.keys())

        # Format 1: Separate train/val/test arrays
        if f'{split}_images' in keys:
            images = data[f'{split}_images']
            labels = data[f'{split}_labels']
            return images, labels.squeeze()

        # Also check for 'train_image' (singular) format
        if f'{split}_image' in keys:
            images = data[f'{split}_image']
            labels = data[f'{split}_label']
            return images, labels.squeeze()

        # Format 2: Single arrays with indices
        if 'images' in keys and f'{split}_idx' in keys:
            idx = data[f'{split}_idx']
            images = data['images'][idx]
            labels = data['labels'][idx]
            return images, labels.squeeze()

        # Format 3: Single arrays, need to split ourselves
        if 'images' in keys and 'labels' in keys:
            all_images = data['images']
            all_labels = data['labels'].squeeze()

            # Create splits
            n = len(all_images)
            np.random.seed(seed)
            indices = np.random.permutation(n)

            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)

            if split == 'test':
                idx = indices[:n_test]
            elif split == 'val':
                idx = indices[n_test:n_test + n_val]
            else:  # train
                idx = indices[n_test + n_val:]

            return all_images[idx], all_labels[idx]

        # Format 4: Try 'x' and 'y' keys
        if 'x' in keys and 'y' in keys:
            all_images = data['x']
            all_labels = data['y'].squeeze()

            n = len(all_images)
            np.random.seed(seed)
            indices = np.random.permutation(n)

            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)

            if split == 'test':
                idx = indices[:n_test]
            elif split == 'val':
                idx = indices[n_test:n_test + n_val]
            else:
                idx = indices[n_test + n_val:]

            return all_images[idx], all_labels[idx]

        # Format 5: Check for 'data' key
        if 'data' in keys:
            all_images = data['data']
            all_labels = data.get('labels',
                                  data.get('targets',
                                           data.get('y', None)))

            if all_labels is None:
                raise ValueError("Cannot find labels in NPZ file")

            all_labels = all_labels.squeeze()

            n = len(all_images)
            np.random.seed(seed)
            indices = np.random.permutation(n)

            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)

            if split == 'test':
                idx = indices[:n_test]
            elif split == 'val':
                idx = indices[n_test:n_test + n_val]
            else:
                idx = indices[n_test + n_val:]

            return all_images[idx], all_labels[idx]

        raise ValueError(
            f"Cannot determine data format. Available keys: {keys}\n"
            "Expected one of:\n"
            "  - train_images, train_labels, val_images, val_labels, "
            "test_images, test_labels\n"
            "  - images, labels (with optional train_idx, val_idx, test_idx)\n"
            "  - x, y\n"
            "  - data, labels"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image for transforms
        if image.ndim == 2:
            # Grayscale
            image = Image.fromarray(image.astype(np.uint8), mode='L')
        elif image.shape[-1] == 1:
            # Grayscale with channel dim
            image = Image.fromarray(image.squeeze().astype(np.uint8), mode='L')
        elif image.shape[-1] == 3:
            # RGB
            image = Image.fromarray(image.astype(np.uint8), mode='RGB')
        elif image.shape[-1] == 4:
            # RGBA -> RGB
            image = Image.fromarray(image.astype(np.uint8),
                                    mode='RGBA').convert('RGB')
        else:
            # Assume channels first, convert to channels last
            if image.shape[0] in [1, 3, 4]:
                image = np.transpose(image, (1, 2, 0))
                if image.shape[-1] == 1:
                    image = Image.fromarray(image.squeeze().astype(np.uint8),
                                            mode='L')
                else:
                    image = Image.fromarray(image.astype(np.uint8), mode='RGB')
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_npz_loaders(
    npz_path: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    in_channels: int = 3,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders from an NPZ file.

    Args:
        npz_path: Path to the NPZ file
        batch_size: Batch size
        image_size: Target image size (set to None if already resized)
        num_workers: Number of data loading workers
        in_channels: Number of input channels

    Returns:
        train_loader, val_loader, test_loader
    """
    # Check if images are already the right size
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    # Find sample image to check size
    sample = None
    for key in ['train_images', 'train_image', 'images', 'x', 'data']:
        if key in keys:
            sample = data[key][0]
            break

    already_resized = False
    if sample is not None:
        h, w = sample.shape[0], sample.shape[1]
        if h == image_size and w == image_size:
            already_resized = True
            print(f"Images already resized to {image_size}x{image_size}")

    # Create datasets
    train_dataset = NPZDataset(
        npz_path=npz_path,
        split="train",
        transform=get_train_transforms(image_size, in_channels, already_resized),
    )

    val_dataset = NPZDataset(
        npz_path=npz_path,
        split="val",
        transform=get_val_transforms(image_size, in_channels, already_resized),
    )

    test_dataset = NPZDataset(
        npz_path=npz_path,
        split="test",
        transform=get_val_transforms(image_size, in_channels, already_resized),
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# MedMNIST Datasets  (now defaults to 224×224 native resolution)
# ============================================================================

class MedMNISTWrapper(Dataset):
    """
    Wrapper for MedMNIST datasets with custom transforms.

    By default downloads the **224×224** variant of the dataset so that no
    lossy upsampling from 28×28 is needed.  Pass ``size=28`` to revert to the
    original low-resolution version.

    Handles both RGB and grayscale datasets transparently.  Grayscale images
    are converted to 3-channel RGB (by repeating the single channel) when
    ``force_rgb=True``, which is useful when feeding into models that expect
    3-channel input.

    Args:
        dataset_name: MedMNIST key, e.g. 'dermamnist', 'organsmnist'
        split:        'train', 'val', or 'test'
        transform:    Optional torchvision transform
        size:         Native MedMNIST download resolution.
                      Valid values: 28, 64, 128, 224  (default **224**).
        download:     Whether to download if not present
        data_root:    Root directory for data
        force_rgb:    Convert grayscale images to 3-channel RGB
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        size: int = MEDMNIST_DEFAULT_SIZE,
        download: bool = True,
        data_root: str = "./data",
        force_rgb: bool = False,
    ):
        if not MEDMNIST_AVAILABLE:
            raise ImportError("medmnist not installed. Run: pip install medmnist")

        self.transform = transform
        self.size = size
        self.force_rgb = force_rgb

        # Get the dataset class
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])

        # MedMNIST v3+ accepts a ``size`` keyword.  For older versions that
        # lack it we fall back to the default (28) with a warning.
        try:
            self.dataset = DataClass(
                split=split,
                transform=None,  # We handle transforms ourselves
                size=size,
                download=download,
                root=data_root,
            )
            print(f"  Loaded {dataset_name} ({split}) – "
                  f"native {size}x{size}  [{len(self.dataset)} samples]")
        except TypeError:
            # Older medmnist without ``size`` parameter
            if size != 28:
                print(f"  Warning: medmnist version does not support "
                      f"size={size}. Falling back to 28x28. "
                      f"Upgrade with: pip install --upgrade medmnist")
            self.dataset = DataClass(
                split=split,
                transform=None,
                download=download,
                root=data_root,
            )
            self.size = 28  # record actual size
            print(f"  Loaded {dataset_name} ({split}) – "
                  f"native 28x28  [{len(self.dataset)} samples]")

        self.num_classes = len(info['label'])
        self.class_names = list(info['label'].values())
        self.n_channels = info['n_channels']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # MedMNIST returns (image, label) where image is PIL or numpy
        image, label = self.dataset[idx]

        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image, mode='L')
            elif image.shape[-1] == 1:
                image = Image.fromarray(image.squeeze(), mode='L')
            else:
                image = Image.fromarray(image)

        # Convert grayscale -> RGB if requested
        if self.force_rgb and image.mode == 'L':
            image = image.convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Label is a numpy array, convert to tensor
        label = torch.tensor(label.squeeze(), dtype=torch.long)

        return image, label


def get_medmnist_loaders(
    dataset_name: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    data_root: str = "./data",
    medmnist_size: Optional[int] = None,
    force_rgb: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test loaders for MedMNIST datasets.

    By default the **native 224×224** variant is downloaded so that the
    transform pipeline does not need to upsample from 28×28.

    Args:
        dataset_name:  MedMNIST key, e.g. 'dermamnist', 'organsmnist'
        batch_size:    Batch size
        image_size:    Final image size the model expects (default 224)
        num_workers:   Number of data loading workers
        data_root:     Root directory for data
        medmnist_size: Override the MedMNIST download resolution.
                       When *None* (default), the best native size for
                       ``image_size`` is chosen automatically.
        force_rgb:     Convert grayscale datasets to 3-channel RGB

    Returns:
        train_loader, val_loader, test_loader
    """
    # Look up number of channels from MedMNIST info
    info = INFO[dataset_name]
    in_channels = info['n_channels']

    # If forcing RGB, treat as 3-channel for transforms
    if force_rgb:
        in_channels = 3

    # Resolve the download size
    if medmnist_size is not None:
        dl_size = medmnist_size
    else:
        dl_size = _resolve_medmnist_size(image_size)

    # If the native download already matches the model size we can skip
    # the Resize transform entirely.
    already_resized = (dl_size == image_size)

    if already_resized:
        print(f"Using native {dl_size}x{dl_size} MedMNIST images "
              f"(no resize needed)")
    else:
        print(f"Downloading {dl_size}x{dl_size} MedMNIST images, "
              f"will resize to {image_size}x{image_size}")

    # Create datasets
    train_dataset = MedMNISTWrapper(
        dataset_name=dataset_name,
        split="train",
        transform=get_train_transforms(image_size, in_channels, already_resized),
        size=dl_size,
        data_root=data_root,
        force_rgb=force_rgb,
    )

    val_dataset = MedMNISTWrapper(
        dataset_name=dataset_name,
        split="val",
        transform=get_val_transforms(image_size, in_channels, already_resized),
        size=dl_size,
        data_root=data_root,
        force_rgb=force_rgb,
    )

    test_dataset = MedMNISTWrapper(
        dataset_name=dataset_name,
        split="test",
        transform=get_val_transforms(image_size, in_channels, already_resized),
        size=dl_size,
        data_root=data_root,
        force_rgb=force_rgb,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# CIFAR-10 Dataset
# ============================================================================

class CIFAR10Wrapper(Dataset):
    """
    Wrapper around torchvision CIFAR-10 with our transform pipeline.

    CIFAR-10 images are 32×32 RGB; they are resized to ``image_size`` by the
    transform.  This dataset is included as a non-medical baseline to
    demonstrate that the controllability analysis generalises beyond the
    medical imaging domain.

    Args:
        split:     'train', 'val', or 'test'
        transform: torchvision transform
        download:  Whether to download if not present
        data_root: Root directory for data
        val_ratio: Fraction of training data held out for validation
        seed:      Random seed for the train/val split
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = True,
        data_root: str = "./data",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        if not TORCHVISION_DATASETS_AVAILABLE:
            raise ImportError(
                "torchvision datasets not available. "
                "Install with: pip install torchvision"
            )

        self.transform = transform
        self.split = split

        if split in ("train", "val"):
            full_dataset = tv_datasets.CIFAR10(
                root=data_root, train=True, download=download,
            )
            # Deterministic train/val split
            n = len(full_dataset)
            n_val = int(n * val_ratio)
            n_train = n - n_val

            generator = torch.Generator().manual_seed(seed)
            train_subset, val_subset = random_split(
                full_dataset, [n_train, n_val], generator=generator,
            )

            if split == "train":
                self.indices = train_subset.indices
                print(f"  Loaded CIFAR-10 (train) [{n_train} samples]")
            else:
                self.indices = val_subset.indices
                print(f"  Loaded CIFAR-10 (val) [{n_val} samples]")

            self.data = full_dataset.data        # [N, 32, 32, 3] uint8
            self.targets = full_dataset.targets   # list of int
        else:
            test_dataset = tv_datasets.CIFAR10(
                root=data_root, train=False, download=download,
            )
            self.indices = list(range(len(test_dataset)))
            self.data = test_dataset.data
            self.targets = test_dataset.targets
            print(f"  Loaded CIFAR-10 (test) [{len(self.indices)} samples]")

        self.num_classes = 10
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.data[real_idx]  # [32, 32, 3] uint8 numpy
        label = self.targets[real_idx]

        # Convert to PIL for transforms (consistent with MedMNIST path)
        image = Image.fromarray(image, mode='RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_cifar10_loaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 train / val / test loaders.

    Images are resized from 32×32 to ``image_size`` (default 224) so they
    can be fed into the same model architecture as the medical datasets.

    Args:
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers
        data_root: Root directory

    Returns:
        train_loader, val_loader, test_loader
    """
    in_channels = 3

    train_dataset = CIFAR10Wrapper(
        split="train",
        transform=get_train_transforms(image_size, in_channels,
                                       already_resized=False),
        data_root=data_root,
    )

    val_dataset = CIFAR10Wrapper(
        split="val",
        transform=get_val_transforms(image_size, in_channels,
                                     already_resized=False),
        data_root=data_root,
    )

    test_dataset = CIFAR10Wrapper(
        split="test",
        transform=get_val_transforms(image_size, in_channels,
                                     already_resized=False),
        data_root=data_root,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# CMMD Dataset (Mammography)
# ============================================================================

class CMMMDataset(Dataset):
    """
    Chinese Mammography Database (CMMD) loader.

    Note: This dataset requires manual download from The Cancer Imaging Archive
    (TCIA):
    https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508

    Expected directory structure:
        data_root/
        +-- CMMD/
            +-- images/
            |   +-- D1-0001/
            |   |   +-- 1-1.dcm (or .png)
            |   |   +-- ...
            |   +-- ...
            +-- labels.csv (columns: patient_id, image_id, label)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.data_root = Path(data_root) / "CMMD"
        self.transform = transform
        self.split = split

        # Load labels
        self.samples = self._load_samples(val_ratio, test_ratio, seed)

        self.num_classes = 2
        self.class_names = ["benign", "malignant"]

    def _load_samples(
        self,
        val_ratio: float,
        test_ratio: float,
        seed: int
    ) -> list:
        """Load and split samples."""
        import pandas as pd

        labels_path = self.data_root / "labels.csv"

        if not labels_path.exists():
            # Create dummy structure for demonstration
            print(f"Warning: {labels_path} not found. "
                  f"Creating placeholder dataset.")
            return self._create_placeholder_samples()

        df = pd.read_csv(labels_path)

        # Create sample list: (image_path, label)
        all_samples = []
        for _, row in df.iterrows():
            img_path = (self.data_root / "images"
                        / row['patient_id'] / f"{row['image_id']}.png")
            if img_path.exists():
                all_samples.append((str(img_path), row['label']))

        # Split
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))

        n_test = int(len(all_samples) * test_ratio)
        n_val = int(len(all_samples) * val_ratio)

        if self.split == "test":
            selected = indices[:n_test]
        elif self.split == "val":
            selected = indices[n_test:n_test + n_val]
        else:  # train
            selected = indices[n_test + n_val:]

        return [all_samples[i] for i in selected]

    def _create_placeholder_samples(self) -> list:
        """Create placeholder samples for testing without real data."""
        print("CMMD dataset requires manual download from TCIA.")
        print("See: https://wiki.cancerimagingarchive.net/"
              "pages/viewpage.action?pageId=70230508")
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path)

        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_cmmd_loaders(
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CMMD data loaders.

    Args:
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of workers
        data_root: Root directory

    Returns:
        train_loader, val_loader, test_loader
    """
    in_channels = 1  # Grayscale

    train_dataset = CMMMDataset(
        data_root=data_root,
        split="train",
        transform=get_train_transforms(image_size, in_channels),
    )

    val_dataset = CMMMDataset(
        data_root=data_root,
        split="val",
        transform=get_val_transforms(image_size, in_channels),
    )

    test_dataset = CMMMDataset(
        data_root=data_root,
        split="test",
        transform=get_val_transforms(image_size, in_channels),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Custom Dataset
# ============================================================================

class CustomImageDataset(Dataset):
    """
    Generic image classification dataset.

    Expected structure:
        data_root/
        +-- class_0/
        |   +-- img1.jpg
        |   +-- ...
        +-- class_1/
        |   +-- ...
        +-- ...
    """

    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        # Find classes (subdirectories)
        self.classes = sorted(
            [d.name for d in self.data_root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Collect samples
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_root / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append(
                        (str(img_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ============================================================================
# Main Interface
# ============================================================================

def get_dataloader(
    dataset_type: DatasetType,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    data_root: str = "./data",
    npz_path: Optional[str] = None,
    medmnist_size: Optional[int] = None,
    force_rgb: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for any supported dataset.

    For MedMNIST datasets the **native 224×224** variant is downloaded by
    default.  Pass ``medmnist_size=28`` to revert to the original
    low-resolution version.

    Args:
        dataset_type:  Type of dataset (see ``DatasetType`` enum)
        batch_size:    Batch size
        image_size:    Target image size that the model expects
        num_workers:   Number of data loading workers
        data_root:     Root directory for data
        npz_path:      Path to NPZ file (overrides standard loading)
        medmnist_size: Override the MedMNIST native download resolution.
                       When *None*, the best size for ``image_size`` is
                       chosen automatically (usually 224).
        force_rgb:     Convert grayscale MedMNIST datasets to 3-channel RGB

    Returns:
        train_loader, val_loader, test_loader
    """
    # If NPZ path provided, use NPZ loader
    if npz_path is not None:
        info = DATASET_INFO.get(dataset_type, {})
        in_channels = info.get('in_channels', 3)
        return get_npz_loaders(npz_path, batch_size, image_size,
                               num_workers, in_channels)

    # ---- MedMNIST datasets ----
    if dataset_type in _MEDMNIST_TYPES:
        medmnist_key = _MEDMNIST_TYPES[dataset_type]
        return get_medmnist_loaders(
            medmnist_key, batch_size, image_size,
            num_workers, data_root, medmnist_size, force_rgb,
        )

    # ---- CMMD ----
    if dataset_type == DatasetType.CMMD:
        return get_cmmd_loaders(batch_size, image_size, num_workers, data_root)

    # ---- CIFAR-10 ----
    if dataset_type == DatasetType.CIFAR10:
        return get_cifar10_loaders(batch_size, image_size, num_workers,
                                   data_root)

    raise ValueError(
        f"Unknown dataset type: {dataset_type}. "
        f"Supported: {[d.value for d in DatasetType]}"
    )


# ============================================================================
# Convenience: list all available datasets
# ============================================================================

def list_datasets(verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Print and return information about all supported datasets."""
    if verbose:
        print(f"{'Dataset':<20s} {'Classes':>7s}  {'Ch':>2s}  "
              f"{'Task':<20s}  Description")
        print("-" * 90)

    out = {}
    for dtype in DatasetType:
        if dtype == DatasetType.CUSTOM:
            continue
        info = DATASET_INFO.get(dtype, {})
        if not info:
            continue
        out[dtype.value] = info
        if verbose:
            print(f"{info['name']:<20s} {info['num_classes']:>7d}  "
                  f"{info['in_channels']:>2d}  "
                  f"{info['task']:<20s}  {info['description']}")
    return out


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Data Loader Test")
    print("=" * 90)

    # Print all datasets
    list_datasets()

    # Test MedMNIST if available
    if MEDMNIST_AVAILABLE:
        print("\n" + "=" * 90)
        # Test one RGB and one grayscale dataset
        for dtype, name in [
            (DatasetType.DERMAMNIST, "DermaMNIST"),
            (DatasetType.ORGANSMNIST, "OrganSMNIST"),
        ]:
            print(f"\nTesting {name} loader (224x224 native)...")
            try:
                train_loader, val_loader, test_loader = get_dataloader(
                    dtype,
                    batch_size=4,
                    image_size=224,
                    num_workers=0,
                )

                images, labels = next(iter(train_loader))
                print(f"  Batch shape: {images.shape}")
                print(f"  Labels: {labels.tolist()}")
                print(f"  Train size: {len(train_loader.dataset)}")
                print(f"  Val size: {len(val_loader.dataset)}")
                print(f"  Test size: {len(test_loader.dataset)}")
                print(f"  {name} loader works!")
            except Exception as e:
                print(f"  Error: {e}")

    # Test CIFAR-10
    if TORCHVISION_DATASETS_AVAILABLE:
        print(f"\nTesting CIFAR-10 loader...")
        try:
            train_loader, val_loader, test_loader = get_dataloader(
                DatasetType.CIFAR10,
                batch_size=4,
                image_size=224,
                num_workers=0,
            )

            images, labels = next(iter(train_loader))
            print(f"  Batch shape: {images.shape}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Train size: {len(train_loader.dataset)}")
            print(f"  Val size: {len(val_loader.dataset)}")
            print(f"  Test size: {len(test_loader.dataset)}")
            print("  CIFAR-10 loader works!")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("\nSkipping CIFAR-10 test (torchvision datasets not available)")