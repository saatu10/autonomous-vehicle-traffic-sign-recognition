"""
=============================================================
  AV Perception Lab — Dataset & Preprocessing
  Handles GTSRB (speed limit classes) and a Traffic Light
  dataset (4 states: Red / Yellow / Green / Off)
=============================================================

Preprocessing rationale
------------------------
1. Resize to 224×224   — matches MobileNetV2 expected input
2. Color images (RGB)  — color is critical for TL detection and
                         helps speed signs too (red borders)
3. Normalize with      — ImageNet mean/std since we fine-tune
   ImageNet stats        a pre-trained backbone
4. Augmentation        — handles the long-tail distribution of
                         real-world traffic sign frequencies
"""

import os
import csv
import random
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ─────────────────────────────────────────────
#  ImageNet normalization constants
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# GTSRB class IDs that correspond to speed limits
# 0: 20 km/h, 1: 30 km/h, 2: 50 km/h, 3: 60 km/h
# 4: 70 km/h, 5: 80 km/h, 7: 100 km/h, 8: 120 km/h
SPEED_LIMIT_CLASS_IDS = {0, 1, 2, 3, 4, 5, 7, 8}
SPEED_ID_REMAP = {old: new for new, old in enumerate(sorted(SPEED_LIMIT_CLASS_IDS))}

# Traffic light state labels
TL_LABELS = ["Red", "Yellow", "Green", "Off"]


# ─────────────────────────────────────────────
#  Transform factories
# ─────────────────────────────────────────────
def build_train_transform(img_size: int = 224) -> transforms.Compose:
    """
    Augmentation-heavy transform for training.

    Key choices:
    - RandomResizedCrop: simulates varying distances/viewpoints
    - ColorJitter: handles illumination changes (day/night/weather)
    - RandomRotation ±15°: signs can be slightly tilted
    - GaussianBlur: mimics motion blur at speed
    - Normalize: ImageNet stats for fine-tuning backbone
    """
    return transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),      # slight oversize
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.1),                  # rare for signs
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation/test — no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalisation for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ─────────────────────────────────────────────
#  GTSRB Speed Limit Dataset
# ─────────────────────────────────────────────
class GTSRBSpeedDataset(Dataset):
    """
    Loads GTSRB images for speed-limit sign classes only.

    Expected directory layout (standard GTSRB):
        root/
          Train/
            00000/   ← class 0 (20 km/h)
              GT-00000.csv
              *.ppm
            00001/   ← class 1 (30 km/h)
            ...

    Parameters
    ----------
    root       : Path to GTSRB root directory
    split      : 'train' | 'val' | 'test'
    transform  : torchvision transform (auto-selected if None)
    val_frac   : fraction of train set held out for validation
    seed       : random seed for reproducible splits
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or (
            build_train_transform() if split == "train" else build_val_transform()
        )

        self.samples: list[tuple[Path, int]] = []  # (image_path, remapped_label)
        self._load_samples(val_frac, seed)

    def _load_samples(self, val_frac: float, seed: int):
        train_dir = self.root / "Train"
        all_samples: list[tuple[Path, int]] = []

        for class_id in sorted(SPEED_LIMIT_CLASS_IDS):
            class_dir = train_dir / f"{class_id:05d}"
            if not class_dir.exists():
                continue

            label = SPEED_ID_REMAP[class_id]
            imgs  = sorted(class_dir.glob("*.ppm"))
            all_samples.extend([(p, label) for p in imgs])

        # Reproducible split
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        n_val = int(len(all_samples) * val_frac)
        if self.split == "train":
            self.samples = all_samples[n_val:]
        elif self.split == "val":
            self.samples = all_samples[:n_val]
        else:
            # Use the GTSRB test CSV if available, else fallback to val split
            test_csv = self.root / "Test.csv"
            if test_csv.exists():
                self._load_test_csv(test_csv)
            else:
                self.samples = all_samples[:n_val]

    def _load_test_csv(self, csv_path: Path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_id = int(row["ClassId"])
                if class_id in SPEED_LIMIT_CLASS_IDS:
                    img_path = self.root / row["Path"]
                    self.samples.append((img_path, SPEED_ID_REMAP[class_id]))

    # ── Class weights for loss / sampler ──────────────────────────
    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for each class.
        Used by WeightedRandomSampler and weighted CE loss.
        """
        counts = torch.zeros(len(SPEED_LIMIT_CLASS_IDS))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum()            # normalised

    def sample_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler."""
        cw = self.class_weights()
        return [cw[label].item() for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


# ─────────────────────────────────────────────
#  Traffic Light Dataset
# ─────────────────────────────────────────────
class TrafficLightDataset(Dataset):
    """
    Simple folder-based dataset for traffic light state.

    Expected structure:
        root/
          Red/     *.jpg / *.png
          Yellow/
          Green/
          Off/

    Parameters
    ----------
    root       : path to dataset root
    split      : 'train' | 'val'
    transform  : optional override
    val_frac   : fraction held out for val
    seed       : reproducibility
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_frac: float = 0.15,
        seed: int = 42,
    ):
        self.root      = Path(root)
        self.split     = split
        self.transform = transform or (
            build_train_transform() if split == "train" else build_val_transform()
        )
        self.samples: list[tuple[Path, int]] = []
        self._load(val_frac, seed)

    def _load(self, val_frac: float, seed: int):
        all_samples: list[tuple[Path, int]] = []
        for label_idx, state in enumerate(TL_LABELS):
            state_dir = self.root / state
            if not state_dir.exists():
                continue
            exts = ("*.jpg", "*.jpeg", "*.png", "*.ppm")
            imgs = [p for ext in exts for p in sorted(state_dir.glob(ext))]
            all_samples.extend([(p, label_idx) for p in imgs])

        rng = random.Random(seed)
        rng.shuffle(all_samples)
        n_val = int(len(all_samples) * val_frac)
        self.samples = all_samples[n_val:] if self.split == "train" else all_samples[:n_val]

    def class_weights(self) -> torch.Tensor:
        counts = torch.zeros(len(TL_LABELS))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum()

    def sample_weights(self) -> list[float]:
        cw = self.class_weights()
        return [cw[label].item() for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


# ─────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────
def build_dataloaders(
    speed_root: str | Path,
    tl_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> dict[str, dict[str, DataLoader]]:
    """
    Returns nested dict:
        {
          'speed':   {'train': DataLoader, 'val': DataLoader},
          'tl'   :   {'train': DataLoader, 'val': DataLoader},
        }
    """
    loaders: dict = {}

    for task, root, Cls in [
        ("speed", speed_root, GTSRBSpeedDataset),
        ("tl",    tl_root,    TrafficLightDataset),
    ]:
        tr_ds  = Cls(root, split="train")
        val_ds = Cls(root, split="val")

        sampler = None
        if use_weighted_sampler:
            sw = tr_ds.sample_weights()
            sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

        loaders[task] = {
            "train": DataLoader(
                tr_ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            "val": DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            ),
        }

    return loaders
