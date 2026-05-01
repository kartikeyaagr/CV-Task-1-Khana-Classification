"""Dataset, transforms, sampling, splits, and collation."""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KhanaDataset(Dataset):
    """Reads images from a manifest JSON. Handles extension-less JPEG files."""

    def __init__(self, manifest_path: str, split: str, data_root: str,
                 transform: Optional[Callable] = None,
                 max_per_class: Optional[int] = None):
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.classes = manifest["classes"]
        self.data_root = Path(data_root)
        self.transform = transform

        samples = manifest[split]
        if max_per_class:
            counts: dict[int, int] = defaultdict(int)
            filtered = []
            for path, label in samples:
                if counts[label] < max_per_class:
                    filtered.append((path, label))
                    counts[label] += 1
            samples = filtered

        self.samples = [(str(self.data_root / p), int(lbl)) for p, lbl in samples]
        self.targets  = [lbl for _, lbl in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms ────────────────────────────────────────────────────────────────

def train_transforms(image_size=224, scale_min=0.7, random_erase=0.1):
    return v2.Compose([
        v2.RandomResizedCrop(image_size, scale=(scale_min, 1.0),
                             interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.TrivialAugmentWide(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        v2.RandomErasing(p=random_erase),
    ])


def val_transforms(image_size=224):
    resize = int(image_size * 1.143)
    return v2.Compose([
        v2.Resize(resize, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tta_transforms(image_size=224):
    """Resize only — TTA crops are applied in evaluate_tta()."""
    resize = int(image_size * 1.143)
    return v2.Compose([
        v2.Resize(resize, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Collation (MixUp / CutMix) ────────────────────────────────────────────────

def mixup_collate(num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    mixup  = v2.MixUp(alpha=mixup_alpha, num_classes=num_classes)
    cutmix = v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes)
    either = v2.RandomChoice([mixup, cutmix])

    def collate(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        if torch.rand(1).item() < prob:
            images, labels = either(images, labels)
        return images, labels

    return collate


def plain_collate(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


# ── Sampler ───────────────────────────────────────────────────────────────────

def weighted_sampler(targets: list[int], power=0.5) -> WeightedRandomSampler:
    counts = Counter(targets)
    w = torch.tensor([1.0 / (counts[t] ** power) for t in targets])
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


# ── Split generation ──────────────────────────────────────────────────────────

def build_manifest(dataset_root: str, output_path: str,
                   val_ratio=0.10, test_ratio=0.10, seed=42):
    root = Path(dataset_root).resolve()
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    classes = [d.name for d in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_paths, all_labels = [], []
    per_class_counts = defaultdict(int)
    for class_dir in class_dirs:
        idx = class_to_idx[class_dir.name]
        with os.scandir(class_dir) as it:
            for entry in it:
                if entry.name.startswith(".") or not entry.is_file(follow_symlinks=False):
                    continue
                all_paths.append(str(Path(entry.path).relative_to(root)))
                all_labels.append(idx)
                per_class_counts[idx] += 1

    paths_np  = np.array(all_paths)
    labels_np = np.array(all_labels)

    sp1 = StratifiedShuffleSplit(1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sp1.split(paths_np, labels_np))

    sp2 = StratifiedShuffleSplit(1, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    train_idx, val_idx = next(sp2.split(paths_np[trainval_idx], labels_np[trainval_idx]))

    def pairs(idxs, base_paths, base_labels):
        return [[base_paths[i], int(base_labels[i])] for i in idxs]

    manifest = {
        "classes": classes,
        "split_counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "train": pairs(train_idx,  paths_np[trainval_idx],  labels_np[trainval_idx]),
        "val":   pairs(val_idx,    paths_np[trainval_idx],  labels_np[trainval_idx]),
        "test":  pairs(test_idx,   paths_np,                labels_np),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f)

    print(f"Manifest → {output_path}")
    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    return manifest
