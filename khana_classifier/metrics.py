"""Per-class accuracy and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def per_class_accuracy(preds: list[int], targets: list[int], num_classes: int) -> np.ndarray:
    preds_np   = np.array(preds)
    targets_np = np.array(targets)
    acc = np.zeros(num_classes)
    for c in range(num_classes):
        mask = targets_np == c
        if mask.sum() > 0:
            acc[c] = (preds_np[mask] == c).mean()
    return acc


def save_confusion_matrix(preds, targets, class_names, path):
    cm = confusion_matrix(targets, preds, normalize="true")
    n  = len(class_names)
    sz = max(12, n // 4)
    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
