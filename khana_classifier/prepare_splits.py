"""Generate stratified train/val/test split manifest.

Run once before training:
    python prepare_splits.py
"""

from data import build_manifest

build_manifest(
    dataset_root="../khana",
    output_path="../splits/manifest.json",
    val_ratio=0.10,
    test_ratio=0.10,
    seed=42,
)
