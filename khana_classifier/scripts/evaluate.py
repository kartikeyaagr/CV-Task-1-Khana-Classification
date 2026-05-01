#!/usr/bin/env python
"""Final evaluation script with optional 5-crop TTA.

Usage:
    uv run python scripts/evaluate.py --checkpoint checkpoints/best_ema_hires.pt --split test
    uv run python scripts/evaluate.py --checkpoint checkpoints/best_ema_hires.pt --split test --tta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import _bootstrap  # noqa: F401 — adds src/ to sys.path

import torch
from torch.utils.data import DataLoader
import numpy as np

from khana.config import load_config
from khana.data import ManifestDataset, build_val_transforms, identity_collate
from khana.models import build_model
from khana.training import build_loss
from khana.training.engine import evaluate
from khana.eval import compute_per_class_accuracy, save_confusion_matrix
from khana.eval.tta import evaluate_tta
from khana.utils import Logger, load_checkpoint


def build_tta_loader(manifest_path, data_root, image_size, batch_size, num_workers):
    """For TTA: resize only (no center-crop); the TTA evaluator handles cropping."""
    from torchvision.transforms import v2
    from khana.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    import torch

    resize_size = int(image_size * 1.143)
    tfm = v2.Compose([
        v2.Resize(resize_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = ManifestDataset(manifest_path, "test", data_root=data_root, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=identity_collate, num_workers=num_workers, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/convnext_small.yaml")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--tta",        action="store_true", help="Enable 5-crop TTA")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    project_dir  = Path(__file__).parent.parent  # khana_classifier/
    cfg          = load_config(project_dir / args.config)
    log          = Logger()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_abs = (project_dir / cfg.data.manifest).resolve()

    amp_dtype = None
    if cfg.train.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif cfg.train.amp_dtype == "float16":
        amp_dtype = torch.float16

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        cfg.model.name,
        num_classes=cfg.data.num_classes,
        drop_path_rate=0.0,
        pretrained=False,
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── Loaders ───────────────────────────────────────────────────────────────
    data_root = (project_dir / cfg.data.root).resolve()
    val_tfm   = build_val_transforms(cfg.data.image_size)
    test_ds   = ManifestDataset(manifest_abs, args.split, data_root=data_root, transform=val_tfm)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=identity_collate, num_workers=cfg.data.num_workers, pin_memory=True,
    )

    loss_fn = build_loss(label_smoothing=0.0)   # no smoothing for eval

    # ── Standard eval ─────────────────────────────────────────────────────────
    log.info(f"Evaluating {args.split} split  ({len(test_ds):,} images)")
    metrics = evaluate(model, test_loader, loss_fn, device, amp_dtype, log, prefix=args.split)
    log.log_metrics(metrics, epoch=0)

    # ── TTA ───────────────────────────────────────────────────────────────────
    if args.tta:
        log.info("Running 5-crop TTA...")
        # Need a loader with only Resize (no center-crop) for TTA
        tta_loader = build_tta_loader(manifest_abs, data_root, cfg.data.image_size, args.batch_size, cfg.data.num_workers)
        tta_metrics = evaluate_tta(model, tta_loader, device, amp_dtype, log)
        log.log_metrics(tta_metrics, epoch=0)

    # ── Per-class accuracy ────────────────────────────────────────────────────
    log.info("Computing per-class accuracy...")
    per_class = compute_per_class_accuracy(model, test_loader, device, cfg.data.num_classes, amp_dtype)
    worst = np.argsort(per_class)[:10]
    log.info("Bottom-10 classes:")
    for idx in worst:
        log.info(f"  {test_ds.classes[idx]:<30}  {per_class[idx]*100:.1f}%")

    # Save per-class CSV
    out_csv = Path(args.checkpoint).parent / f"per_class_{args.split}.csv"
    with open(out_csv, "w") as f:
        f.write("class,accuracy\n")
        for i, cls in enumerate(test_ds.classes):
            f.write(f"{cls},{per_class[i]:.4f}\n")
    log.success(f"Per-class accuracy → {out_csv}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    with torch.inference_mode():
        from contextlib import nullcontext
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        for imgs, tgts in log.tqdm(test_loader, desc="Confusion matrix pass"):
            imgs = imgs.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(tgts.tolist())

    cm_path = Path(args.checkpoint).parent / f"confusion_matrix_{args.split}.png"
    save_confusion_matrix(all_preds, all_targets, test_ds.classes, cm_path)
    log.success(f"Confusion matrix → {cm_path}")


if __name__ == "__main__":
    main()
