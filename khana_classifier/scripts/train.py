#!/usr/bin/env python
"""Phase-1 training script.

Usage:
    uv run python scripts/train.py --config configs/convnext_small.yaml
    uv run python scripts/train.py --config configs/debug.yaml
    uv run python scripts/train.py --config configs/convnext_small.yaml train.lr=0.001 train.epochs=30
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import shutil
import torch
from torch.utils.data import DataLoader
import timm.optim

from khana.config import parse_args
from khana.data import (
    ManifestDataset,
    build_train_transforms,
    build_val_transforms,
    build_mixup_collator,
    identity_collate,
    build_weighted_sampler,
)
from khana.models import build_model
from khana.models.factory import freeze_stem, unfreeze_all, count_params
from khana.training import train_one_epoch, evaluate, EMA, build_loss, build_scheduler
from khana.eval import compute_per_class_accuracy, save_confusion_matrix
from khana.tracking import MLflowTracker
from khana.utils import set_seed, Logger, save_checkpoint


def _aug_preview(loader: DataLoader, output_path: Path, num_images: int = 16) -> None:
    """Save a grid of augmented training images for pipeline sanity-check."""
    import torchvision
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from khana.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    images, _ = next(iter(loader))
    images = images[:num_images].cpu()
    images = images * std + mean          # un-normalise for display
    images = images.clamp(0, 1)

    grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main() -> None:
    cfg, _ = parse_args()
    log    = Logger()
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    project_dir  = Path(__file__).parent.parent  # khana_classifier/
    manifest_abs = (project_dir / cfg.data.manifest).resolve()
    data_root    = (project_dir / cfg.data.root).resolve()

    train_tfm = build_train_transforms(
        image_size=cfg.data.image_size,
        scale_min=cfg.aug.scale_min,
        trivial_augment=cfg.aug.trivial_augment,
        random_erase_prob=cfg.aug.random_erase_prob,
    )
    val_tfm = build_val_transforms(cfg.data.image_size)

    train_ds = ManifestDataset(manifest_abs, "train", data_root=data_root, transform=train_tfm,
                               max_samples_per_class=cfg.data.max_samples_per_class)
    val_ds   = ManifestDataset(manifest_abs, "val",   data_root=data_root, transform=val_tfm,
                               max_samples_per_class=cfg.data.max_samples_per_class)

    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Classes: {cfg.data.num_classes}")

    sampler     = build_weighted_sampler(train_ds.targets, power=cfg.data.sampler_power)
    collate_fn  = build_mixup_collator(
        num_classes=cfg.data.num_classes,
        mixup_alpha=cfg.aug.mixup_alpha,
        cutmix_alpha=cfg.aug.cutmix_alpha,
        prob=cfg.aug.mixup_prob,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        collate_fn=identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        cfg.model.name,
        num_classes=cfg.data.num_classes,
        drop_path_rate=cfg.model.drop_path_rate,
    ).to(device)

    total, trainable = count_params(model)
    log.info(f"Model: {cfg.model.name}  params={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M")

    # ── AMP / EMA / Loss ──────────────────────────────────────────────────────
    amp_dtype = None
    if cfg.train.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif cfg.train.amp_dtype == "float16":
        amp_dtype = torch.float16

    scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
    ema    = EMA(model, decay=cfg.train.ema_decay)
    loss_fn = build_loss(cfg.train.label_smoothing)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = timm.optim.create_optimizer_v2(
        model,
        opt=cfg.train.optimizer,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        layer_decay=cfg.model.layer_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer,
        epochs=cfg.train.epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        steps_per_epoch=len(train_loader),
        min_lr=cfg.train.min_lr,
    )

    # ── Augmentation preview ───────────────────────────────────────────────────
    preview_path = Path(cfg.train.checkpoint_dir) / "aug_preview.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    _aug_preview(train_loader, preview_path)
    log.info(f"Aug preview saved: {preview_path}")

    # ── Tracking ──────────────────────────────────────────────────────────────
    tracker = MLflowTracker(
        cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_system_metrics=cfg.mlflow.log_system_metrics,
    )

    best_top1 = 0.0
    best_ckpt = Path(cfg.train.checkpoint_dir) / "best_ema.pt"

    with tracker.start(run_name=cfg.mlflow.run_name, params=cfg.to_flat_dict()):
        tracker.log_artifact(preview_path)

        # Freeze stem for warmup epochs
        freeze_stem(model, epochs=cfg.train.warmup_epochs)

        for epoch in range(1, cfg.train.epochs + 1):
            # Unfreeze after warmup
            if epoch == cfg.train.warmup_epochs + 1:
                unfreeze_all(model)
                log.info("Stem unfrozen")

            train_metrics = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scheduler,
                device, epoch, amp_dtype, cfg.train.grad_accum_steps,
                cfg.train.clip_grad_norm, ema, scaler, log,
            )

            val_metrics = evaluate(
                model, val_loader, loss_fn, device, amp_dtype, log, prefix="val"
            )
            val_ema_metrics = evaluate(
                ema.module, val_loader, loss_fn, device, amp_dtype, log, prefix="val_ema"
            )

            current_lr = optimizer.param_groups[0]["lr"]
            all_metrics = {**train_metrics, **val_metrics, **val_ema_metrics, "lr": current_lr}

            tracker.log_metrics(all_metrics, step=epoch)
            log.log_metrics(all_metrics, epoch)

            # Checkpoint
            if epoch % cfg.train.save_every == 0:
                ckpt_path = Path(cfg.train.checkpoint_dir) / f"epoch_{epoch:03d}.pt"
                save_checkpoint(ckpt_path, epoch, model, ema, optimizer, scheduler, all_metrics, cfg.to_flat_dict())

            # Best model
            ema_top1 = val_ema_metrics["val_ema/top1"]
            if ema_top1 > best_top1:
                best_top1 = ema_top1
                save_checkpoint(best_ckpt, epoch, model, ema, optimizer, scheduler, all_metrics, cfg.to_flat_dict())
                log.success(f"New best EMA top-1: {best_top1:.2f}%  → {best_ckpt}")

        # ── Final eval + artifacts ────────────────────────────────────────────
        log.info(f"Training complete. Best val_ema/top1: {best_top1:.2f}%")

        # Confusion matrix on val set
        all_preds, all_targets = [], []
        ema.module.eval()
        from contextlib import nullcontext
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        with torch.inference_mode():
            for imgs, tgts in log.tqdm(val_loader, desc="Confusion matrix"):
                imgs = imgs.to(device, non_blocking=True)
                with amp_ctx:
                    preds = ema.module(imgs).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(tgts.tolist())

        cm_path = Path(cfg.train.checkpoint_dir) / "confusion_matrix.png"
        save_confusion_matrix(all_preds, all_targets, train_ds.classes, cm_path)
        tracker.log_artifact(cm_path)
        tracker.log_artifact(best_ckpt)
        tracker.log_model(ema.module, "best_ema_model")

        log.success(f"Done. Artifacts in {cfg.train.checkpoint_dir}/")


if __name__ == "__main__":
    main()
