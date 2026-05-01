#!/usr/bin/env python
"""Phase-2 high-resolution fine-tuning.

Loads a phase-1 EMA checkpoint, then trains at a larger resolution
with no MixUp (sharpening on hard labels).

Usage:
    uv run python scripts/finetune_hires.py \\
        --config configs/convnext_small_hires.yaml \\
        --checkpoint checkpoints/best_ema.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
import timm.optim

from khana.config import load_config
from khana.data import (
    ManifestDataset,
    build_train_transforms,
    build_val_transforms,
    build_mixup_collator,
    identity_collate,
    build_weighted_sampler,
)
from khana.models import build_model
from khana.models.factory import count_params
from khana.training import train_one_epoch, evaluate, EMA, build_loss, build_scheduler
from khana.eval import save_confusion_matrix
from khana.tracking import MLflowTracker
from khana.utils import set_seed, Logger, save_checkpoint, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True, help="Phase-1 best_ema.pt")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent  # khana_classifier/
    cfg = load_config(args.config, args.overrides)
    log = Logger()
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Phase-2 fine-tune @ {cfg.data.image_size}px  device={device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    manifest_abs = (project_dir / cfg.data.manifest).resolve()
    data_root    = (project_dir / cfg.data.root).resolve()

    train_tfm = build_train_transforms(
        image_size=cfg.data.image_size,
        scale_min=cfg.aug.scale_min,
        trivial_augment=cfg.aug.trivial_augment,
        random_erase_prob=cfg.aug.random_erase_prob,
    )
    val_tfm = build_val_transforms(cfg.data.image_size)

    train_ds = ManifestDataset(manifest_abs, "train", data_root=data_root, transform=train_tfm)
    val_ds   = ManifestDataset(manifest_abs, "val",   data_root=data_root, transform=val_tfm)

    sampler = build_weighted_sampler(train_ds.targets, power=cfg.data.sampler_power)

    # Phase-2: mixup disabled (prob=0 in hires config)
    collate_fn = build_mixup_collator(
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
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=identity_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
    )

    # ── Model — load from EMA checkpoint ──────────────────────────────────────
    model = build_model(
        cfg.model.name,
        num_classes=cfg.data.num_classes,
        drop_path_rate=cfg.model.drop_path_rate,
        pretrained=False,   # weights come from checkpoint
    ).to(device)

    ema = EMA(model, decay=cfg.train.ema_decay)
    ckpt = load_checkpoint(args.checkpoint, model, ema=ema, device=device)
    log.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  (phase-1 best_ema/top1={ckpt['metrics'].get('val_ema/top1', '?'):.2f}%)")

    amp_dtype = None
    if cfg.train.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif cfg.train.amp_dtype == "float16":
        amp_dtype = torch.float16

    scaler  = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
    loss_fn = build_loss(cfg.train.label_smoothing)

    optimizer = timm.optim.create_optimizer_v2(
        model,
        opt=cfg.train.optimizer,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        layer_decay=cfg.model.layer_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=cfg.train.epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        steps_per_epoch=len(train_loader),
        min_lr=cfg.train.min_lr,
    )

    tracker = MLflowTracker(
        cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_system_metrics=cfg.mlflow.log_system_metrics,
    )

    best_top1 = 0.0
    best_ckpt = Path(cfg.train.checkpoint_dir) / "best_ema_hires.pt"

    with tracker.start(run_name=cfg.mlflow.run_name, params=cfg.to_flat_dict()):
        for epoch in range(1, cfg.train.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scheduler,
                device, epoch, amp_dtype, cfg.train.grad_accum_steps,
                cfg.train.clip_grad_norm, ema, scaler, log,
            )
            val_metrics     = evaluate(model,       val_loader, loss_fn, device, amp_dtype, log, "val")
            val_ema_metrics = evaluate(ema.module,  val_loader, loss_fn, device, amp_dtype, log, "val_ema")

            current_lr = optimizer.param_groups[0]["lr"]
            all_metrics = {**train_metrics, **val_metrics, **val_ema_metrics, "lr": current_lr}
            tracker.log_metrics(all_metrics, step=epoch)
            log.log_metrics(all_metrics, epoch)

            if epoch % cfg.train.save_every == 0:
                save_checkpoint(
                    Path(cfg.train.checkpoint_dir) / f"hires_epoch_{epoch:03d}.pt",
                    epoch, model, ema, optimizer, scheduler, all_metrics, cfg.to_flat_dict()
                )

            ema_top1 = val_ema_metrics["val_ema/top1"]
            if ema_top1 > best_top1:
                best_top1 = ema_top1
                save_checkpoint(best_ckpt, epoch, model, ema, optimizer, scheduler, all_metrics, cfg.to_flat_dict())
                log.success(f"Hires best: {best_top1:.2f}% → {best_ckpt}")

        log.info(f"Phase-2 complete. Best val_ema/top1: {best_top1:.2f}%")
        tracker.log_artifact(best_ckpt)
        tracker.log_model(ema.module, "best_ema_hires_model")


if __name__ == "__main__":
    main()
