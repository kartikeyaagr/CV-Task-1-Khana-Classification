"""Phase-2 high-resolution fine-tuning @ 320px.

Run after phase-1:
    python finetune_hires.py --checkpoint models/best_ema.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import timm.optim
from timm.utils import ModelEmaV3
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from data    import KhanaDataset, train_transforms, val_transforms, plain_collate, weighted_sampler
from model   import build_model
from engine  import train_one_epoch, evaluate
from plots   import save_training_curves
from tracker import Tracker
from utils   import set_seed, Logger, save_checkpoint, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--manifest",    default="splits/manifest.json")
    p.add_argument("--data-root",   default="khana")
    p.add_argument("--model",       default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--epochs",      type=int,   default=8)
    p.add_argument("--batch-size",  type=int,   default=24)
    p.add_argument("--grad-accum",  type=int,   default=3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--wd",          type=float, default=0.05)
    p.add_argument("--image-size",  type=int,   default=320)
    p.add_argument("--amp",         default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--out-dir",     default="models")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num-workers", type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    log  = Logger()
    set_seed(args.seed)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 80

    train_ds = KhanaDataset(args.manifest, "train", args.data_root,
                            transform=train_transforms(args.image_size, random_erase=0.05))
    val_ds   = KhanaDataset(args.manifest, "val",   args.data_root,
                            transform=val_transforms(args.image_size))

    # No MixUp in phase-2 — sharpen on hard labels
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=weighted_sampler(train_ds.targets),
                              collate_fn=plain_collate,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              collate_fn=plain_collate, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    model = build_model(args.model, num_classes=num_classes, pretrained=False).to(device)
    ema   = ModelEmaV3(model, decay=0.9998)
    load_checkpoint(args.checkpoint, model, ema=ema, device=device)
    log.info(f"Loaded {args.checkpoint}  →  fine-tuning @ {args.image_size}px")

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[args.amp]
    scaler    = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = timm.optim.create_optimizer_v2(model, opt="adamw", lr=args.lr,
                                               weight_decay=args.wd, layer_decay=0.75)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs * len(train_loader),
                                  lr_min=1e-7, warmup_t=len(train_loader),
                                  warmup_lr_init=1e-7, warmup_prefix=True, t_in_epochs=False)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tracker   = Tracker("khana-phase2", system_metrics=True)
    best, best_path = 0.0, out / "best_ema_hires.pt"
    history   = []

    with tracker.run(params=vars(args)):
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch, amp_dtype, args.grad_accum, 1.0, ema, scaler,
            )
            val   = evaluate(model,      val_loader, criterion, device, amp_dtype)
            v_ema = evaluate(ema.module, val_loader, criterion, device, amp_dtype)

            lr = max(g["lr"] for g in optimizer.param_groups)
            m  = {"train_loss": train_loss, "val_loss": val["loss"],
                  "val_top1": val["top1"],  "ema_top1": v_ema["top1"],
                  "ema_top5": v_ema["top5"], "lr": lr}
            tracker.log(m, epoch); log.metrics(m, epoch)

            history.append({"epoch": epoch, **m})

            if v_ema["top1"] > best:
                best = v_ema["top1"]
                save_checkpoint(best_path, epoch, model, ema, optimizer, scheduler,
                                m, vars(args), history)
                log.ok(f"Hires best: {best:.2f}%")

    log.ok(f"Phase-2 done. Best EMA top-1: {best:.2f}%")

    save_training_curves(history, out, prefix="phase2")
    log.info(f"Training curves → {out}/phase2_training_curves.png")

    tracker.artifact(best_path); tracker.model(ema.module, "best_ema_hires")


if __name__ == "__main__":
    main()
