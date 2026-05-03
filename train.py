"""Phase-1 training: ConvNeXt-Small @ 224px for 25 epochs.

Run:
    python train.py
    python train.py --debug
    python train.py --resume models/epoch_020.pt --epochs 50
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import timm.optim
import torchvision
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader

from data     import KhanaDataset, train_transforms, val_transforms, mixup_collate, plain_collate, weighted_sampler
from model    import build_model, freeze_stem, unfreeze_all, count_params
from engine   import train_one_epoch, evaluate
from metrics  import per_class_accuracy, save_confusion_matrix
from plots    import save_training_curves
from tracker  import Tracker
from utils    import set_seed, Logger, save_checkpoint, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--debug",       action="store_true", help="Quick 2-epoch smoke test")
    p.add_argument("--resume",      default=None,        help="Checkpoint to resume training from")
    p.add_argument("--manifest",    default="splits/manifest.json")
    p.add_argument("--data-root",   default="khana")
    p.add_argument("--model",       default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--epochs",      type=int,   default=25)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--wd",          type=float, default=0.05)
    p.add_argument("--image-size",  type=int,   default=224)
    p.add_argument("--amp",         default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--out-dir",     default="models")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def aug_preview(loader, path, n=16):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    imgs, _ = next(iter(loader))
    imgs = (imgs[:n].cpu() * std + mean).clamp(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, padding=2)
    plt.figure(figsize=(10,10)); plt.imshow(grid.permute(1,2,0)); plt.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()


def main():
    args = parse_args()
    log  = Logger()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Debug overrides ────────────────────────────────────────────────────────
    if args.debug:
        args.epochs, args.batch_size = 2, 32
        max_per_class = 13
        log.warn("DEBUG mode: 2 epochs, 13 images/class")
    else:
        max_per_class = None

    # ── Data ──────────────────────────────────────────────────────────────────
    num_classes = 80
    train_ds = KhanaDataset(args.manifest, "train", args.data_root,
                            transform=train_transforms(args.image_size),
                            max_per_class=max_per_class)
    val_ds   = KhanaDataset(args.manifest, "val",   args.data_root,
                            transform=val_transforms(args.image_size))
    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Classes: {num_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=weighted_sampler(train_ds.targets),
                              collate_fn=mixup_collate(num_classes),
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              collate_fn=plain_collate, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes=num_classes).to(device)
    ema   = ModelEmaV3(model, decay=0.9998)
    total, _ = count_params(model)
    log.info(f"Model: {args.model}  ({total/1e6:.1f}M params)")

    # ── AMP ───────────────────────────────────────────────────────────────────
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[args.amp]
    scaler    = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None

    # ── Loss / optimizer / scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = timm.optim.create_optimizer_v2(model, opt="adamw", lr=args.lr,
                                               weight_decay=args.wd, layer_decay=0.75)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs * len(train_loader),
                                  lr_min=1e-6, warmup_t=5 * len(train_loader),
                                  warmup_lr_init=1e-6, warmup_prefix=True, t_in_epochs=False)

    # ── Resume ────────────────────────────────────────────────────────────────
    history     = []
    start_epoch = 1
    if args.resume:
        ckpt        = load_checkpoint(args.resume, model, ema=ema,
                                      optimizer=optimizer, scheduler=scheduler, device=device)
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt.get("history", [])
        if start_epoch > 6:
            unfreeze_all(model)
        log.info(f"Resumed from {args.resume}  (epoch {ckpt['epoch']} → continuing to {args.epochs})")

    # ── Setup ─────────────────────────────────────────────────────────────────
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        aug_preview(train_loader, out / "aug_preview.png")
        log.info(f"Aug preview → {out/'aug_preview.png'}")

    tracker  = Tracker("khana-phase1", system_metrics=not args.debug)
    best     = max((h["ema_top1"] for h in history), default=0.0)
    best_path = out / "best_ema.pt"

    if not args.resume:
        freeze_stem(model)

    with tracker.run(name="debug" if args.debug else None, params=vars(args)):

        for epoch in range(start_epoch, args.epochs + 1):
            if epoch == 6 and not args.resume:
                unfreeze_all(model)

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch, amp_dtype, grad_accum=1, clip_norm=1.0,
                ema=ema, scaler=scaler,
            )
            val   = evaluate(model,       val_loader, criterion, device, amp_dtype)
            v_ema = evaluate(ema.module,  val_loader, criterion, device, amp_dtype)

            lr = max(g["lr"] for g in optimizer.param_groups)
            m  = {"train_loss": train_loss, "val_loss": val["loss"],
                  "val_top1": val["top1"],  "val_top5": val["top5"],
                  "ema_top1": v_ema["top1"], "ema_top5": v_ema["top5"], "lr": lr}
            tracker.log(m, epoch)
            log.metrics(m, epoch)

            history.append({"epoch": epoch, **m})

            if epoch % 5 == 0:
                save_checkpoint(out / f"epoch_{epoch:03d}.pt", epoch, model, ema,
                                optimizer, scheduler, m, vars(args), history)

            if v_ema["top1"] > best:
                best = v_ema["top1"]
                save_checkpoint(best_path, epoch, model, ema, optimizer, scheduler,
                                m, vars(args), history)
                log.ok(f"New best EMA top-1: {best:.2f}%")

        log.ok(f"Done. Best EMA top-1: {best:.2f}%  →  {best_path}")

        # ── Diagnostic plots ──────────────────────────────────────────────────
        save_training_curves(history, out, prefix="phase1")
        log.info(f"Training curves → {out}/phase1_training_curves.png")

        # ── Confusion matrix ──────────────────────────────────────────────────
        all_p, all_t = [], []
        with torch.inference_mode():
            amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
            for imgs, tgts in log.tqdm(val_loader, desc="Confusion matrix"):
                imgs = imgs.to(device, non_blocking=True)
                with amp_ctx:
                    all_p.extend(ema.module(imgs).argmax(1).cpu().tolist())
                all_t.extend(tgts.tolist())
        cm_path = out / "confusion_matrix.png"
        save_confusion_matrix(all_p, all_t, val_ds.classes, cm_path)
        tracker.artifact(cm_path); tracker.artifact(best_path)
        tracker.model(ema.module, "best_ema")


if __name__ == "__main__":
    main()
