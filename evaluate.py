"""Final evaluation with optional 5-crop TTA.

Run:
    python evaluate.py --checkpoint checkpoints/best_ema_hires.pt
    python evaluate.py --checkpoint checkpoints/best_ema_hires.pt --tta
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data    import KhanaDataset, val_transforms, tta_transforms, plain_collate
from model   import build_model
from engine  import evaluate as eval_loop, evaluate_tta
from metrics import per_class_accuracy, save_confusion_matrix
from utils   import Logger, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--manifest",    default="splits/manifest.json")
    p.add_argument("--data-root",   default="khana")
    p.add_argument("--model",       default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--split",       default="test", choices=["val", "test"])
    p.add_argument("--image-size",  type=int, default=320)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--amp",         default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--tta",         action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    log    = Logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[args.amp]

    model = build_model(args.model, num_classes=80, drop_path_rate=0.0, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    log.info(f"Loaded {args.checkpoint}")

    ds = KhanaDataset(args.manifest, args.split, args.data_root,
                      transform=val_transforms(args.image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=plain_collate,
                        shuffle=False, num_workers=4, pin_memory=True)
    log.info(f"Evaluating {args.split}: {len(ds):,} images")

    criterion = nn.CrossEntropyLoss()
    results   = eval_loop(model, loader, criterion, device, amp_dtype)
    log.info(f"top-1: {results['top1']:.2f}%  top-5: {results['top5']:.2f}%")

    if args.tta:
        tta_ds     = KhanaDataset(args.manifest, args.split, args.data_root,
                                  transform=tta_transforms(args.image_size))
        tta_loader = DataLoader(tta_ds, batch_size=args.batch_size, collate_fn=plain_collate,
                                shuffle=False, num_workers=4, pin_memory=True)
        tta_res = evaluate_tta(model, tta_loader, device, amp_dtype)
        log.info(f"TTA top-1: {tta_res['tta_top1']:.2f}%  top-5: {tta_res['tta_top5']:.2f}%")

    # Per-class accuracy and confusion matrix
    out = Path(args.checkpoint).parent
    all_p, all_t = [], []
    from contextlib import nullcontext
    with torch.inference_mode():
        amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
        for imgs, tgts in log.tqdm(loader, desc="Per-class pass"):
            imgs = imgs.to(device, non_blocking=True)
            with amp_ctx:
                all_p.extend(model(imgs).argmax(1).cpu().tolist())
            all_t.extend(tgts.tolist())

    acc = per_class_accuracy(all_p, all_t, 80)
    worst = np.argsort(acc)[:10]
    log.info("Bottom-10 classes:")
    for i in worst:
        log.info(f"  {ds.classes[i]:<30}  {acc[i]*100:.1f}%")

    csv_path = out / f"per_class_{args.split}.csv"
    with open(csv_path, "w") as f:
        f.write("class,accuracy\n")
        for i, cls in enumerate(ds.classes):
            f.write(f"{cls},{acc[i]:.4f}\n")

    cm_path = out / f"confusion_matrix_{args.split}.png"
    save_confusion_matrix(all_p, all_t, ds.classes, cm_path)
    log.ok(f"Results → {out}")


if __name__ == "__main__":
    main()
