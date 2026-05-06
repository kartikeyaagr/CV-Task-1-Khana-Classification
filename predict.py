"""Test a trained checkpoint on a new folder of images.

Labeled mode   — folder contains one subdirectory per class:
    predict/
    ├── biryani/img1.jpg
    └── chapati/img2.jpg
  → Reports top-1 / top-5 accuracy, per-class breakdown, and a CSV.

Unlabeled mode — flat folder with images, no subdirectories:
    predict/
    ├── photo1.jpg
    └── photo2.jpg
  → Reports top-5 predicted classes per image (no accuracy).

Run:
    uv run python predict.py --images-dir /path/to/folder --checkpoint models/384/best_ema_hires.pt --image-size 384
    uv run python predict.py --images-dir /path/to/folder --checkpoint models/384/best_ema_hires.pt --image-size 384 --tta
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from data import val_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import build_model
from utils import Logger, load_checkpoint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ── Dataset helpers ───────────────────────────────────────────────────────────


class ImageFolderDataset(Dataset):
    """Loads images from a flat or class-structured directory."""

    def __init__(self, image_paths: list[Path], transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)


def collate_fn(batch):
    imgs, paths = zip(*batch)
    return torch.stack(imgs), list(paths)


# ── TTA (5-crop) ──────────────────────────────────────────────────────────────


def tta_transform(image_size: int):
    resize = int(image_size * 1.143)
    return v2.Compose(
        [
            v2.Resize(
                resize, interpolation=v2.InterpolationMode.BICUBIC, antialias=True
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def predict_tta(
    model, img_batch: torch.Tensor, image_size: int, device, amp_ctx
) -> torch.Tensor:
    """Five-crop TTA: centre + four corners, averaged in prob space."""
    crops = v2.FiveCrop(image_size)
    all_logits = []
    with amp_ctx:
        for crop in crops(img_batch):
            all_logits.append(model(crop.to(device)))
    return torch.stack(all_logits).mean(0)


# ── Inference ─────────────────────────────────────────────────────────────────


def run_inference(
    model, loader, device, amp_dtype, use_tta: bool, image_size: int, log
):
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype
        else nullcontext()
    )
    all_probs, all_paths = [], []

    with torch.inference_mode():
        for imgs, paths in log.tqdm(loader, desc="Classifying"):
            imgs = imgs.to(device, non_blocking=True)
            if use_tta:
                logits = predict_tta(model, imgs, image_size, device, amp_ctx)
            else:
                with amp_ctx:
                    logits = model(imgs)
            all_probs.append(F.softmax(logits, dim=-1).cpu())
            all_paths.extend(paths)

    return torch.cat(all_probs, dim=0), all_paths


# ── Accuracy helpers ──────────────────────────────────────────────────────────


def topk_correct(probs: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = probs.topk(k, dim=1).indices
    return topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Test model on a new image folder")
    p.add_argument(
        "--images-dir", required=True, help="Folder of test images", default="images"
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint",
        default="models/320/best_ema_hires.pt",
    )
    p.add_argument(
        "--manifest",
        default="splits/manifest.json",
        help="Manifest JSON (used to load class list)",
    )
    p.add_argument("--model", default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", default="bfloat16", choices=["bfloat16", "float16", "none"])
    p.add_argument("--tta", action="store_true", help="5-crop test-time augmentation")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k predictions printed per image in unlabeled mode",
    )
    p.add_argument(
        "--out-file",
        default=None,
        help="Save results JSON here (auto-named if omitted)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log = Logger()

    # ── Device + AMP ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[
        args.amp
    ]
    log.info(f"Device: {device}  AMP: {args.amp}  TTA: {args.tta}")

    # ── Load class list from manifest ─────────────────────────────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.warn(
            f"Manifest not found at {manifest_path}. Cannot map predictions to class names."
        )
        sys.exit(1)
    with open(manifest_path) as f:
        classes: list[str] = json.load(f)["classes"]
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    log.info(f"Classes loaded: {num_classes} from {manifest_path}")

    # ── Discover images & detect labeled vs. unlabeled ────────────────────────
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        log.warn(f"--images-dir '{images_dir}' is not a directory.")
        sys.exit(1)

    # Labeled: subdirs match known class names
    subdirs = [
        d for d in images_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    labeled_subdirs = [d for d in subdirs if d.name in class_to_idx]

    if labeled_subdirs:
        log.info(f"Labeled mode — found {len(labeled_subdirs)} class subdirectories")
        image_paths, labels = [], []
        unknown_dirs = [d.name for d in subdirs if d.name not in class_to_idx]
        if unknown_dirs:
            log.warn(f"Skipping unrecognised subdirs: {unknown_dirs}")
        for d in sorted(labeled_subdirs):
            idx = class_to_idx[d.name]
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    image_paths.append(f)
                    labels.append(idx)
        labeled = True
        log.info(f"Found {len(image_paths):,} labeled images")
    else:
        log.info(
            "Unlabeled mode — no class subdirectories detected, running prediction only"
        )
        image_paths = sorted(
            f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
        )
        labels = []
        labeled = False
        log.info(f"Found {len(image_paths):,} images")

    if not image_paths:
        log.warn(
            "No images found. Check --images-dir and that images have supported extensions."
        )
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        args.model, num_classes=num_classes, drop_path_rate=0.0, pretrained=False
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    transform = (
        tta_transform(args.image_size) if args.tta else val_transforms(args.image_size)
    )
    ds = ImageFolderDataset(image_paths, transform)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    probs, returned_paths = run_inference(
        model, loader, device, amp_dtype, args.tta, args.image_size, log
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report: dict = {
        "checkpoint": str(args.checkpoint),
        "images_dir": str(images_dir),
        "num_images": len(image_paths),
        "tta": args.tta,
    }

    if labeled:
        label_tensor = torch.tensor(labels, dtype=torch.long)
        top1 = topk_correct(probs, label_tensor, 1) / len(labels) * 100
        top5 = topk_correct(probs, label_tensor, 5) / len(labels) * 100
        log.ok(f"Top-1 accuracy: {top1:.2f}%")
        log.ok(f"Top-5 accuracy: {top5:.2f}%")
        report.update({"top1": round(top1, 4), "top5": round(top5, 4)})

        # Per-class accuracy
        per_class_correct = {c: [0, 0] for c in classes}  # [correct, total]
        for prob_row, true_label in zip(probs, labels):
            cls_name = classes[true_label]
            pred = prob_row.argmax().item()
            per_class_correct[cls_name][1] += 1
            if pred == true_label:
                per_class_correct[cls_name][0] += 1

        per_class_acc = {
            c: round(v[0] / v[1], 4) if v[1] > 0 else None
            for c, v in per_class_correct.items()
            if v[1] > 0
        }
        report["per_class_accuracy"] = per_class_acc

        # Show bottom-10 classes
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
        log.info("Bottom-10 classes:")
        for cls, acc in sorted_classes[:10]:
            log.info(f"  {cls:<35} {acc*100:.1f}%")

        # Save CSV
        out_dir = Path(args.out_file).parent if args.out_file else images_dir
        csv_path = out_dir / "per_class_accuracy.csv"
        with open(csv_path, "w") as f:
            f.write("class,correct,total,accuracy\n")
            for cls in classes:
                if cls in per_class_correct and per_class_correct[cls][1] > 0:
                    correct, total = per_class_correct[cls]
                    f.write(f"{cls},{correct},{total},{correct/total:.4f}\n")
        log.ok(f"Per-class CSV → {csv_path}")

    else:
        # Unlabeled: print top-k predictions per image
        topk = min(args.top_k, num_classes)
        predictions = []
        for path, prob_row in zip(returned_paths, probs):
            top_probs, top_idxs = prob_row.topk(topk)
            preds = [
                {"class": classes[i.item()], "confidence": round(p.item(), 4)}
                for p, i in zip(top_probs, top_idxs)
            ]
            predictions.append({"image": Path(path).name, "predictions": preds})

        # Print table
        log.info(f"\nTop-{topk} predictions:")
        for entry in predictions:
            top1_pred = entry["predictions"][0]
            log.info(
                f"  {entry['image']:<40} → {top1_pred['class']:<30} "
                f"({top1_pred['confidence']*100:.1f}%)"
            )
        report["predictions"] = predictions

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_file = (
        Path(args.out_file) if args.out_file else images_dir / "test_results.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    log.ok(f"Results saved → {out_file}")


if __name__ == "__main__":
    main()
