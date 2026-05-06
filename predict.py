"""Classify a single image and print the predicted label.

Run:
    uv run python predict.py --image photo.jpg
    uv run python predict.py --image photo.jpg --checkpoint models/384/best_ema_hires.pt --image-size 384
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from data import val_transforms
from model import build_model
from utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Classify a single image")
    p.add_argument("--image", required=True, help="Path to image file")
    p.add_argument("--checkpoint", default="models/384/best_ema_hires.pt", help="Path to .pt checkpoint")
    p.add_argument("--manifest", default="splits/manifest.json", help="Manifest JSON with class list")
    p.add_argument("--model", default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--amp", default="bfloat16", choices=["bfloat16", "float16", "none"])
    return p.parse_args()


def main():
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    with open(manifest_path) as f:
        classes: list[str] = json.load(f)["classes"]

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[args.amp]

    model = build_model(args.model, num_classes=len(classes), drop_path_rate=0.0, pretrained=False).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    transform = val_transforms(args.image_size)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        with (torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else torch.inference_mode()):
            logits = model(tensor)

    label = classes[logits.argmax(dim=1).item()]
    print(label)


if __name__ == "__main__":
    main()
