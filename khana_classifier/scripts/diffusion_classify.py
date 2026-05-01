#!/usr/bin/env python
"""Zero-shot diffusion classifier evaluation on a stratified subset of the test set.

Smoke test (fast):
    uv run python scripts/diffusion_classify.py --n-images 8 --timesteps 4 --noise-reps 1

Full overnight run:
    uv run python scripts/diffusion_classify.py --n-images 800 --timesteps 10 --noise-reps 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
import _bootstrap  # noqa: F401 — adds src/ to sys.path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2

from khana.config import load_config
from khana.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from khana.diffusion import DiffusionClassifier, build_prompts
from khana.eval import save_confusion_matrix
from khana.tracking import MLflowTracker
from khana.utils import Logger, set_seed


def _load_subset(manifest_path: str, n_per_class: int, seed: int) -> tuple[list, list, list[str]]:
    """Load a stratified subset: n_per_class images per class from the test split."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    rng     = random.Random(seed)

    by_class: dict[int, list] = defaultdict(list)
    for path, label in manifest["test"]:
        by_class[label].append(path)

    paths, labels = [], []
    for cls_idx, cls_paths in sorted(by_class.items()):
        sample = rng.sample(cls_paths, min(n_per_class, len(cls_paths)))
        paths.extend(sample)
        labels.extend([cls_idx] * len(sample))

    return paths, labels, classes


def _preprocess(path: str, image_size: int = 224) -> torch.Tensor:
    """Load + resize + normalize a single image → (1, 3, H, W) float32."""
    resize_size = int(image_size * 1.143)
    tfm = v2.Compose([
        v2.Resize(resize_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/convnext_small.yaml")
    parser.add_argument("--n-images",    type=int, default=8,   help="Images per class")
    parser.add_argument("--timesteps",   type=int, default=10,  help="Noise timesteps (T)")
    parser.add_argument("--noise-reps",  type=int, default=2,   help="Noise samples per timestep (N)")
    parser.add_argument("--template",    type=int, default=1,   help="Prompt template index (0-2)")
    parser.add_argument("--model-id",    default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    project_dir  = Path(__file__).parent.parent  # khana_classifier/
    cfg          = load_config(project_dir / args.config)
    log          = Logger()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    manifest_abs = (project_dir / cfg.data.manifest).resolve()
    paths, labels, classes = _load_subset(str(manifest_abs), args.n_images, args.seed)
    log.info(f"Subset: {len(paths)} images ({args.n_images}/class × {len(classes)} classes)")

    # ── Prompts ───────────────────────────────────────────────────────────────
    prompts = build_prompts(classes, template_idx=args.template)
    log.info(f"Prompt template: '{prompts[0]}'  (showing class 0)")

    # ── Classifier ────────────────────────────────────────────────────────────
    log.info(f"Loading {args.model_id} ...")
    clf = DiffusionClassifier(
        model_id=args.model_id,
        device=device,
        timesteps=args.timesteps,
        noise_reps=args.noise_reps,
        dtype=torch.float16,
    )

    log.info("Encoding class text prompts...")
    text_embeddings = clf.encode_prompts(prompts)   # (C, seq, D)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds: list[int] = []
    t0 = time.time()

    for i, (path, gt) in enumerate(log.tqdm(list(zip(paths, labels)), desc="Classifying")):
        image = _preprocess(path, image_size=cfg.data.image_size).to(device)
        pred  = clf.classify_image(image, text_embeddings)
        all_preds.append(pred)

    elapsed = time.time() - t0
    per_image_sec = elapsed / len(paths)

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_targets = np.array(labels)
    all_preds_np = np.array(all_preds)

    top1 = float((all_preds_np == all_targets).mean()) * 100
    # Top-5 approximation via argmin-5 is non-trivial; report top-1 only
    log.success(f"Top-1 accuracy: {top1:.2f}%  ({len(paths)} images)")
    log.info(f"Inference time: {per_image_sec:.1f}s/image  total: {elapsed/60:.1f}min")

    # Per-class
    per_class_acc = {}
    for cls_idx, cls_name in enumerate(classes):
        mask = all_targets == cls_idx
        if mask.sum() > 0:
            per_class_acc[cls_name] = float((all_preds_np[mask] == all_targets[mask]).mean()) * 100

    # ── Artifacts ─────────────────────────────────────────────────────────────
    out_dir = project_dir / "diffusion_results"
    out_dir.mkdir(exist_ok=True)

    cm_path = out_dir / "confusion_matrix_diffusion.png"
    save_confusion_matrix(all_preds, labels, classes, cm_path)

    csv_path = out_dir / "per_class_diffusion.csv"
    with open(csv_path, "w") as f:
        f.write("class,accuracy\n")
        for cls_name, acc in per_class_acc.items():
            f.write(f"{cls_name},{acc:.2f}\n")

    # ── MLflow logging ────────────────────────────────────────────────────────
    tracker = MLflowTracker("khana-diffusion-zs", tracking_uri=cfg.mlflow.tracking_uri, log_system_metrics=False)
    with tracker.start(run_name=f"sd21_T{args.timesteps}_N{args.noise_reps}_tmpl{args.template}",
                       params={
                           "model_id":          args.model_id,
                           "timesteps":         args.timesteps,
                           "noise_reps":        args.noise_reps,
                           "template_idx":      args.template,
                           "n_images_per_class": args.n_images,
                           "seed":              args.seed,
                           "prompt_example":    prompts[0],
                       }):
        tracker.log_metric("top1", top1, step=0)
        tracker.log_metric("per_image_sec", per_image_sec, step=0)
        tracker.log_artifact(cm_path)
        tracker.log_artifact(csv_path)

    log.success("Done. Results in diffusion_results/  MLflow run logged.")


if __name__ == "__main__":
    main()
