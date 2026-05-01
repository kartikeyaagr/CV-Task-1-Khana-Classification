"""Zero-shot diffusion classifier evaluation.

Smoke test:   python diffusion_classify.py --n-per-class 1 --timesteps 4 --noise-reps 1
Full run:     python diffusion_classify.py --n-per-class 10 --timesteps 10 --noise-reps 2
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from data      import val_transforms
from diffusion import DiffusionClassifier, build_prompts
from metrics   import save_confusion_matrix
from tracker   import Tracker
from utils     import Logger, set_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",     default="splits/manifest.json")
    p.add_argument("--data-root",    default="khana")
    p.add_argument("--n-per-class",  type=int, default=10)
    p.add_argument("--timesteps",    type=int, default=10)
    p.add_argument("--noise-reps",   type=int, default=2)
    p.add_argument("--template",     type=int, default=1)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    log  = Logger(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.manifest) as f:
        manifest = json.load(f)
    classes = manifest["classes"]

    # Build stratified subset
    rng = random.Random(args.seed)
    by_class = {}
    for path, label in manifest["test"]:
        by_class.setdefault(label, []).append(path)

    paths, labels = [], []
    data_root = Path(args.data_root)
    for cls_idx in sorted(by_class):
        sample = rng.sample(by_class[cls_idx], min(args.n_per_class, len(by_class[cls_idx])))
        paths.extend([str(data_root / p) for p in sample])
        labels.extend([cls_idx] * len(sample))

    log.info(f"Subset: {len(paths)} images ({args.n_per_class}/class × {len(classes)} classes)")

    tfm = val_transforms(224)

    log.info(f"Loading Stable Diffusion 2.1 ...")
    clf = DiffusionClassifier(timesteps=args.timesteps, noise_reps=args.noise_reps, device=device)
    prompts = build_prompts(classes, template_idx=args.template)
    log.info(f"Prompt example: '{prompts[0]}'")
    text_emb = clf.encode_prompts(prompts)

    preds, t0 = [], time.time()
    for path in log.tqdm(paths, desc="Classifying"):
        img = tfm(Image.open(path).convert("RGB")).unsqueeze(0)
        preds.append(clf.classify(img, text_emb))

    elapsed = time.time() - t0
    top1 = float((np.array(preds) == np.array(labels)).mean()) * 100
    log.ok(f"Top-1: {top1:.2f}%   {elapsed/len(paths):.1f}s/image   total: {elapsed/60:.1f}min")

    out = Path("diffusion_results"); out.mkdir(exist_ok=True)
    cm_path = out / "confusion_matrix.png"
    save_confusion_matrix(preds, labels, classes, cm_path)

    tracker = Tracker("khana-diffusion-zs", system_metrics=False)
    with tracker.run(name=f"T{args.timesteps}_N{args.noise_reps}",
                     params=vars(args)):
        tracker.log({"top1": top1, "sec_per_image": elapsed / len(paths)}, step=0)
        tracker.artifact(cm_path)

    log.ok(f"Done → {out}/")


if __name__ == "__main__":
    main()
