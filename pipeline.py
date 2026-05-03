"""Full training pipeline: Phase-1 → Phase-2 → Evaluate.

Run:
    python pipeline.py
    python pipeline.py --epochs 50 --finetune-epochs 15
    python pipeline.py --resume models/epoch_020.pt --epochs 50
    python pipeline.py --model convnext_base.fb_in22k_ft_in1k --batch-size 32
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n[pipeline] {label} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(description="Full Khana training pipeline")
    # Phase-1 args
    p.add_argument("--resume",           default=None)
    p.add_argument("--manifest",         default="splits/manifest.json")
    p.add_argument("--data-root",        default="khana")
    p.add_argument("--model",            default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--epochs",           type=int,   default=25)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=5e-4)
    p.add_argument("--out-dir",          default="models")
    # Phase-2 args
    p.add_argument("--finetune-epochs",  type=int,   default=8)
    p.add_argument("--finetune-batch",   type=int,   default=24)
    p.add_argument("--finetune-lr",      type=float, default=1e-4)
    p.add_argument("--finetune-size",    type=int,   default=320)
    p.add_argument("--num-workers",      type=int,   default=2)
    # Evaluate args
    p.add_argument("--tta",              action="store_true")
    p.add_argument("--eval-split",       default="test", choices=["val", "test"])
    # Skip flags
    p.add_argument("--skip-train",       action="store_true", help="Skip phase-1, use existing best_ema.pt")
    p.add_argument("--skip-finetune",    action="store_true", help="Skip phase-2, evaluate best_ema.pt directly")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = args.out_dir
    best_p1 = str(Path(out_dir) / "best_ema.pt")
    best_p2 = str(Path(out_dir) / "best_ema_hires.pt")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if not args.skip_train:
        cmd = [
            "train.py",
            "--manifest",   args.manifest,
            "--data-root",  args.data_root,
            "--model",      args.model,
            "--epochs",     str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr",         str(args.lr),
            "--out-dir",    out_dir,
        ]
        if args.resume:
            cmd += ["--resume", args.resume]
        run(cmd, f"Phase 1 — {args.model} @ 224px for {args.epochs} epochs")
    else:
        print(f"[pipeline] Skipping phase-1, using {best_p1}")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if not args.skip_finetune:
        cmd = [
            "finetune_hires.py",
            "--checkpoint", best_p1,
            "--manifest",   args.manifest,
            "--data-root",  args.data_root,
            "--model",      args.model,
            "--epochs",     str(args.finetune_epochs),
            "--batch-size", str(args.finetune_batch),
            "--lr",         str(args.finetune_lr),
            "--image-size", str(args.finetune_size),
            "--out-dir",    out_dir,
            "--num-workers", str(args.num_workers),
        ]
        run(cmd, f"Phase 2 — hires finetune @ {args.finetune_size}px for {args.finetune_epochs} epochs")
        eval_ckpt = best_p2
    else:
        print(f"[pipeline] Skipping phase-2, evaluating {best_p1}")
        eval_ckpt = best_p1

    # ── Evaluate ──────────────────────────────────────────────────────────────
    cmd = [
        "evaluate.py",
        "--checkpoint", eval_ckpt,
        "--manifest",   args.manifest,
        "--data-root",  args.data_root,
        "--model",      args.model,
        "--split",      args.eval_split,
        "--image-size", str(args.finetune_size),
    ]
    if args.tta:
        cmd.append("--tta")
    run(cmd, f"Evaluate on {args.eval_split} split{' + TTA' if args.tta else ''}")

    print(f"\n[pipeline] All done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
