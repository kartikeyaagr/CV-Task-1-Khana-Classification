"""Full training pipeline: Phase-1 → progressive hires finetune stages → Evaluate.

Examples:
    # Default: 224 → 320 → evaluate
    python pipeline.py

    # Progressive: 224 → 320 → 384 → evaluate
    python pipeline.py --finetune-sizes 320 384

    # Three stages: 224 → 320 → 384 → 448 → evaluate
    python pipeline.py --finetune-sizes 320 384 448

    # Resume phase-1 from checkpoint, then run all finetune stages
    python pipeline.py --resume models/epoch_020.pt --epochs 50 --finetune-sizes 320 384

    # Skip phase-1 (already done), run finetune stages from existing best_ema.pt
    python pipeline.py --skip-train --finetune-sizes 320 384

    # Skip straight to a specific stage using an existing checkpoint
    python pipeline.py --skip-train --skip-sizes 320 --finetune-sizes 320 384
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


def auto_batch(base_batch: int, base_size: int, target_size: int) -> int:
    """Scale batch size inversely with image area, clamped to a minimum of 8."""
    scaled = int(base_batch * (base_size / target_size) ** 2)
    return max(8, scaled)


def parse_args():
    p = argparse.ArgumentParser(description="Full Khana training pipeline")

    # Phase-1
    p.add_argument("--resume",          default=None,   help="Checkpoint to resume phase-1 from")
    p.add_argument("--manifest",        default="splits/manifest.json")
    p.add_argument("--data-root",       default="khana")
    p.add_argument("--model",           default="convnext_small.fb_in22k_ft_in1k")
    p.add_argument("--epochs",          type=int,   default=25)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--out-dir",         default="models")

    # Finetune stages
    p.add_argument("--finetune-sizes",  type=int, nargs="+", default=[320],
                   metavar="SIZE",
                   help="Progressive finetune resolutions in order (e.g. 320 384 448)")
    p.add_argument("--finetune-epochs", type=int,   default=8,
                   help="Epochs per finetune stage")
    p.add_argument("--finetune-lr",     type=float, default=1e-4,
                   help="Starting LR for each finetune stage (halved per stage)")
    p.add_argument("--finetune-batch",  type=int,   default=24,
                   help="Batch size for first finetune stage; auto-scaled for larger sizes")
    p.add_argument("--num-workers",     type=int,   default=2)

    # Evaluate
    p.add_argument("--tta",             action="store_true")
    p.add_argument("--eval-split",      default="test", choices=["val", "test"])

    # Skip flags
    p.add_argument("--skip-train",      action="store_true",
                   help="Skip phase-1; start from existing best_ema.pt")
    p.add_argument("--skip-sizes",      type=int, nargs="+", default=[],
                   metavar="SIZE",
                   help="Skip finetune stages at these sizes (already completed)")

    return p.parse_args()


def stage_dir(out_dir: str, size: int) -> str:
    return str(Path(out_dir) / str(size))


def main():
    args    = parse_args()
    out_dir = args.out_dir
    best_p1 = str(Path(out_dir) / "best_ema.pt")

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

    # ── Progressive finetune stages ───────────────────────────────────────────
    prev_ckpt = best_p1
    prev_size = 224
    lr        = args.finetune_lr

    for size in args.finetune_sizes:
        out   = stage_dir(out_dir, size)
        ckpt  = str(Path(out) / "best_ema_hires.pt")
        batch = auto_batch(args.finetune_batch, args.finetune_sizes[0], size)

        if size in args.skip_sizes:
            print(f"[pipeline] Skipping {size}px stage, using {ckpt}")
        else:
            cmd = [
                "finetune_hires.py",
                "--checkpoint", prev_ckpt,
                "--manifest",   args.manifest,
                "--data-root",  args.data_root,
                "--model",      args.model,
                "--epochs",     str(args.finetune_epochs),
                "--batch-size", str(batch),
                "--lr",         str(lr),
                "--image-size", str(size),
                "--out-dir",    out,
                "--num-workers", str(args.num_workers),
            ]
            run(cmd, f"Finetune stage {prev_size}px → {size}px  "
                     f"(batch {batch}, lr {lr:.0e}, {args.finetune_epochs} epochs)")

        prev_ckpt = ckpt
        prev_size = size
        lr        = lr / 2   # halve LR each stage

    # ── Evaluate final checkpoint ─────────────────────────────────────────────
    final_size = args.finetune_sizes[-1]
    cmd = [
        "evaluate.py",
        "--checkpoint", prev_ckpt,
        "--manifest",   args.manifest,
        "--data-root",  args.data_root,
        "--model",      args.model,
        "--split",      args.eval_split,
        "--image-size", str(final_size),
    ]
    if args.tta:
        cmd.append("--tta")
    run(cmd, f"Evaluate @ {final_size}px on {args.eval_split}{' + TTA' if args.tta else ''}")

    print(f"\n[pipeline] All done. Results in {out_dir}/")


if __name__ == "__main__":
    main()
