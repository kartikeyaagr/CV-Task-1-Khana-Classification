#!/usr/bin/env python
"""One-time script: scan the khana/ dataset root and write a stratified split manifest.

Usage:
    uv run python scripts/prepare_splits.py
    uv run python scripts/prepare_splits.py --root ../khana --output ../splits/manifest.json --seed 42
"""

import argparse
import sys
from pathlib import Path
import _bootstrap  # noqa: F401 — adds src/ to sys.path

from khana.data.splits import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified train/val/test manifest")
    parser.add_argument("--root",   default="../khana",                    help="Path to khana/ dataset root")
    parser.add_argument("--output", default="../splits/manifest.json",     help="Output manifest JSON path")
    parser.add_argument("--val",    type=float, default=0.10,              help="Validation fraction")
    parser.add_argument("--test",   type=float, default=0.10,              help="Test fraction")
    parser.add_argument("--seed",   type=int,   default=42,                help="Random seed")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent  # khana_classifier/
    root   = (project_dir / args.root).resolve()
    output = (project_dir / args.output).resolve()

    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    build_manifest(root, output, val_ratio=args.val, test_ratio=args.test, seed=args.seed)


if __name__ == "__main__":
    main()
