"""Diagnostic training curve plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_training_curves(history: list[dict], out_dir: Path, prefix: str = "") -> None:
    """Generate and save 4 diagnostic plots from per-epoch training history.

    Detects and annotates: overfitting, data leakage, underfitting, LR issues.

    history entries must contain: train_loss, val_loss, ema_top1, val_top1, lr.
    """
    if not history:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]
    ema_top1    = [h["ema_top1"]   for h in history]
    val_top1    = [h["val_top1"]   for h in history]
    lr          = [h["lr"]         for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{prefix}Training Diagnostics", fontsize=14, fontweight="bold")

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train loss", color="steelblue", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val loss",   color="darkorange", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend(); ax.grid(alpha=0.3)

    # Annotate risks
    if len(epochs) >= 3:
        gap = val_loss[-1] - train_loss[-1]
        early_gap = val_loss[2] - train_loss[2] if len(epochs) > 2 else gap
        if gap > 0.3 and gap > early_gap * 1.5:
            ax.annotate("⚠ Overfitting", xy=(epochs[-1], val_loss[-1]),
                        xytext=(-40, 10), textcoords="offset points",
                        color="red", fontsize=9, fontweight="bold")
        if any(v < t for v, t in zip(val_loss, train_loss)):
            ax.annotate("⚠ Val < Train loss — check for data leak",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        color="red", fontsize=9, fontweight="bold", va="top")

    # ── 2. Accuracy curves ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, ema_top1, label="EMA top-1",  color="green",  linewidth=2)
    ax.plot(epochs, val_top1, label="Raw top-1",  color="royalblue", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curves"); ax.legend(); ax.grid(alpha=0.3)

    best_acc   = max(ema_top1)
    best_epoch = epochs[ema_top1.index(best_acc)]
    ax.axvline(best_epoch, color="green", linestyle=":", alpha=0.6)
    ax.annotate(f"Best: {best_acc:.2f}%\n(epoch {best_epoch})",
                xy=(best_epoch, best_acc), xytext=(8, -20),
                textcoords="offset points", fontsize=8, color="green")

    # ── 3. Generalisation gap ─────────────────────────────────────────────────
    ax = axes[1, 0]
    gap = [v - t for v, t in zip(val_loss, train_loss)]
    ax.plot(epochs, gap, color="purple", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(epochs, gap, 0,
                    where=[g > 0 for g in gap], alpha=0.15, color="red",
                    label="Overfit region")
    ax.fill_between(epochs, gap, 0,
                    where=[g <= 0 for g in gap], alpha=0.15, color="green",
                    label="Underfit / leak region")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss − Train loss")
    ax.set_title("Generalisation Gap"); ax.legend(); ax.grid(alpha=0.3)

    if len(gap) >= 3:
        trend = np.polyfit(epochs, gap, 1)[0]
        direction = "↑ widening" if trend > 0.005 else ("↓ closing" if trend < -0.005 else "→ stable")
        ax.annotate(f"Trend: {direction}", xy=(0.05, 0.05), xycoords="axes fraction", fontsize=9)

    # ── 4. LR schedule ────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, lr, color="crimson", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule"); ax.set_yscale("log"); ax.grid(alpha=0.3)

    if lr[0] < lr[min(4, len(lr) - 1)]:
        warmup_end = next((i for i in range(1, len(lr)) if lr[i] < lr[i - 1]), len(lr))
        ax.axvspan(epochs[0], epochs[min(warmup_end, len(epochs) - 1)],
                   alpha=0.1, color="orange", label="Warmup")
        ax.legend(fontsize=8)

    plt.tight_layout()
    tag = f"{prefix}_" if prefix else ""
    fig.savefig(out_dir / f"{tag}training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
