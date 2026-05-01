"""Seed, logger, and checkpoint utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self):
        self.con = Console()

    def info(self, msg):    self.con.print(f"[cyan]INFO[/cyan]  {msg}")
    def ok(self, msg):      self.con.print(f"[green]OK[/green]    {msg}")
    def warn(self, msg):    self.con.print(f"[yellow]WARN[/yellow] {msg}")

    def metrics(self, d: dict, epoch: int):
        parts = "  ".join(f"[bold]{k}[/bold]={v:.4f}" for k, v in d.items())
        self.con.print(f"  epoch {epoch:03d}  {parts}")

    def tqdm(self, iterable, desc=""):
        with Progress(SpinnerColumn(), TextColumn(f"[progress.description]{desc}"),
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                      console=self.con, transient=True) as p:
            task = p.add_task(desc, total=len(iterable) if hasattr(iterable, "__len__") else None)
            for item in iterable:
                yield item
                p.advance(task)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, ema, optimizer, scheduler, metrics, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "ema":   ema.state_dict() if ema else None,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "metrics": metrics, "config": config,
    }, path)


def load_checkpoint(path, model, ema=None, optimizer=None, scheduler=None,
                    device=torch.device("cpu")):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema and ckpt.get("ema"):      ema.load_state_dict(ckpt["ema"])
    if optimizer and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
