"""Training and evaluation loops."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler


def accuracy(logits, targets, k=1):
    _, pred = logits.topk(k, dim=1)
    return pred.t().eq(targets.unsqueeze(0)).any(0).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    epoch, amp_dtype, grad_accum, clip_norm, ema, scaler):
    model.train()
    total_loss, n = 0.0, len(loader)
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        last    = (step + 1) % grad_accum == 0 or step + 1 == n

        with amp_ctx:
            loss = criterion(model(images), targets) / grad_accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if last:
            if scaler:
                if clip_norm: scaler.unscale_(optimizer)
                if clip_norm: nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer); scaler.update()
            else:
                if clip_norm: nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step_update(epoch * n + step + 1)
            if ema: ema.update(model)

        total_loss += loss.item() * grad_accum

    return total_loss / n


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_dtype):
    model.eval()
    loss_sum, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        bs = images.size(0)
        with amp_ctx:
            logits = model(images)
        loss_sum += criterion(logits, targets).item() * bs
        top1_sum += accuracy(logits, targets, k=1) * bs
        top5_sum += accuracy(logits, targets, k=5) * bs
        n += bs

    return {"loss": loss_sum / n, "top1": top1_sum / n * 100, "top5": top5_sum / n * 100}


@torch.inference_mode()
def evaluate_tta(model, loader, device, amp_dtype):
    """5-crop TTA: center + 4 corners, average softmax."""
    model.eval()
    top1_sum, top5_sum, n = 0.0, 0.0, 0
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype else nullcontext()

    for images, targets in loader:
        bs, c, h, w = images.shape
        cs = min(h, w)
        cx, cy = (w - cs) // 2, (h - cs) // 2
        crops = torch.stack([
            images[:, :, cy:cy+cs, cx:cx+cs],
            images[:, :,  0:cs,     0:cs],
            images[:, :,  0:cs,  w-cs:w],
            images[:, :, h-cs:h,    0:cs],
            images[:, :, h-cs:h, w-cs:w],
        ], dim=1).view(bs * 5, c, cs, cs).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_ctx:
            probs = F.softmax(model(crops), dim=1).view(bs, 5, -1).mean(1)

        top1_sum += accuracy(probs, targets, k=1) * bs
        top5_sum += accuracy(probs, targets, k=5) * bs
        n += bs

    return {"tta_top1": top1_sum / n * 100, "tta_top5": top5_sum / n * 100}
