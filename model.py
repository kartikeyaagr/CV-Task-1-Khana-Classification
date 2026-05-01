"""Model factory."""

import timm
import torch.nn as nn


def build_model(name="convnext_small.fb_in22k_ft_in1k", num_classes=80,
                drop_path_rate=0.1, pretrained=True) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained,
                              num_classes=num_classes, drop_path_rate=drop_path_rate)


def freeze_stem(model):
    stem = getattr(model, "stem", None)
    if stem:
        for p in stem.parameters():
            p.requires_grad = False


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
