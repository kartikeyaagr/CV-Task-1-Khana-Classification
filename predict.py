"""Offline prediction wrapper for the Khana food classifier.

The evaluator imports this file and calls:

    predict(image)

where ``image`` is a PIL RGB image. The function returns only the predicted
class name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
IMAGE_SIZE = 320
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_CANDIDATES = (
    BASE_DIR / "best_ema_hires.pt",
    BASE_DIR / "best_ema.pt",
    BASE_DIR / "models" / "320" / "best_ema_hires.pt",
    BASE_DIR / "models" / "best_ema.pt",
    BASE_DIR / "models" / "384" / "best_ema_hires.pt",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASSES = [
    "aloo gobi",
    "aloo methi",
    "aloo mutter",
    "aloo paratha",
    "amritsari kulcha",
    "anda curry",
    "balushahi",
    "banana chips",
    "besan laddu",
    "bhindi masala",
    "biryani",
    "boondi laddu",
    "chaas",
    "chana masala",
    "chapati",
    "chicken pizza",
    "chicken wings",
    "chikki",
    "chivda",
    "chole bhature",
    "dabeli",
    "dal khichdi",
    "dhokla",
    "falooda",
    "fish curry",
    "gajar ka halwa",
    "garlic bread",
    "garlic naan",
    "ghevar",
    "grilled sandwich",
    "gujhia",
    "gulab jamun",
    "hara bhara kabab",
    "idiyappam",
    "idli",
    "jalebi",
    "kaju katli",
    "khakhra",
    "kheer",
    "kulfi",
    "margherita pizza",
    "masala dosa",
    "masala papad",
    "medu vada",
    "misal pav",
    "modak",
    "moong dal halwa",
    "murukku",
    "mysore pak",
    "navratan korma",
    "neer dosa",
    "onion pakoda",
    "palak paneer",
    "paneer masala",
    "paneer pizza",
    "pani puri",
    "paniyaram",
    "papdi chaat",
    "patrode",
    "pav bhaji",
    "pepperoni pizza",
    "phirni",
    "poha",
    "pongal",
    "puri bhaji",
    "rajma chawal",
    "rasgulla",
    "rava dosa",
    "sabudana khichdi",
    "sabudana vada",
    "samosa",
    "seekh kebab",
    "set dosa",
    "sev puri",
    "solkadhi",
    "steamed momo",
    "thali",
    "thukpa",
    "uttapam",
    "vada pav",
]

_device: Optional[torch.device] = None
_model: Optional[torch.nn.Module] = None
_transform: Optional[torch.nn.Module] = None


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_checkpoint() -> Path:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(f"Checkpoint not found. Checked: {checked}")


def build_model() -> torch.nn.Module:
    """Build the exact ConvNeXt-Small architecture used for training."""
    return timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(CLASSES),
        drop_path_rate=0.0,
    )


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key.removeprefix("module.")
        cleaned[key] = value
    return cleaned


def _load_model_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidates = []

    if isinstance(checkpoint, dict):
        for key in ("ema", "model", "state_dict"):
            if checkpoint.get(key) is not None:
                candidates.append((key, checkpoint[key]))
    else:
        candidates.append(("checkpoint", checkpoint))

    errors = []
    for name, state_dict in candidates:
        try:
            model.load_state_dict(_clean_state_dict(state_dict))
            return
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")

    details = "\n".join(errors) if errors else "No model weights found."
    raise RuntimeError(f"Could not load checkpoint {checkpoint_path}:\n{details}")


def _get_transform() -> torch.nn.Module:
    resize = int(IMAGE_SIZE * 1.143)
    return v2.Compose(
        [
            v2.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _get_model() -> torch.nn.Module:
    global _device, _model, _transform

    if _model is None:
        checkpoint_path = _find_checkpoint()
        _device = _select_device()
        _model = build_model()
        _load_model_weights(_model, checkpoint_path)
        _model.to(_device)
        _model.eval()
        _transform = _get_transform()

    return _model


def predict(image: Image.Image) -> str:
    """Return the predicted class name for a single PIL RGB image."""
    model = _get_model()
    assert _device is not None
    assert _transform is not None

    if not isinstance(image, Image.Image):
        raise TypeError("predict(image) expects a PIL Image")

    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(_device)
    with torch.inference_mode():
        logits = model(tensor)

    class_index = int(logits.argmax(dim=1).item())
    return CLASSES[class_index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify one image.")
    parser.add_argument("image", help="Path to an image file")
    args = parser.parse_args()

    with Image.open(args.image) as img:
        print(predict(img))
