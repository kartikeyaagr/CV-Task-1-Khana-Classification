"""Zero-shot diffusion classifier (Li et al., ICCV 2023).

For each image and each class, computes the noise-prediction MSE under
Stable Diffusion 2.1 conditioned on a text prompt. Predicts argmin.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TEMPLATES = [
    "a photo of {dish}, indian food",
    "a photo of {dish}, indian cuisine, professional food photography",
    "a high quality photo of {dish}",
]


def build_prompts(class_names: list[str], template_idx=1) -> list[str]:
    t = TEMPLATES[template_idx]
    return [t.format(dish=name) for name in class_names]


class DiffusionClassifier:
    """Stable Diffusion 2.1 zero-shot classifier via noise-prediction MSE."""

    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device=None,
                 timesteps=10, noise_reps=2, dtype=torch.float16):
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = timesteps
        self.noise_reps = noise_reps
        self.dtype     = dtype
        self._load(model_id)

    def _load(self, model_id):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, safety_checker=None, requires_safety_checker=False
        ).to(self.device)
        pipe.enable_attention_slicing()
        self.vae, self.unet = pipe.vae, pipe.unet
        self.text_encoder, self.tokenizer = pipe.text_encoder, pipe.tokenizer
        self.scheduler = pipe.scheduler
        T = self.scheduler.config.num_train_timesteps
        step = T // self.timesteps
        self._ts = torch.arange(step // 2, T, step)[:self.timesteps]
        for m in (self.vae, self.unet, self.text_encoder):
            m.eval(); m.requires_grad_(False)

    @torch.inference_mode()
    def encode_prompts(self, prompts):
        toks = self.tokenizer(prompts, padding="max_length",
                              max_length=self.tokenizer.model_max_length,
                              truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(**toks).last_hidden_state

    @torch.inference_mode()
    def classify(self, image: torch.Tensor, text_embeddings: torch.Tensor) -> int:
        """image: (1,3,H,W) float32. Returns predicted class index."""
        image = image.to(self.device, dtype=self.dtype)
        latent = self.vae.encode(image).latent_dist.mode() * self.vae.config.scaling_factor
        _, c, h, w = latent.shape
        C = text_embeddings.size(0)
        scores = torch.zeros(C, device=self.device)

        for t in self._ts.to(self.device):
            for _ in range(self.noise_reps):
                noise = torch.randn(1, c, h, w, dtype=self.dtype, device=self.device)
                x_t   = self.scheduler.add_noise(latent.to(self.dtype), noise, t.unsqueeze(0))
                pred  = self.unet(x_t.expand(C,-1,-1,-1), t.expand(C),
                                  encoder_hidden_states=text_embeddings.to(self.dtype)).sample
                scores += F.mse_loss(pred, noise.expand(C,-1,-1,-1).to(self.dtype),
                                     reduction="none").mean(dim=(1,2,3))
        return int(scores.argmin())
