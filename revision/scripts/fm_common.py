"""Shared helpers for the IEEE Access revision experiments (frozen foundation models).

The image transform and embedding definitions here reproduce the ones used by the
original training scripts exactly:
  scripts/train_uni2h_preprocessed_colab.py   (UNI2-h: CLS token, 1536-d)
  scripts/train_virchow_preprocessed_colab.py (Virchow2: CLS + mean patch tokens, 2560-d)
so that a linear head trained on cached embeddings is the same model as a head trained
end-to-end with the frozen encoder in the loop.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODELS: Dict[str, Dict] = {
    "uni2h": {"hf_repo": "MahmoodLab/UNI2-h", "embed_dim": 1536},
    "virchow2": {"hf_repo": "paige-ai/Virchow2", "embed_dim": 2560},
    # Small random-init model used only to test the pipeline without GPU/HF access.
    "debug_tiny": {"hf_repo": None, "embed_dim": 192},
}


def to_unit_range(x: np.ndarray) -> np.ndarray:
    """float32 HWC/NHWC in [0, 1]. Raw uint8 inputs are divided by 255 by dtype, not by value."""
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32, copy=False)


def transform_batch(x_nhwc: np.ndarray, resize: int = 224) -> torch.Tensor:
    """Same ops as the original per-sample Dataset: bicubic 96->224, clamp [0,1], ImageNet norm.

    Runs on CPU in float32; F.interpolate on a batch gives the same per-sample result as the
    original one-sample calls.
    """
    x = torch.from_numpy(np.ascontiguousarray(to_unit_range(x_nhwc))).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(resize, resize), mode="bicubic", align_corners=False)
    x = x.clamp(0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def load_backbone(name: str, device: torch.device):
    import timm

    if name == "uni2h":
        from timm.layers import SwiGLUPacked

        kwargs = dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
        )
        m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs)
    elif name == "virchow2":
        from timm.layers import SwiGLUPacked

        m = timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        )
    elif name == "debug_tiny":
        torch.manual_seed(0)
        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    else:
        raise ValueError(name)
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device).eval()


@torch.no_grad()
def embed(name: str, backbone, x: torch.Tensor) -> torch.Tensor:
    out = backbone(x)
    if name == "virchow2":
        # CLS + mean of patch tokens (tokens 1-4 are registers), as in the original script.
        return torch.cat([out[:, 0], out[:, 5:].mean(dim=1)], dim=-1)
    if out.dim() == 3:  # UNI2-h / debug: CLS token
        return out[:, 0]
    return out


def state_dict_fingerprint(sd: Dict[str, torch.Tensor], strip_prefix: str = "") -> str:
    """SHA-256 over sorted (key, float32 bytes). Identifies the exact weights used."""
    h = hashlib.sha256()
    keys = sorted(k for k in sd if k.startswith(strip_prefix))
    for k in keys:
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        h.update(k[len(strip_prefix):].encode())
        h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()
