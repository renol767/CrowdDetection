"""Density estimation pipeline using LWCC and pre-trained crowd density models."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from crowd.models import DENSITY_MATRIX, DENSITY_WEIGHTS_DIR

# --- LWCC local weights patch ---
DENSITY_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
import lwcc.util.functions as _lwcc_funcs


def _custom_weights_check(model_name: str, model_weights: str) -> str:
    """Check and return local weight file from crowd-eye/density/."""
    fn = f"{model_name}_{model_weights}.pth"
    out = DENSITY_WEIGHTS_DIR / fn
    if not out.exists():
        import gdown
        print(f"[CrowdDensity] Weight not found locally, downloading {fn} to {out}...")
        gdown.download(_lwcc_funcs.build_url(fn), str(out), quiet=False)
    return str(out)


# Apply patch to functions and model modules
_lwcc_funcs.weights_check = _custom_weights_check
for _mod in ("CSRNet", "DMCount", "Bay", "SFANet"):
    try:
        m = __import__(f"lwcc.models.{_mod}", fromlist=[_mod])
        if hasattr(m, "weights_check"):
            m.weights_check = _custom_weights_check
    except Exception:
        pass

from lwcc import LWCC

# Density model memory cache: (model_name, model_weights, device) -> model instance
_DENSITY_MODELS: Dict[Tuple[str, str, str], Any] = {}

_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_density_model(
    model_name: str = "DM-Count",
    model_weights: str = "QNRF",
    device: str = "cpu"
):
    """Get or lazily load density model instance into cache on specified device."""
    valid_weights = DENSITY_MATRIX.get(model_name, ["QNRF"])
    if model_weights not in valid_weights:
        model_weights = valid_weights[0]

    key = (model_name, model_weights, device)
    if key not in _DENSITY_MODELS or _DENSITY_MODELS[key] is None:
        model = LWCC.load_model(model_name=model_name, model_weights=model_weights)
        try:
            torch_dev = torch.device(device)
            model = model.to(torch_dev)
        except Exception as e:
            print(f"[CrowdDensity] Could not move model to {device}: {e}, using CPU")
        model.eval()
        _DENSITY_MODELS[key] = model

    return _DENSITY_MODELS[key], model_weights


def density(
    image: Union[str, Path, np.ndarray, Image.Image],
    weights: str = "QNRF",
    model_name: str = "DM-Count",
    full_res: bool = True,
    device: str = "cpu",
) -> Tuple[int, np.ndarray]:
    """Run crowd density estimation on an image with fast in-memory GPU path.
    
    Args:
        image: Path to image file, PIL Image, or numpy ndarray (BGR).
        weights: Dataset weights (e.g. 'QNRF', 'SHA', 'SHB').
        model_name: Density architecture ('DM-Count', 'CSRNet', 'Bay', 'SFANet').
        full_res: If True, do not downscale high-resolution images.
        device: 'mps', 'cuda:0', or 'cpu'.

    Returns:
        tuple: (estimated_count: int, density_map: np.ndarray)
    """
    model, validated_weights = get_density_model(model_name, weights, device=device)
    torch_dev = torch.device(device)

    # In-memory fast GPU path (avoids disk temporary file write/read)
    try:
        if isinstance(image, (str, Path)):
            pil_img = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            pil_img = Image.fromarray(rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # SFANet requires multiples of 16
        if model_name == "SFANet":
            w, h = pil_img.size
            nw = max(16, round(w / 16) * 16)
            nh = max(16, round(h / 16) * 16)
            pil_img = pil_img.resize((nw, nh), Image.BILINEAR)
        elif not full_res:
            long_side = max(pil_img.size[0], pil_img.size[1])
            if long_side > 1000:
                factor = 1000.0 / long_side
                pil_img = pil_img.resize(
                    (int(pil_img.size[0] * factor), int(pil_img.size[1] * factor)),
                    Image.BILINEAR
                )

        t_img = transforms.ToTensor()(pil_img)
        t_img = _NORMALIZE(t_img).unsqueeze(0).to(torch_dev)

        with torch.no_grad():
            output = model(t_img)
            count = float(torch.sum(output).item())
            dmap = output[0, 0].cpu().numpy().astype(np.float32)

        return max(0, round(count)), dmap

    except Exception as e:
        # Fallback to standard LWCC disk-based path if in-memory fails
        temp_path = None
        try:
            if isinstance(image, (str, Path)):
                path_str = str(image)
            else:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    temp_path = tf.name
                if isinstance(image, np.ndarray):
                    cv2.imwrite(temp_path, image)
                else:
                    image.save(temp_path, format="JPEG")
                path_str = temp_path

            c, d = LWCC.get_count(
                path_str,
                model_name=model_name,
                model_weights=validated_weights,
                model=model,
                return_density=True,
                resize_img=not full_res,
            )
            if c is None:
                return 0, np.zeros((1, 1), dtype=np.float32)
            return max(0, round(float(c))), np.asarray(d, dtype=np.float32)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
