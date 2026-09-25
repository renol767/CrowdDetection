"""YOLO + SAHI sliced detection pipeline with persistent model caching."""

import os
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional

import numpy as np
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from crowd.models import WORKSPACE_ROOT, YOLO_WEIGHTS_DIR
from crowd.schemas import Detection

PERSON_CLASS = 0
EXCLUDE = [i for i in range(80) if i != PERSON_CLASS]

# Global cache for detector model instance: (model_path, conf, imgsz, device)
_DET_MODEL = {"key": None, "model": None}


def pick_device(use_gpu: bool = True) -> str:
    """Select appropriate inference device: CUDA -> MPS (Apple Silicon) -> CPU."""
    if use_gpu:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def resolve_yolo_path(name: str) -> str:
    """Resolve YOLO weights path from workspace root, crowd-eye/yolo, or absolute."""
    path_obj = Path(name)
    if path_obj.is_absolute() and path_obj.exists():
        return str(path_obj)

    # Check in workspace root
    ws_candidate = WORKSPACE_ROOT / name
    if ws_candidate.exists():
        return str(ws_candidate)

    # Check in crowd-eye/yolo/
    eye_candidate = YOLO_WEIGHTS_DIR / name
    if eye_candidate.exists():
        return str(eye_candidate)

    # Fallback to provided name
    return name


def get_detector(
    model_path: str = "yakhyo.pt",
    conf: float = 0.1,
    imgsz: int = 1280,
    device: str = "cpu",
) -> AutoDetectionModel:
    """Get or lazily load YOLO model wrapped in SAHI AutoDetectionModel with persistent cache."""
    resolved_path = resolve_yolo_path(model_path)
    key = (resolved_path, conf, imgsz, device)

    if _DET_MODEL["key"] != key or _DET_MODEL["model"] is None:
        # Prevent Ultralytics from downloading into random directories
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_WEIGHTS_DIR))
        _DET_MODEL["model"] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=resolved_path,
            confidence_threshold=conf,
            device=device,
            image_size=imgsz,
        )
        _DET_MODEL["key"] = key

    return _DET_MODEL["model"]


def detect(
    image: Union[str, Path, np.ndarray, Image.Image],
    model_path: str = "yakhyo.pt",
    conf: float = 0.1,
    imgsz: int = 1280,
    slice_px: int = 768,
    overlap: float = 0.2,
    device: str = "cpu",
) -> Tuple[List[Detection], List[Any]]:
    """Run sliced inference on an image using SAHI and YOLO.
    
    Returns:
        tuple: (list of Detection schema objects, raw SAHI prediction objects)
    """
    model = get_detector(model_path, conf, imgsz, device)

    # Convert image format if needed
    if isinstance(image, (str, Path)):
        img_input = str(image)
    elif isinstance(image, np.ndarray):
        # OpenCV BGR -> RGB for SAHI
        if image.ndim == 3 and image.shape[2] == 3:
            img_input = image[:, :, ::-1]
        else:
            img_input = image
    elif isinstance(image, Image.Image):
        img_input = np.array(image.convert("RGB"))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    res = get_sliced_prediction(
        img_input,
        model,
        slice_height=slice_px,
        slice_width=slice_px,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        exclude_classes_by_id=EXCLUDE,
        verbose=0,
    )

    raw_preds = res.object_prediction_list
    detections: List[Detection] = []

    for p in raw_preds:
        b = p.bbox
        score = float(p.score.value) if hasattr(p, "score") and hasattr(p.score, "value") else float(p.score)
        cat_id = int(p.category.id) if hasattr(p, "category") and hasattr(p.category, "id") else 0
        cat_name = str(p.category.name) if hasattr(p, "category") and hasattr(p.category, "name") else "person"

        detections.append(
            Detection(
                bbox=[float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)],
                confidence=round(score, 4),
                category_id=cat_id,
                category_name=cat_name,
            )
        )

    return detections, raw_preds
