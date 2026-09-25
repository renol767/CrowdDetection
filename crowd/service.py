"""Unified Crowd Inference Service coordinating detection, density, and auto-routing."""

from pathlib import Path
from typing import Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

from crowd.density import density
from crowd.detector import detect, pick_device
from crowd.router import calculate_packedness, route
from crowd.schemas import InferenceResult, InferenceSettings

try:
    _TURBO_CMAP = matplotlib.colormaps["turbo"]
except AttributeError:
    from matplotlib import cm
    _TURBO_CMAP = cm.get_cmap("turbo")


class CrowdService:
    """Production-ready crowd inference service with cached models and auto-routing."""

    def __init__(self, default_settings: Optional[InferenceSettings] = None):
        self.default_settings = default_settings or InferenceSettings()

    def analyze(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        settings: Optional[InferenceSettings] = None,
    ) -> InferenceResult:
        """Run crowd inference according to configured mode ('auto', 'detection', 'density')."""
        s = settings or self.default_settings
        mode = s.mode.lower()
        device = pick_device(s.use_gpu)

        detections = []
        raw_preds = []
        det_count = None
        dens_count = None
        dmap = None
        packedness_score = 0.0

        run_det = mode in ("auto", "detection")
        run_dens = mode in ("auto", "density")

        # 1. Detection pipeline (YOLO + SAHI)
        if run_det:
            detections, raw_preds = detect(
                image=image,
                model_path=s.model,
                conf=s.conf,
                imgsz=s.imgsz,
                slice_px=s.slice,
                overlap=s.overlap,
                device=device,
            )
            det_count = len(detections)
            try:
                packedness_score = calculate_packedness(detections)
            except Exception:
                packedness_score = 0.0

        # 2. Density pipeline (LWCC)
        if run_dens:
            dens_count, dmap = density(
                image=image,
                weights=s.density_weights,
                model_name=s.density_model,
                full_res=s.full_res,
                device=device,
            )

        # 3. Auto-routing pipeline
        method, final_count, ratio = route(
            det_count=det_count,
            dens_count=dens_count,
            mode=mode,
            ratio_threshold=s.ratio,
        )

        ratio_val = None
        if ratio is not None:
            ratio_val = round(ratio, 2) if ratio != float("inf") else 999.0

        return InferenceResult(
            method=method,
            final_count=final_count,
            detection_count=det_count,
            density_count=dens_count,
            ratio=ratio_val,
            packedness=round(packedness_score, 3),
            model={
                "detection_model": s.model,
                "density_model": s.density_model,
                "density_weights": s.density_weights,
                "conf": s.conf,
                "imgsz": s.imgsz,
                "slice": s.slice,
                "overlap": s.overlap,
                "ratio_threshold": s.ratio,
            },
            detections=detections,
            density_map=dmap,
            device=device,
        )

    def render_overlay(
        self,
        image: Union[str, Path, np.ndarray],
        result: InferenceResult,
        style: str = "auto",
        heat_vmax: Optional[float] = 0.3,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Render visualization overlay (boxes or density heatmap) onto the image.
        
        Args:
            image: Original image as filepath or numpy BGR array.
            result: InferenceResult produced by analyze().
            style: 'auto' (picks method), 'detection', or 'density'.
            heat_vmax: Max density normalization value for heatmap.
            alpha: Transparency for heatmap blend.

        Returns:
            np.ndarray: BGR image with overlay drawn.
        """
        if isinstance(image, (str, Path)):
            base_bgr = cv2.imread(str(image))
        else:
            base_bgr = image.copy()

        render_style = result.method if style == "auto" else style

        if render_style == "density" and result.density_map is not None:
            d = np.asarray(result.density_map, dtype=np.float32)
            vmax = heat_vmax if heat_vmax is not None else (float(d.max()) if d.max() > 0 else 1.0)
            norm = np.clip(d / vmax, 0.0, 1.0) ** 0.6
            rgba = _TURBO_CMAP(norm)
            rgb_heat = (rgba[..., :3] * 255).astype(np.uint8)
            bgr_heat = cv2.cvtColor(rgb_heat, cv2.COLOR_RGB2BGR)
            bgr_heat_resized = cv2.resize(bgr_heat, (base_bgr.shape[1], base_bgr.shape[0]))
            return cv2.addWeighted(base_bgr, 1.0 - alpha, bgr_heat_resized, alpha, 0)

        # Default to detection boxes
        out = base_bgr.copy()
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 243, 255), 2)
            # Dot at center
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

        return out
