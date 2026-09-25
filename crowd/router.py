"""Routing logic and spatial packedness calculation based on Crowd Eye."""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from crowd.schemas import Detection


def calculate_packedness(preds: Union[List[Detection], List[Any]]) -> float:
    """Calculate crowd packedness metric.
    
    Measures how close detections are to each other relative to their median bounding box size.
    0.0 -> sparse (isolated people)
    0.7+ -> highly congested / overlapping crowd
    """
    if not preds or len(preds) < 5:
        return 0.0

    centers = []
    sizes = []

    for p in preds:
        if isinstance(p, Detection):
            minx, miny, maxx, maxy = p.bbox
        elif hasattr(p, "bbox"):
            minx, miny, maxx, maxy = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
        elif isinstance(p, (list, tuple)) and len(p) >= 4:
            minx, miny, maxx, maxy = p[0], p[1], p[2], p[3]
        else:
            continue

        centers.append([(minx + maxx) / 2.0, (miny + maxy) / 2.0])
        sizes.append(max(maxx - minx, maxy - miny))

    if len(centers) < 5:
        return 0.0

    centers_arr = np.array(centers, dtype=np.float32)
    sizes_arr = np.array(sizes, dtype=np.float32)

    med = float(np.median(sizes_arr)) or 1.0
    tree = cKDTree(centers_arr)
    distances, _ = tree.query(centers_arr, k=2)

    # Ratio of 1st nearest neighbor distance vs median box size
    return float(((distances[:, 1] / med) < 1.0).mean())


def route(
    det_count: Optional[int],
    dens_count: Optional[int],
    mode: str = "auto",
    ratio_threshold: float = 5.0,
) -> Tuple[str, int, Optional[float]]:
    """Determine winning method, final count, and count ratio.
    
    Crowd Eye Baseline Heuristic:
    - If mode == 'detection' -> detection
    - If mode == 'density' -> density
    - If mode == 'auto' ->
        ratio = density_count / det_count
        if ratio >= ratio_threshold (default 5.0) -> 'density' else 'detection'

    Returns:
        tuple: (method: str, final_count: int, ratio: Optional[float])
    """
    if mode == "detection":
        final_count = det_count if det_count is not None else 0
        return "detection", final_count, None

    if mode == "density":
        final_count = dens_count if dens_count is not None else 0
        return "density", final_count, None

    # Auto routing mode
    if det_count is not None and det_count > 0 and dens_count is not None:
        ratio = float(dens_count) / float(det_count)
    elif dens_count is not None and dens_count > 0:
        # If detector found 0 people but density found crowd, density wins
        ratio = float("inf")
    else:
        ratio = 0.0

    if ratio >= ratio_threshold:
        method = "density"
        final_count = dens_count if dens_count is not None else 0
    else:
        method = "detection"
        final_count = det_count if det_count is not None else 0

    return method, final_count, ratio
