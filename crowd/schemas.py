"""Crowd inference schemas and data models."""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        return max(0.0, self.maxx - self.minx)

    @property
    def height(self) -> float:
        return max(0.0, self.maxy - self.miny)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.minx + self.maxx) / 2.0, (self.miny + self.maxy) / 2.0)


class Detection(BaseModel):
    bbox: List[float]  # [minx, miny, maxx, maxy]
    confidence: float
    category_id: int = 0
    category_name: str = "person"


class InferenceSettings(BaseModel):
    model: str = "yakhyo.pt"
    conf: float = 0.1
    imgsz: int = 1280
    slice: int = 768
    overlap: float = 0.2
    mode: str = "auto"  # 'auto', 'detection', 'density'
    density_model: str = "DM-Count"
    density_weights: str = "QNRF"
    full_res: bool = True
    ratio: float = 5.0
    use_gpu: bool = True
    heat_vmax: Optional[float] = 0.3


class InferenceResult(BaseModel):
    method: str  # "detection" | "density"
    final_count: int
    detection_count: Optional[int] = None
    density_count: Optional[int] = None
    ratio: Optional[float] = None
    packedness: float = 0.0
    model: dict = Field(default_factory=dict)
    detections: List[Detection] = Field(default_factory=list)
    density_map: Optional[Any] = None
    device: str = "cpu"

    class Config:
        arbitrary_types_allowed = True
