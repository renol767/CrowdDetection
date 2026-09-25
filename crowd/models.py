"""Model constants and mappings."""

from enum import Enum
from pathlib import Path


class InferenceMode(str, Enum):
    AUTO = "auto"
    DETECTION = "detection"
    DENSITY = "density"


DENSITY_MATRIX = {
    "DM-Count": ["QNRF", "SHA", "SHB"],
    "CSRNet": ["SHA", "SHB"],
    "Bay": ["QNRF", "SHA", "SHB"],
    "SFANet": ["SHB"],
}

# Root directory of the project
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Default directories for models and weights
DENSITY_WEIGHTS_DIR = WORKSPACE_ROOT / "crowd-eye" / "density"
YOLO_WEIGHTS_DIR = WORKSPACE_ROOT / "crowd-eye" / "yolo"
