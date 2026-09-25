"""Crowd Eye modular inference architecture."""

from crowd.density import density, get_density_model
from crowd.detector import detect, get_detector
from crowd.models import DENSITY_MATRIX, InferenceMode
from crowd.router import calculate_packedness, route
from crowd.schemas import Detection, InferenceResult, InferenceSettings
from crowd.service import CrowdService
from crowd.p2pnet import P2PNet, load_p2pnet, predict_points as predict_p2p_points

__all__ = [
    "CrowdService",
    "InferenceSettings",
    "InferenceResult",
    "Detection",
    "InferenceMode",
    "DENSITY_MATRIX",
    "detect",
    "density",
    "route",
    "calculate_packedness",
    "get_detector",
    "get_density_model",
    "P2PNet",
    "load_p2pnet",
    "predict_p2p_points",
]
