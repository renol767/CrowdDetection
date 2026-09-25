"""P2PNet implementation for point-based crowd counting and localization.

Based on Tencent's P2PNet and HoseinRanjbar/Crowd-Counting-and-Localization:
"Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework" (ICCV 2021)
"""

import os
import threading
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as standard_transforms
import torchvision.models as tv_models
import cv2


def generate_anchor_points(stride: int = 8, row: int = 2, line: int = 2) -> np.ndarray:
    """Generates anchor points for a stride window."""
    row_step = stride / row
    line_step = stride / line
    x, y = np.meshgrid(
        np.arange(row) * row_step + row_step / 2,
        np.arange(line) * line_step + line_step / 2
    )
    return np.vstack((x.ravel(), y.ravel())).transpose()


def shift(shape: Tuple[int, int], stride: int, anchor_points: np.ndarray) -> np.ndarray:
    """Shifts base anchor points across the feature map spatial dimensions."""
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    a = anchor_points.shape[0]
    k = shifts.shape[0]
    all_anchor_points = (
        anchor_points.reshape((1, a, 2)) + shifts.reshape((1, k, 2)).transpose((1, 0, 2))
    )
    return all_anchor_points.reshape((k * a, 2))


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels: Optional[List[int]] = None, strides: Optional[List[int]] = None, row: int = 2, line: int = 2):
        super().__init__()
        self.pyramid_levels = [3,] if pyramid_levels is None else pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels] if strides is None else strides
        self.row = row
        self.line = line

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2), dtype=np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            base_anchors = generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted = shift(image_shapes[idx], self.strides[idx], base_anchors)
            all_anchor_points = np.append(all_anchor_points, shifted, axis=0)
        return torch.from_numpy(np.expand_dims(all_anchor_points, axis=0).astype(np.float32)).to(image.device)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in: int = 256, num_anchor_points: int = 4, feature_size: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in: int = 256, num_anchor_points: int = 4, num_classes: int = 2, feature_size: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output(out)
        out1 = out.permute(0, 2, 3, 1)
        b, w, h, _ = out1.shape
        out2 = out1.view(b, w, h, self.num_anchor_points, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class BackboneVGG16BN(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16_bn(weights=None)
        features = list(vgg.features.children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)
        x4 = self.body4(x3)
        return [x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, C3_size: int = 256, C4_size: int = 512, C5_size: int = 512, feature_size: int = 256):
        super().__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        c3, c4, c5 = inputs
        p5_x = self.P5_1(c5)
        p5_upsampled_x = self.P5_upsampled(p5_x)
        p5_x = self.P5_2(p5_x)

        p4_x = self.P4_1(c4)
        p4_x = p5_upsampled_x + p4_x
        p4_upsampled_x = self.P4_upsampled(p4_x)
        p4_x = self.P4_2(p4_x)

        p3_x = self.P3_1(c3)
        p3_x = p3_x + p4_upsampled_x
        p3_x = self.P3_2(p3_x)
        return [p3_x, p4_x, p5_x]


class P2PNet(nn.Module):
    def __init__(self, row: int = 2, line: int = 2):
        super().__init__()
        self.backbone = BackboneVGG16BN()
        num_anchor_points = row * line
        self.regression = RegressionModel(256, num_anchor_points)
        self.classification = ClassificationModel(256, num_anchor_points, 2)
        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=row, line=line)
        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(samples)
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]
        regression = self.regression(features_fpn[1]) * 100
        classification = self.classification(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = regression + anchor_points
        output_class = classification
        return {'pred_logits': output_class, 'pred_points': output_coord}


# Global model cache: key -> (P2PNet, device)
_P2P_CACHE: Dict[str, Any] = {}
_P2P_LOCK = threading.Lock()

_DEFAULT_TRANSFORM = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_default_device() -> torch.device:
    """Chooses CUDA for NVIDIA RTX, MPS for Apple Silicon, or CPU fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_p2pnet(
    weights_path: str = "crowd-eye/yolo/p2pcrowdcounting.pth",
    device: Optional[str] = None
) -> P2PNet:
    """Loads and caches P2PNet with specified checkpoint weights."""
    target_device = torch.device(device) if device else get_default_device()
    cache_key = f"{weights_path}_{target_device}"

    if cache_key in _P2P_CACHE:
        return _P2P_CACHE[cache_key]

    if not os.path.exists(weights_path):
        # Check alternative common locations
        candidates = [
            weights_path,
            f"crowd-eye/yolo/{os.path.basename(weights_path)}",
            f"models/{os.path.basename(weights_path)}",
            f"yolo/{os.path.basename(weights_path)}",
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if not found:
            raise FileNotFoundError(f"P2PNet weight file not found at {weights_path}")
        weights_path = found

    print(f"[P2PNet] Initializing architecture on {target_device}...")
    model = P2PNet().to(target_device)
    print(f"[P2PNet] Loading checkpoint from {weights_path}...")
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    _P2P_CACHE[cache_key] = model
    print(f"[P2PNet] Model successfully loaded and cached ({len(state_dict)} weights).")
    return model


def sync_device():
    """Synchronizes GPU / MPS queue to ensure complete thread safety."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def predict_points(
    frame: np.ndarray,
    threshold: float = 0.5,
    weights_path: str = "crowd-eye/yolo/p2pcrowdcounting.pth",
    target_max_size: int = 1280,
    device: Optional[str] = None
) -> List[Tuple[int, int, float]]:
    """Runs point localization on an image frame and returns list of (x, y, confidence).

    Args:
        frame: BGR numpy array from OpenCV.
        threshold: Score threshold for point detection (default 0.5).
        weights_path: Path to p2pcrowdcounting.pth.
        target_max_size: Maximum dimension to keep inference fast and accurate.
        device: 'mps', 'cuda', or 'cpu'.

    Returns:
        List of tuples: [(x, y, score), ...] mapped to original frame resolution.
    """
    orig_h, orig_w = frame.shape[:2]
    target_device = torch.device(device) if device else get_default_device()

    # Scale dimensions to reasonable size and round to multiples of 128
    scale = min(1.0, float(target_max_size) / max(orig_w, orig_h))
    new_w = max(128, int(orig_w * scale))
    new_h = max(128, int(orig_h * scale))
    nw = (new_w // 128) * 128
    nh = (new_h // 128) * 128

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    tensor_img = _DEFAULT_TRANSFORM(pil_img).unsqueeze(0).to(target_device)

    model = load_p2pnet(weights_path=weights_path, device=str(target_device))

    with _P2P_LOCK:
        with torch.no_grad():
            outputs = model(tensor_img)
            # Softmax probability for head foreground (class index 1)
            scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
            pred_points = outputs["pred_points"][0]
            mask = scores > threshold
            if not mask.any():
                sync_device()
                return []
            valid_pts = pred_points[mask].cpu().numpy()
            valid_scores = scores[mask].cpu().numpy()
            sync_device()

    # Scale coordinates back to original frame dimensions
    scale_x = orig_w / float(nw)
    scale_y = orig_h / float(nh)

    results = []
    for pt, sc in zip(valid_pts, valid_scores):
        px = int(round(pt[0] * scale_x))
        py = int(round(pt[1] * scale_y))
        px = max(0, min(orig_w - 1, px))
        py = max(0, min(orig_h - 1, py))
        results.append((px, py, float(sc)))

    return results
