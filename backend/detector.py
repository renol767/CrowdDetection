import cv2
import numpy as np
import time
import uuid
import threading
from typing import Optional, Dict, Any
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")

try:
    _TURBO_CMAP = matplotlib.colormaps["turbo"]
except AttributeError:
    from matplotlib import cm
    _TURBO_CMAP = cm.get_cmap("turbo")

from crowd import CrowdService, InferenceSettings
from crowd.p2pnet import predict_points as predict_p2p_points, load_p2pnet, sync_device


class CrowdDetector:
    def __init__(
        self,
        model_name: str = "aminesam.pt",
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.75,
        imgsz: int = 1280,
        engine: str = "yolo"
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model_name = model_name
        self.engine = engine  # 'yolo' or 'p2pnet'
        self.p2p_weights = "crowd-eye/yolo/p2pcrowdcounting.pth"

        # Hardware acceleration auto-detection: CUDA (NVIDIA RTX) -> MPS (Apple Silicon) -> CPU
        import torch
        if torch.cuda.is_available():
            self.device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            self.device_name = f"NVIDIA {gpu_name} (CUDA)"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.device_name = "Apple Silicon GPU (MPS)"
        else:
            self.device = "cpu"
            self.device_name = "CPU"
        print(f"[Hardware] Compute device auto-selected: {self.device} ({self.device_name})")

        print(f"Loading detector: {model_name} (engine={self.engine}, imgsz={self.imgsz}, conf={self.conf_threshold}, device={self.device})...")
        if "p2p" in model_name.lower() or model_name.endswith(".pth"):
            self.engine = "p2pnet"
            self.p2p_weights = model_name
            self.model = None
            self.is_head_model = True
            load_p2pnet(self.p2p_weights, device=self.device)
        else:
            self.engine = "yolo"
            self.model = YOLO(model_name)
            self.is_head_model = any("head" in str(name).lower() for name in self.model.names.values())
            print(f"Model loaded: {model_name}, is_head_model={self.is_head_model}, classes={self.model.names}")
        
        self.show_boxes = True
        self.show_points = True  # Toggle point dots for heads
        self.show_zones = True
        self.show_heatmap = False
        self.threshold_alert = 100
        self.max_det = 2500  # Expand max detection limit from default 300 to 2500 for massive crowds

        # Dynamic zones dictionary (start empty so user creates them manually)
        self.zones = {}

        # Colors (BGR)
        self.colors = {
            "box": (0, 255, 136),        # Neon Green
            "box_default": (0, 255, 136),
            "dot": (0, 243, 255),        # Cyan Point Dot
            "text": (255, 255, 255)
        }

        # --- CROWD EYE DUAL-ENGINE INTEGRATION ---
        self.crowd_service = CrowdService()
        self.crowd_mode = "auto"  # 'auto', 'detection', 'density'
        self.density_model = "DM-Count"
        self.density_weights = "QNRF"
        self.ratio_threshold = 5.0

        self.latest_crowd_eye: Dict[str, Any] = {
            "method": "detection",
            "final_count": 0,
            "detection_count": 0,
            "density_count": 0,
            "p2p_count": 0,
            "ratio": 1.0,
            "packedness": 0.0,
            "density_map": None,
            "heatmap_bgr": None,
            "last_scan_ms": 0,
        }

        self._current_raw_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._worker_running = True
        self._trigger_scan_event = threading.Event()

        # Launch async background worker for continuous Crowd Eye deep analysis
        self._bg_thread = threading.Thread(target=self._crowd_eye_loop, daemon=True)
        self._bg_thread.start()

    def _crowd_eye_loop(self):
        """Asynchronous worker that performs Crowd Eye density & routing analysis without blocking video feed."""
        import torch
        while self._worker_running:
            try:
                # Wait for next periodic interval or immediate trigger
                self._trigger_scan_event.wait(timeout=2.0)
                self._trigger_scan_event.clear()

                frame_copy = None
                with self._frame_lock:
                    if self._current_raw_frame is not None:
                        frame_copy = self._current_raw_frame.copy()

                if frame_copy is not None:
                    t_start = time.time()
                    settings = InferenceSettings(
                        model=self.model_name,
                        conf=self.conf_threshold,
                        imgsz=self.imgsz,
                        slice=768,
                        overlap=0.2,
                        mode=self.crowd_mode,
                        density_model=self.density_model,
                        density_weights=self.density_weights,
                        ratio=self.ratio_threshold,
                        use_gpu=True,
                    )
                    
                    with self._inference_lock:
                        res = self.crowd_service.analyze(frame_copy, settings)
                        p2p_cnt = 0
                        try:
                            p2p_pts = predict_p2p_points(
                                frame_copy,
                                threshold=0.5,
                                weights_path=self.p2p_weights,
                                target_max_size=1024,
                                device=self.device
                            )
                            p2p_cnt = len(p2p_pts)
                        except Exception as p_err:
                            pass

                        # Ensure all compute kernels finish before releasing lock
                        sync_device()

                        # Free GPU / MPS cached tensors to keep VRAM / unified memory lean
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()

                    scan_ms = round((time.time() - t_start) * 1000, 1)

                    # Precompute heatmap texture for fast frame overlay blending
                    heatmap_bgr = None
                    if res.density_map is not None:
                        d = np.asarray(res.density_map, dtype=np.float32)
                        vmax = float(d.max()) if d.max() > 0 else 1.0
                        norm = np.clip(d / vmax, 0.0, 1.0) ** 0.6
                        rgba = _TURBO_CMAP(norm)
                        rgb_heat = (rgba[..., :3] * 255).astype(np.uint8)
                        heatmap_bgr = cv2.cvtColor(rgb_heat, cv2.COLOR_RGB2BGR)

                    self.latest_crowd_eye = {
                        "method": res.method,
                        "final_count": res.final_count,
                        "detection_count": res.detection_count or 0,
                        "density_count": res.density_count or 0,
                        "p2p_count": p2p_cnt,
                        "ratio": res.ratio if res.ratio is not None else 1.0,
                        "packedness": res.packedness,
                        "density_map": res.density_map,
                        "heatmap_bgr": heatmap_bgr,
                        "last_scan_ms": scan_ms,
                    }
            except Exception as e:
                print(f"[CrowdEye Worker] Analysis error: {e}")
                time.sleep(1.0)

    def trigger_deep_scan(self):
        """Signals the background worker to execute a deep scan immediately."""
        self._trigger_scan_event.set()

    def _hex_to_bgr(self, hex_str: str) -> tuple[int, int, int]:
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (b, g, r)
        return (255, 51, 75)

    def _init_default_zones(self):
        """Initializes default reference zones (Zona B, Zona A, Zona C)."""
        self.zones = {
            "zone_b": {
                "id": "zone_b",
                "name": "ZONA B",
                "points": [[0.05, 0.40], [0.36, 0.42], [0.33, 0.85], [0.03, 0.78]],
                "color": (216, 180, 0), # Blue/Cyan (BGR)
                "hex_color": "#00b4d8"
            },
            "zone_a": {
                "id": "zone_a",
                "name": "ZONA A",
                "points": [[0.34, 0.42], [0.66, 0.42], [0.70, 0.88], [0.30, 0.85]],
                "color": (75, 51, 255), # Red (BGR)
                "hex_color": "#ff334b"
            },
            "zone_c": {
                "id": "zone_c",
                "name": "ZONA C",
                "points": [[0.64, 0.42], [0.95, 0.44], [0.97, 0.78], [0.68, 0.88]],
                "color": (30, 169, 224), # Yellow/Gold (BGR)
                "hex_color": "#e0a91e"
            }
        }

    def add_zone(self, name: str, points: list[list[float]], hex_color: str = "#00e5ff", zone_id: str | None = None) -> dict:
        """Adds a new polygonal zone with normalized coordinates [[x1, y1], [x2, y2], ...]."""
        if not zone_id:
            zone_id = f"zone_{uuid.uuid4().hex[:6]}"
        
        sanitized_points = []
        for pt in points:
            x = max(0.0, min(1.0, float(pt[0])))
            y = max(0.0, min(1.0, float(pt[1])))
            sanitized_points.append([x, y])

        zone_data = {
            "id": zone_id,
            "name": name.strip() or f"Zona {len(self.zones) + 1}",
            "points": sanitized_points,
            "color": self._hex_to_bgr(hex_color),
            "hex_color": hex_color
        }
        self.zones[zone_id] = zone_data
        print(f"[CrowdDetector] Added zone: {zone_data['name']} (ID: {zone_id}) with {len(points)} points")
        return self.get_zone_info(zone_id)

    def remove_zone(self, zone_id: str) -> bool:
        """Removes a zone by its ID."""
        if zone_id in self.zones:
            del self.zones[zone_id]
            print(f"[CrowdDetector] Removed zone: {zone_id}")
            return True
        return False

    def clear_all_zones(self):
        """Clears all zones."""
        self.zones.clear()
        print("[CrowdDetector] All zones cleared.")

    def reset_default_zones(self):
        """Resets back to initial 3 zones."""
        self._init_default_zones()
        print("[CrowdDetector] Reset to default zones A, B, C.")

    def get_zone_info(self, zone_id: str) -> dict | None:
        if zone_id not in self.zones:
            return None
        z = self.zones[zone_id]
        return {
            "id": z["id"],
            "name": z["name"],
            "points": z["points"],
            "hex_color": z["hex_color"]
        }

    def list_zones(self) -> list[dict]:
        return [
            {
                "id": z["id"],
                "name": z["name"],
                "points": z["points"],
                "hex_color": z["hex_color"]
            }
            for z in self.zones.values()
        ]

    def set_confidence(self, conf: float):
        self.conf_threshold = max(0.05, min(0.95, conf))

    def set_alert_threshold(self, threshold: int):
        self.threshold_alert = max(1, threshold)

    def set_engine(self, engine: str):
        """Switches detection engine between 'yolo' and 'p2pnet'."""
        if engine in ("yolo", "p2pnet"):
            self.engine = engine
            if engine == "p2pnet":
                load_p2pnet(self.p2p_weights, device=self.device)
            print(f"[CrowdDetector] Detection engine switched to: {self.engine.upper()} on {self.device}")

    def set_model(self, model_name: str):
        """Loads a model weight file (.pt for YOLO or .pth for P2PNet)."""
        self.model_name = model_name
        if "p2p" in model_name.lower() or model_name.endswith(".pth"):
            self.engine = "p2pnet"
            self.p2p_weights = model_name
            self.is_head_model = True
            load_p2pnet(self.p2p_weights, device=self.device)
            print(f"[CrowdDetector] Active model: P2PNet ({model_name}) on {self.device}")
        else:
            self.engine = "yolo"
            print(f"[CrowdDetector] Loading YOLO model: {model_name}...")
            self.model = YOLO(model_name)
            self.is_head_model = any("head" in str(name).lower() for name in self.model.names.values())
            print(f"[CrowdDetector] Loaded YOLO model: {model_name}, is_head_model={self.is_head_model}")

    def set_points_visible(self, visible: bool):
        self.show_points = visible

    def set_boxes_visible(self, visible: bool):
        self.show_boxes = visible

    def set_crowd_mode(self, mode: str):
        if mode in ("auto", "detection", "density"):
            self.crowd_mode = mode
            if mode == "density":
                self.show_heatmap = True
            elif mode == "detection":
                self.show_heatmap = False
            self.trigger_deep_scan()

    def set_heatmap_visible(self, visible: bool):
        self.show_heatmap = visible

    def _get_pixel_zones(self, width: int, height: int):
        """Converts normalized zone coordinates to pixel coordinates for current frame resolution."""
        pixel_zones = {}
        for z_id, z_info in self.zones.items():
            pts = (np.array(z_info["points"], dtype=np.float32) * [width, height]).astype(np.int32)
            pixel_zones[z_id] = {
                "name": z_info["name"],
                "pts": pts,
                "color": z_info["color"],
                "hex_color": z_info["hex_color"]
            }
        return pixel_zones

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Runs fast person detection (P2PNet head points or YOLO bounding boxes) on Apple Silicon GPU,
        evaluates polygon zone boundaries, blends density heatmap if enabled, and returns stats.
        """
        h, w = frame.shape[:2]
        
        # Share latest frame with background Crowd Eye worker
        with self._frame_lock:
            self._current_raw_frame = frame

        pixel_zones = self._get_pixel_zones(w, h)

        start_t = time.time()
        zone_counts = {z_id: 0 for z_id in self.zones.keys()}
        total_detected_all = 0
        total_in_zones = 0
        detections = []
        points = []

        # 1. Execute Detection based on active engine
        with self._inference_lock:
            if self.engine == "p2pnet":
                # Point-based localization directly returning (x, y, score)
                pts_list = predict_p2p_points(
                    frame,
                    threshold=self.conf_threshold,
                    weights_path=self.p2p_weights,
                    target_max_size=1280,
                    device=self.device
                )
                results = None
            else:
                # YOLO bounding box detection
                pts_list = None
                results = self.model.predict(
                    source=frame,
                    classes=[0],
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    max_det=self.max_det,
                    verbose=False,
                    device=self.device
                )
            sync_device()

        if self.engine == "p2pnet" and pts_list is not None:
            for px, py, sc in pts_list:
                total_detected_all += 1
                assigned_zone_id = None
                for z_id, z_data in pixel_zones.items():
                    if cv2.pointPolygonTest(z_data["pts"], (px, py), False) >= 0:
                        assigned_zone_id = z_id
                        break

                if assigned_zone_id is not None:
                    zone_counts[assigned_zone_id] += 1
                    total_in_zones += 1

                points.append((px, py, sc, assigned_zone_id))
        elif results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = xyxy

                    # Calculate centroid position (head center vs body feet)
                    cx = int((x1 + x2) / 2)
                    if self.is_head_model:
                        cy = int((y1 + y2) / 2)
                    else:
                        cy = int(y2 - (y2 - y1) * 0.15)

                    total_detected_all += 1

                    # Determine which zone this person belongs to
                    assigned_zone_id = None
                    for z_id, z_data in pixel_zones.items():
                        if cv2.pointPolygonTest(z_data["pts"], (cx, cy), False) >= 0:
                            assigned_zone_id = z_id
                            break

                    if assigned_zone_id is not None:
                        zone_counts[assigned_zone_id] += 1
                        total_in_zones += 1

                    detections.append((x1, y1, x2, y2, conf, assigned_zone_id))
                    points.append((cx, cy, conf, assigned_zone_id))

        infer_duration = time.time() - start_t
        fps = round(1.0 / max(0.001, infer_duration), 1)

        # Total count calculation: if zones exist, total_in_zones takes priority
        if len(self.zones) > 0:
            final_total_count = total_in_zones
        else:
            final_total_count = total_detected_all

        # Mode DENSITY: prioritize density estimation from DM-Count
        if self.crowd_mode == "density" and self.latest_crowd_eye.get("density_count", 0) > 0:
            final_total_count = self.latest_crowd_eye["density_count"]

        # Start rendering on annotated frame
        annotated = frame.copy()

        # 2. Render Density Heatmap Overlay if toggled ON
        if self.show_heatmap and self.latest_crowd_eye.get("heatmap_bgr") is not None:
            hm_bgr = self.latest_crowd_eye["heatmap_bgr"]
            hm_resized = cv2.resize(hm_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(annotated, 0.55, hm_resized, 0.45, 0, annotated)

        # 3. Draw Zones overlay
        if self.show_zones and len(pixel_zones) > 0:
            overlay = annotated.copy()
            alpha = 0.22  # Subtle transparent tactical tint

            for z_id, z_data in pixel_zones.items():
                pts = z_data["pts"]
                col = z_data["color"]
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], col)
                    cv2.polylines(annotated, [pts], isClosed=True, color=col, thickness=2)

                    centroid_x = int(np.mean(pts[:, 0]))
                    centroid_y = int(np.min(pts[:, 1])) - 8
                    centroid_y = max(20, centroid_y)
                    tag_text = f"{z_data['name']}: {zone_counts[z_id]}"
                    (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (centroid_x - tw//2 - 6, centroid_y - th - 4), (centroid_x + tw//2 + 6, centroid_y + 4), (10, 15, 25), -1)
                    cv2.rectangle(annotated, (centroid_x - tw//2 - 6, centroid_y - th - 4), (centroid_x + tw//2 + 6, centroid_y + 4), col, 1)
                    cv2.putText(annotated, tag_text, (centroid_x - tw//2, centroid_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # 4. Draw People Bounding Boxes (when YOLO is used and show_boxes is True)
        if self.show_boxes and len(detections) > 0:
            for x1, y1, x2, y2, conf, z_id in detections:
                if z_id is not None and z_id in pixel_zones:
                    box_color = pixel_zones[z_id]["color"]
                    is_in_zone = True
                else:
                    if len(self.zones) > 0:
                        box_color = (100, 115, 125) # Dim gray for people outside zone
                    else:
                        box_color = self.colors["box"]
                    is_in_zone = False

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)

                if is_in_zone or len(self.zones) == 0:
                    tick_len = min(6, (x2 - x1) // 3, (y2 - y1) // 3)
                    if tick_len > 1:
                        cv2.line(annotated, (x1, y1), (x1 + tick_len, y1), (255, 255, 255), 2)
                        cv2.line(annotated, (x1, y1), (x1, y1 + tick_len), (255, 255, 255), 2)
                        cv2.line(annotated, (x2, y2), (x2 - tick_len, y2), (255, 255, 255), 2)
                        cv2.line(annotated, (x2, y2), (x2, y2 - tick_len), (255, 255, 255), 2)

        # 5. Draw Head Localization Dots (for P2PNet or when show_points is True)
        if (self.engine == "p2pnet" or self.show_points) and len(points) > 0:
            for px, py, sc, z_id in points:
                if z_id is not None and z_id in pixel_zones:
                    dot_color = pixel_zones[z_id]["color"]
                else:
                    dot_color = self.colors["dot"] if len(self.zones) == 0 else (130, 145, 155)

                # Tactical glowing head dot: filled center + subtle outer ring
                cv2.circle(annotated, (px, py), 3, dot_color, -1)
                cv2.circle(annotated, (px, py), 5, (255, 255, 255), 1)

        # 6. Tactical HUD Badge showing active Engine, Hardware & Crowd Eye routing
        eye = self.latest_crowd_eye
        engine_label = "P2PNET (DOTS)" if self.engine == "p2pnet" else f"YOLO ({'HEAD' if self.is_head_model else 'BODY'})"
        active_method = eye.get("method", "detection").upper()
        hw_tag = "CUDA" if "cuda" in self.device else ("MPS" if "mps" in self.device else "CPU")
        tag_col = (0, 243, 255) if self.engine == "p2pnet" else (0, 255, 136)
        ratio_str = f"{eye.get('ratio', 1.0):.2f}" if isinstance(eye.get('ratio'), (int, float)) else str(eye.get('ratio'))
        hud_badge = f"ENGINE: {engine_label}  |  HW: {hw_tag}  |  CROWD-EYE: {active_method}  |  RATIO: {ratio_str}"
        
        (bw, bh), _ = cv2.getTextSize(hud_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (12, h - 36), (22 + bw, h - 12), (10, 15, 25), -1)
        cv2.rectangle(annotated, (12, h - 36), (22 + bw, h - 12), tag_col, 1)
        cv2.putText(annotated, hud_badge, (17, h - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Compile per-zone statistics dictionary
        zones_stat_list = []
        for z_id, z_data in self.zones.items():
            zones_stat_list.append({
                "id": z_id,
                "name": z_data["name"],
                "count": zone_counts.get(z_id, 0),
                "hex_color": z_data["hex_color"]
            })

        max_zone_count = max(zone_counts.values()) if zone_counts else 0
        alert_triggered = (max_zone_count >= self.threshold_alert)

        stats = {
            "total": final_total_count,
            "total_detected_all": total_detected_all,
            "total_in_zones": total_in_zones,
            "zones_count": len(self.zones),
            "zones": zones_stat_list,
            "zone_a": zone_counts.get("zone_a", 0),
            "zone_b": zone_counts.get("zone_b", 0),
            "zone_c": zone_counts.get("zone_c", 0),
            "fps": fps,
            "inference_ms": round(infer_duration * 1000, 1),
            "alert_triggered": alert_triggered,
            "device": self.device,
            "device_name": self.device_name,
            "engine": self.engine,
            "model_name": self.model_name,
            "show_boxes": self.show_boxes,
            "show_points": self.show_points,
            "crowd_eye": {
                "mode": self.crowd_mode,
                "method": eye.get("method", "detection"),
                "detection_count": eye.get("detection_count", total_detected_all),
                "density_count": eye.get("density_count", 0),
                "p2p_count": total_detected_all if self.engine == "p2pnet" else eye.get("p2p_count", 0),
                "final_count": eye.get("final_count", total_detected_all),
                "ratio": eye.get("ratio", 1.0),
                "packedness": eye.get("packedness", 0.0),
                "density_model": self.density_model,
                "density_weights": self.density_weights,
                "show_heatmap": self.show_heatmap,
                "scan_ms": eye.get("last_scan_ms", 0),
            }
        }

        return annotated, stats
