import cv2
import time
import os
import threading
import traceback
from pathlib import Path
import numpy as np
from backend.detector import CrowdDetector

class VideoStreamer:
    def __init__(self, detector: CrowdDetector):
        self.detector = detector
        self.current_video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Latest cached state
        self.latest_frame_jpeg: bytes | None = None
        self.latest_raw_frame: np.ndarray | None = None
        self.latest_stats = {
            "total": 0,
            "total_detected_all": 0,
            "total_in_zones": 0,
            "zones_count": 0,
            "zones": [],
            "zone_a": 0,
            "zone_b": 0,
            "zone_c": 0,
            "fps": 0.0,
            "inference_ms": 0.0,
            "alert_triggered": False,
            "video_name": "No Video Active"
        }
        self.worker_thread: threading.Thread | None = None

    def set_video(self, video_path: str):
        with self.lock:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            self.current_video_path = video_path
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"[VideoStreamer ERROR] Could not open video: {video_path}")
            else:
                print(f"[VideoStreamer] Successfully opened video: {video_path}")
                
            self.latest_stats["video_name"] = Path(video_path).name

        # Ensure worker thread is running
        self.start()

    def start(self):
        with self.lock:
            self.is_running = True
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
                self.worker_thread.start()
                print("[VideoStreamer] Background processing worker started.")

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

    def _process_loop(self):
        target_fps = 25.0
        frame_interval = 1.0 / target_fps
        print("[VideoStreamer] Processing loop entered.")

        while self.is_running:
            loop_start = time.time()
            frame = None

            try:
                with self.lock:
                    if self.cap is not None and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            # Video ended, loop back to the start
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()

                if frame is not None:
                    # Resize if frame is larger than 1280w to keep inference fast
                    h, w = frame.shape[:2]
                    if w > 1280:
                        scale = 1280.0 / w
                        frame = cv2.resize(frame, (1280, int(h * scale)))

                    # Run YOLO crowd detection
                    annotated_frame, stats = self.detector.process_frame(frame)
                    
                    # Encode to JPEG
                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        jpeg_bytes = buffer.tobytes()
                        with self.lock:
                            self.latest_frame_jpeg = jpeg_bytes
                            self.latest_raw_frame = annotated_frame
                            stats["video_name"] = Path(self.current_video_path).name if self.current_video_path else "Active Stream"
                            self.latest_stats = stats

            except Exception as e:
                print(f"[VideoStreamer Loop Exception] {e}")
                traceback.print_exc()
                time.sleep(0.2)

            # Rate limiter to avoid burning 100% CPU unnecessarily
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, frame_interval - elapsed)
            time.sleep(sleep_time)

        print("[VideoStreamer] Processing loop exited.")

    async def generate_mjpeg(self):
        """Yields multipart/x-mixed-replace frame stream for HTTP response asynchronously."""
        import asyncio
        last_sent_bytes = None
        try:
            while self.is_running:
                frame_bytes = None
                with self.lock:
                    frame_bytes = self.latest_frame_jpeg

                if frame_bytes is not None and frame_bytes is not last_sent_bytes:
                    last_sent_bytes = frame_bytes
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                await asyncio.sleep(0.033)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
