import os
import shutil
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.detector import CrowdDetector
from backend.streamer import VideoStreamer
from sample_data.create_demo import create_synthetic_crowd_video

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="CrowdVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO Detector with aminesam.pt (Baseline: imgsz=1280, conf=0.10, iou=0.75)
detector = CrowdDetector(model_name="aminesam.pt", conf_threshold=0.10, iou_threshold=0.75, imgsz=1280)
streamer = VideoStreamer(detector=detector)

@app.on_event("startup")
async def startup_event():
    # Print operational hardware summary
    print("\n" + "=" * 60)
    print(" [CROWDVISION PRO - DRONE COMMAND CENTER]")
    print(f" [HARDWARE] Active Device : {detector.device_name} ({detector.device})")
    print(f" [ENGINE]   Active Model  : {detector.model_name} (Engine: {detector.engine.upper()})")
    print(f" [ROUTING]  Crowd Eye     : Mode {detector.crowd_mode.upper()} (Density: {detector.density_model})")
    print("=" * 60 + "\n")

    # Check or generate default demo crowd video
    demo_video = UPLOAD_DIR / "demo_crowd.mp4"
    if not demo_video.exists():
        print("[Startup] Generating initial synthetic crowd demo video...")
        try:
            create_synthetic_crowd_video(str(demo_video), duration_sec=15)
        except Exception as e:
            print(f"[Startup Error] Could not generate demo video: {e}")

    if demo_video.exists():
        streamer.set_video(str(demo_video))

@app.on_event("shutdown")
def shutdown_event():
    streamer.stop()

# --- PYDANTIC SCHEMAS ---

class ZoneCreateModel(BaseModel):
    name: str = Field(..., example="Zona VIP")
    points: list[list[float]] = Field(..., min_items=3, example=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]])
    hex_color: str = Field("#00e5ff", example="#ff334b")

# --- ZONE MANAGEMENT API ---

@app.get("/api/zones")
async def get_zones():
    """Returns the list of all active zones."""
    return {"zones": detector.list_zones()}

@app.post("/api/zones")
async def create_zone(zone: ZoneCreateModel):
    """Creates a new detection zone."""
    created = detector.add_zone(
        name=zone.name,
        points=zone.points,
        hex_color=zone.hex_color
    )
    return {"status": "success", "zone": created}

@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """Deletes a specific zone by ID."""
    success = detector.remove_zone(zone_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Zona {zone_id} tidak ditemukan")
    return {"status": "success", "deleted_id": zone_id}

@app.delete("/api/zones")
async def clear_zones():
    """Clears all zones."""
    detector.clear_all_zones()
    return {"status": "success", "message": "Semua zona telah dihapus"}

@app.post("/api/zones/reset")
async def reset_zones():
    """Resets to default reference zones (A, B, C)."""
    detector.reset_default_zones()
    return {"status": "success", "zones": detector.list_zones()}

# --- VIDEO UPLOAD & MANAGEMENT API ---

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video recording and switches stream to it."""
    valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in valid_extensions:
        raise HTTPException(status_code=400, detail="Format video tidak didukung. Gunakan MP4, MOV, AVI, atau MKV.")

    safe_filename = file.filename.replace(" ", "_")
    target_path = UPLOAD_DIR / safe_filename

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Switch streamer to newly uploaded video
        streamer.set_video(str(target_path))

        return {
            "status": "success",
            "filename": safe_filename,
            "message": f"Video {safe_filename} berhasil diunggah dan aktif.",
            "path": f"/uploads/{safe_filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan video: {str(e)}")

@app.get("/api/videos")
async def list_videos():
    """Lists all available uploaded videos."""
    videos = []
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                videos.append({
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "is_active": streamer.current_video_path is not None and Path(streamer.current_video_path).name == f.name
                })
    return {"videos": videos}

@app.post("/api/select-video/{video_name}")
async def select_video(video_name: str):
    """Selects an existing video from the library to stream."""
    target_path = UPLOAD_DIR / video_name
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Video tidak ditemukan")
    
    streamer.set_video(str(target_path))
    return {"status": "success", "active_video": video_name}

# --- LIVE VIDEO STREAMING & STATS ---

@app.get("/api/stream")
async def video_stream():
    """Streams annotated video frames in real-time using MJPEG."""
    return StreamingResponse(
        streamer.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stats")
async def get_stats():
    """Returns current crowd metrics and inference telemetry."""
    return streamer.latest_stats

@app.post("/api/settings")
async def update_settings(
    conf: float | None = Form(None),
    show_boxes: bool | None = Form(None),
    show_points: bool | None = Form(None),
    show_zones: bool | None = Form(None),
    show_heatmap: bool | None = Form(None),
    alert_threshold: int | None = Form(None),
    crowd_mode: str | None = Form(None),
    engine: str | None = Form(None),
    model_name: str | None = Form(None)
):
    """Updates YOLO & P2PNet detection and Crowd Eye settings on the fly."""
    if conf is not None:
        detector.set_confidence(conf)
    if show_boxes is not None:
        detector.show_boxes = show_boxes
    if show_points is not None:
        detector.show_points = show_points
    if show_zones is not None:
        detector.show_zones = show_zones
    if show_heatmap is not None:
        detector.set_heatmap_visible(show_heatmap)
    if alert_threshold is not None:
        detector.set_alert_threshold(alert_threshold)
    if crowd_mode is not None:
        detector.set_crowd_mode(crowd_mode)
    if engine is not None:
        detector.set_engine(engine)
    if model_name is not None:
        detector.set_model(model_name)

    return {
        "status": "success",
        "conf": detector.conf_threshold,
        "engine": detector.engine,
        "model_name": detector.model_name,
        "show_boxes": detector.show_boxes,
        "show_points": detector.show_points,
        "show_zones": detector.show_zones,
        "show_heatmap": detector.show_heatmap,
        "crowd_mode": detector.crowd_mode,
        "alert_threshold": detector.threshold_alert
    }

class CrowdSettingsModel(BaseModel):
    mode: str | None = None
    engine: str | None = None
    model: str | None = None
    show_heatmap: bool | None = None
    show_boxes: bool | None = None
    show_points: bool | None = None
    density_model: str | None = None
    density_weights: str | None = None
    ratio_threshold: float | None = None

@app.post("/api/crowd/settings")
async def update_crowd_settings(settings: CrowdSettingsModel):
    """Configures Crowd Eye dual-routing parameters and visualization."""
    if settings.mode is not None:
        if settings.mode == "p2pnet":
            detector.set_engine("p2pnet")
            detector.show_points = True
        else:
            detector.set_crowd_mode(settings.mode)
            if settings.mode == "detection":
                detector.set_engine("yolo")
    if settings.engine is not None:
        detector.set_engine(settings.engine)
    if settings.model is not None:
        detector.set_model(settings.model)
    if settings.show_heatmap is not None:
        detector.set_heatmap_visible(settings.show_heatmap)
    if settings.show_boxes is not None:
        detector.show_boxes = settings.show_boxes
    if settings.show_points is not None:
        detector.show_points = settings.show_points
    if settings.density_model is not None:
        detector.density_model = settings.density_model
    if settings.density_weights is not None:
        detector.density_weights = settings.density_weights
    if settings.ratio_threshold is not None:
        detector.ratio_threshold = settings.ratio_threshold

    return {
        "status": "success",
        "mode": detector.crowd_mode,
        "engine": detector.engine,
        "model_name": detector.model_name,
        "show_heatmap": detector.show_heatmap,
        "show_boxes": detector.show_boxes,
        "show_points": detector.show_points,
        "density_model": detector.density_model,
        "density_weights": detector.density_weights,
        "ratio_threshold": detector.ratio_threshold,
    }

@app.post("/api/crowd/scan")
async def trigger_deep_scan():
    """Triggers an immediate asynchronous deep density and routing scan."""
    detector.trigger_deep_scan()
    return {"status": "success", "message": "Deep scan triggered"}


@app.get("/api/snapshot")
async def take_snapshot():
    """Returns the latest annotated frame as a JPEG image download."""
    if streamer.latest_frame_jpeg is None:
        raise HTTPException(status_code=503, detail="Frame belum tersedia")
    
    from fastapi.responses import Response
    return Response(
        content=streamer.latest_frame_jpeg,
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=crowdvision_snapshot.jpg"}
    )

# --- WEBSOCKET FOR REALTIME HUD UPDATES ---

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            stats = streamer.latest_stats.copy()
            await websocket.send_json(stats)
            await asyncio.sleep(0.35)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

# Mount frontend static files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
