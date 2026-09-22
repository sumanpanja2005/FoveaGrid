import os
import sys
import asyncio
import logging
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualization.websocket import ConnectionManager
from pipeline.pipeline_runner import LidarPerceptionPipeline
from pipeline.ingestion import LidarDataIngestor, SyntheticStreamGenerator
from pipeline.payload import Unified25DPayloadPacker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("foveagrid.backend")

app = FastAPI(title="FoveaGrid Unified 2.5D Elevation Mapping Server")
manager = ConnectionManager()

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

class StreamState:
    def __init__(self):
        self.is_playing: bool = True
        self.fps_target: float = 12.0
        self.source_type: str = "dummy"  # "dummy", "synthetic", or "dataset"
        self.current_frame_id: int = 0
        self.pipeline = LidarPerceptionPipeline(
            voxel_size=0.15,
            dem_resolution=0.25,
            bev_resolution=0.25
        )
        self.synth_gen = SyntheticStreamGenerator(seed=101)
        self.dummy_file = "data/dummy/roadside_sample.npy"
        self.dataset_scans = []
        self.dataset_idx = 0
        self.last_payload: Optional[Dict[str, Any]] = None
        self._ensure_dummy_data()
        self._ensure_synthetic_data()
        self._init_dataset()

    def _ensure_dummy_data(self):
        """Auto-generates dummy roadside LiDAR data if not already present."""
        if not os.path.exists(self.dummy_file):
            logger.info("Dummy roadside data not found. Auto-generating standalone sample...")
            try:
                from scripts.generate_dummy_data import generate_dummy_data
                generate_dummy_data(output_dir="data/dummy")
            except Exception as e:
                logger.error(f"Error auto-generating dummy data: {e}", exc_info=True)

    def _ensure_synthetic_data(self):
        """Auto-generates synthetic 64-beam LiDAR dataset if not already present."""
        sample_file = "data/synthetic/scene_000001/scan_000.npy"
        if not os.path.exists(sample_file):
            logger.info("Synthetic LiDAR dataset not found. Auto-generating 64-beam scans...")
            try:
                from scripts.generate_dataset import generate_dataset
                generate_dataset(num_scenes=2, scans_per_scene=5, output_dir="data/synthetic")
            except Exception as e:
                logger.error(f"Error auto-generating synthetic dataset: {e}", exc_info=True)

    def _init_dataset(self):
        """Loads available scan files for dataset playback."""
        self.dataset_scans = []
        data_dir = "data/synthetic"
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in sorted(files):
                    if f.endswith(".npy"):
                        self.dataset_scans.append(os.path.join(root, f))

        if os.path.exists(self.dummy_file) and self.dummy_file not in self.dataset_scans:
            self.dataset_scans.insert(0, self.dummy_file)

        logger.info(f"Initialized perception sources with {len(self.dataset_scans)} scans.")

stream_state = StreamState()

@app.get("/")
async def get_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FoveaGrid Unified 2.5D Visualizer</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Ensure client immediately receives a frame
    if stream_state.last_payload is None:
        await step_frame()
    else:
        try:
            await websocket.send_json(stream_state.last_payload)
        except Exception:
            pass

    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "play":
                stream_state.is_playing = True
            elif msg == "pause":
                stream_state.is_playing = False
            elif msg == "step":
                await step_frame()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class ConfigRequest(BaseModel):
    voxel_size: Optional[float] = None
    dem_resolution: Optional[float] = None
    ground_method: Optional[str] = None
    fps: Optional[float] = None
    source: Optional[str] = None

@app.get("/api/stream/status")
async def get_status():
    return {
        "is_playing": stream_state.is_playing,
        "fps_target": stream_state.fps_target,
        "source": stream_state.source_type,
        "frame_id": stream_state.current_frame_id,
        "connected_clients": len(manager.active_connections)
    }

@app.post("/api/stream/play")
async def play_stream():
    stream_state.is_playing = True
    return {"status": "playing"}

@app.post("/api/stream/pause")
async def pause_stream():
    stream_state.is_playing = False
    return {"status": "paused"}

@app.post("/api/stream/step")
async def step_stream():
    payload = await step_frame()
    return {"status": "stepped", "frame_id": stream_state.current_frame_id}

@app.post("/api/stream/source")
async def set_source(req: Dict[str, str]):
    src = req.get("source", "dummy").lower()
    if src in ["dummy", "synthetic", "dataset"]:
        stream_state.source_type = src
        await step_frame()
        return {"status": "source_changed", "source": src}
    return JSONResponse(status_code=400, content={"error": f"Invalid source: {src}"})

@app.post("/api/stream/config")
async def update_config(req: ConfigRequest):
    if req.fps:
        stream_state.fps_target = max(1.0, min(30.0, req.fps))
    if req.voxel_size:
        stream_state.pipeline.voxel_size = req.voxel_size
    if req.dem_resolution:
        stream_state.pipeline.dem_resolution = req.dem_resolution
        stream_state.pipeline.dem_generator = stream_state.pipeline.dem_generator.__class__(
            bounds=stream_state.pipeline.bev_bounds, resolution=req.dem_resolution
        )
    if req.ground_method:
        stream_state.pipeline.ground_method = req.ground_method.lower()
    return {"status": "config_updated"}

async def step_frame() -> Dict[str, Any]:
    """Processes a single LiDAR frame and broadcasts the 2.5D payload."""
    stream_state.current_frame_id += 1

    # Ingest scan based on active source
    if stream_state.source_type == "dummy":
        pcd = LidarDataIngestor.load(stream_state.dummy_file)
        source_name = "Dummy Roadside LiDAR Data"
        source_format = pcd.metadata.get("format", "npy")
    elif stream_state.source_type == "dataset" and len(stream_state.dataset_scans) > 0:
        scan_path = stream_state.dataset_scans[stream_state.dataset_idx]
        stream_state.dataset_idx = (stream_state.dataset_idx + 1) % len(stream_state.dataset_scans)
        pcd = LidarDataIngestor.load(scan_path)
        source_name = f"File: {os.path.basename(scan_path)}"
        source_format = pcd.metadata.get("format", "npy")
    else:
        pcd = stream_state.synth_gen.next_frame(dt=1.0 / stream_state.fps_target)
        source_name = "Synthetic Dynamic Stream"
        source_format = "Procedural LiDAR"

    # Run perception pipeline
    result = stream_state.pipeline.process(pcd)

    # Pack into unified 2.5D payload
    payload = Unified25DPayloadPacker.pack(
        dem=result["dem"],
        features=result["features"],
        bev=result["bev"],
        timings=result["timings"],
        raw_count=result["raw_points_count"],
        processed_count=result["processed_points_count"],
        ground_count=result["ground_points_count"],
        source_name=source_name,
        source_format=source_format
    )

    stream_state.last_payload = payload
    await manager.broadcast(payload)
    return payload

async def streaming_worker():
    """Background async worker loop for continuous 2.5D map broadcasting."""
    logger.info("Started 2.5D elevation map background streaming loop.")
    while True:
        try:
            if stream_state.is_playing and len(manager.active_connections) > 0:
                await step_frame()
            await asyncio.sleep(1.0 / stream_state.fps_target)
        except Exception as e:
            logger.error(f"Error in streaming worker: {e}", exc_info=True)
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(streaming_worker())

def start_server(host: str = "127.0.0.1", port: int = 8000):
    logger.info(f"Starting FoveaGrid 2.5D Visualizer at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
