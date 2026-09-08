"""
api/main.py
===========
FastAPI application for LUNA-MATCH Planetary Image Registration Workbench.

Exposes core endpoints:
  - GET  /health           : Health check and diagnostic status
  - POST /register          : Submit a registration job
  - GET  /jobs/{id}         : Query job lifecycle status
  - GET  /jobs/{id}/result  : Retrieve quantitative registration metrics & artifacts
  - GET  /jobs/{id}/preview : Fetch raster or residual preview visualization
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.orchestrator import LunaMatchPipeline, JobState
from api.schemas import RegisterRequest, JobStatusResponse, JobResultResponse, ChatbotSummaryResponse
from core.ingest_preprocess import read_raster
from core.summary_builder import build_chatbot_summary

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

logger = logging.getLogger("luna_match_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LUNA-MATCH Registration API",
    description="Sub-pixel Multi-Modal, Sun-Angle, and Scale-Invariant Lunar Image Correspondence API",
    version="2.0.0",
)

# Enable CORS for frontend workbench integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


def create_checkerboard(img1: np.ndarray, img2: np.ndarray, tile_size: int = 36) -> np.ndarray:
    """Create a checkerboard mosaic interleaving img1 and img2 in blocks of tile_size x tile_size."""
    H, W = img1.shape[:2]
    if img2.shape[:2] != (H, W):
        img2 = cv2.resize(img2, (W, H))

    checkerboard = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            y_end = min(H, y + tile_size)
            x_end = min(W, x + tile_size)

            tile_y = (y // tile_size) % 2
            tile_x = (x // tile_size) % 2

            if (tile_y + tile_x) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = img1[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]

    return checkerboard


def draw_tie_points(
    img_a: np.ndarray,
    img_b: np.ndarray,
    matches: np.ndarray,
    max_draw: int = 80,
) -> np.ndarray:
    """Draw side-by-side correspondence lines connecting matched keypoints."""
    H_a, W_a = img_a.shape[:2]
    H_b, W_b = img_b.shape[:2]

    canvas_h = max(H_a, H_b)
    canvas_w = W_a + W_b

    def to_u8(img):
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        return (norm * 255.0).astype(np.uint8)

    c_a = to_u8(img_a)
    c_b = to_u8(img_b)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:H_a, :W_a] = cv2.cvtColor(c_a, cv2.COLOR_GRAY2BGR)
    canvas[:H_b, W_a:canvas_w] = cv2.cvtColor(c_b, cv2.COLOR_GRAY2BGR)

    if len(matches) > max_draw:
        step = len(matches) // max_draw
        draw_matches = matches[::step][:max_draw]
    else:
        draw_matches = matches

    for m in draw_matches:
        x1, y1, x2, y2 = int(round(m[0])), int(round(m[1])), int(round(m[2])), int(round(m[3]))
        pt1 = (x1, y1)
        pt2 = (x2 + W_a, y2)
        conf = float(m[4]) if len(m) > 4 else 0.8

        color = (0, 240, 100) if conf > 0.6 else (255, 200, 0)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    return canvas


def _find_job_inputs(job_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    paths_file = job_dir / "input" / "input_paths.json"
    if paths_file.exists():
        try:
            with open(paths_file, "r") as f:
                d = json.load(f)
                if d.get("img_a_path") and d.get("img_b_path"):
                    return d["img_a_path"], d["img_b_path"]
        except Exception:
            pass

    meta_file = job_dir / "input" / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                d = json.load(f)
                if "img_a_path" in d and "img_b_path" in d:
                    return d["img_a_path"], d["img_b_path"]
        except Exception:
            pass

    return None, None


@app.get("/health")
def health_check():
    """Service health and diagnostic status."""
    return {
        "status": "ok",
        "service": "LUNA-MATCH Registration API",
        "version": "2.0.0",
        "default_matcher": "classical",
        "structural_matching_available": True,
        "active_workers": _executor._max_workers,
    }


def _execute_pipeline_task(job_id: str, img_a_path: str, img_b_path: str):
    """Background runner for LunaMatchPipeline."""
    try:
        pipeline = LunaMatchPipeline(job_id, img_a_path, img_b_path)
        result = pipeline.run()
        logger.info(f"Job {job_id} finished with status: {result.get('status')}")
    except Exception as e:
        logger.error(f"Execution error for job {job_id}: {e}")


@app.post("/register", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def register_images(req: RegisterRequest, background_tasks: BackgroundTasks):
    """
    Initiate a new registration pipeline run between Chandrayaan-2 moving imagery
    and reference lunar imagery.
    """
    job_id = req.job_id or f"job_{uuid.uuid4().hex[:10]}"

    if not os.path.exists(req.img_a_path):
        raise HTTPException(status_code=400, detail=f"Source image not found: {req.img_a_path}")
    if not os.path.exists(req.img_b_path):
        raise HTTPException(status_code=400, detail=f"Reference image not found: {req.img_b_path}")

    # Initialize job directory structure and PENDING status
    pipeline = LunaMatchPipeline(job_id, req.img_a_path, req.img_b_path)

    # Launch execution asynchronously
    background_tasks.add_task(_execute_pipeline_task, job_id, req.img_a_path, req.img_b_path)

    return JobStatusResponse(
        job_id=job_id,
        status=JobState.PENDING.value,
        elapsed_s=0.0,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Query current execution state and elapsed runtime of a registration job."""
    status_file = Path("data") / "jobs" / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job ID not found: {job_id}")

    try:
        with open(status_file, "r") as f:
            data = json.load(f)
        return JobStatusResponse(
            job_id=data.get("job_id", job_id),
            status=data.get("status", "UNKNOWN"),
            elapsed_s=data.get("elapsed_s", 0.0),
            error=data.get("error", None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse job status: {e}")


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    """Retrieve quantitative validation metrics and output files for a completed job."""
    job_dir = Path("data") / "jobs" / job_id
    status_file = job_dir / "status.json"
    metrics_file = job_dir / "output" / "metrics.json"

    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job ID not found: {job_id}")

    with open(status_file, "r") as f:
        status_data = json.load(f)

    st = status_data.get("status", "UNKNOWN")

    if st == JobState.FAILED.value:
        return JobResultResponse(
            job_id=job_id,
            status=st,
            error=status_data.get("error", "Job execution failed"),
        )

    if not metrics_file.exists():
        return JobResultResponse(
            job_id=job_id,
            status=st,
            elapsed_s=status_data.get("elapsed_s", 0.0),
        )

    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        output_files = {
            "registered": str(job_dir / "output" / "registered.tif"),
            "residual_map": str(job_dir / "output" / "residual_map.png"),
            "metrics": str(metrics_file),
        }

        return JobResultResponse(
            job_id=job_id,
            status=st,
            rmse_px=metrics.get("rmse_px"),
            inlier_ratio=metrics.get("inlier_ratio"),
            sdi=metrics.get("sdi"),
            n_inliers=metrics.get("n_inliers"),
            n_total=metrics.get("n_total"),
            elapsed_s=metrics.get("elapsed_s"),
            transform=metrics.get("transform"),
            output_files=output_files,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {e}")


@app.get("/jobs/{job_id}/summary", response_model=ChatbotSummaryResponse)
def get_job_summary(job_id: str):
    """
    Unified Chatbot-Ready Output Endpoint (Schema v1.0).
    Assembles geodetic metrics, plain-language confidence classification,
    diagnostic reasoning, input metadata, and relative visual artifact paths.
    """
    try:
        summary = build_chatbot_summary(job_id)
        return summary
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except Exception as e:
        logger.error(f"Failed to build summary for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


@app.get("/jobs/{job_id}/preview")
def get_job_preview(job_id: str, kind: str = "registered"):
    """
    Serve preview images for frontend visualization:
      - kind='registered'   : the warped registered GeoTIFF/image
      - kind='residual'     : the 2D residual error heatmap
      - kind='checkerboard' : alternating tile mosaic of source vs warped reference
      - kind='tiepoints'    : side-by-side keypoint correspondence overlay
    """
    job_dir = Path("data") / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check job completion for derived overlays
    status_file = job_dir / "status.json"
    job_status = "UNKNOWN"
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                job_status = json.load(f).get("status", "UNKNOWN")
        except Exception:
            pass

    if kind == "residual":
        preview_path = job_dir / "output" / "residual_map.png"
        if preview_path.exists():
            return FileResponse(str(preview_path), media_type="image/png")
        raise HTTPException(status_code=404, detail="Residual map preview not yet generated.")

    elif kind == "checkerboard":
        if job_status != JobState.DONE.value:
            raise HTTPException(status_code=400, detail=f"Job is not completed (current status: {job_status})")

        cb_cache_path = job_dir / "output" / "preview_checkerboard.png"
        if cb_cache_path.exists():
            return FileResponse(str(cb_cache_path), media_type="image/png")

        path_a, _ = _find_job_inputs(job_dir)
        if not path_a or not os.path.exists(path_a):
            raise HTTPException(status_code=404, detail="Source image input path not recorded or found for checkerboard.")

        # Load raw_a
        try:
            raw_a, _ = read_raster(path_a)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read source image: {e}")

        # Load registered image
        registered_path = job_dir / "output" / "registered.tif"
        if registered_path.exists() and _HAS_RASTERIO:
            with rasterio.open(str(registered_path)) as s:
                registered_img = s.read(1).astype(np.float64)
        elif (job_dir / "output" / "registered.npy").exists():
            registered_img = np.load(str(job_dir / "output" / "registered.npy"))
        else:
            raise HTTPException(status_code=404, detail="Registered output raster not found.")

        checkerboard = create_checkerboard(raw_a, registered_img, tile_size=36)
        cb_u8 = (np.clip(checkerboard, 0, 1) * 255.0).astype(np.uint8)
        cv2.imwrite(str(cb_cache_path), cb_u8)
        return FileResponse(str(cb_cache_path), media_type="image/png")

    elif kind == "tiepoints":
        if job_status != JobState.DONE.value:
            raise HTTPException(status_code=400, detail=f"Job is not completed (current status: {job_status})")

        tp_cache_path = job_dir / "output" / "preview_tiepoints.png"
        if tp_cache_path.exists():
            return FileResponse(str(tp_cache_path), media_type="image/png")

        path_a, path_b = _find_job_inputs(job_dir)
        if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
            raise HTTPException(status_code=404, detail="Input images not recorded or found on disk.")

        matches_path = job_dir / "intermediate" / "matches_verified.npy"
        if not matches_path.exists():
            matches_path = job_dir / "intermediate" / "matches_raw.npy"
        if not matches_path.exists():
            raise HTTPException(status_code=404, detail="Match correspondences not found.")

        try:
            raw_a, _ = read_raster(path_a)
            raw_b, _ = read_raster(path_b)
            matches = np.load(str(matches_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read data for tie-points overlay: {e}")

        tp_canvas = draw_tie_points(raw_a, raw_b, matches)
        cv2.imwrite(str(tp_cache_path), tp_canvas)
        return FileResponse(str(tp_cache_path), media_type="image/png")

    # Default: registered output
    tif_path = job_dir / "output" / "registered.tif"
    if tif_path.exists():
        return FileResponse(str(tif_path), media_type="image/tiff")

    npy_path = job_dir / "output" / "registered.npy"
    if npy_path.exists():
        return FileResponse(str(npy_path), media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail=f"Preview kind '{kind}' not found or output missing.")
