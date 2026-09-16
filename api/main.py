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
import re
import time
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.orchestrator import LunaMatchPipeline, JobState
from api.schemas import (
    RegisterRequest,
    JobStatusResponse,
    JobResultResponse,
    SummaryResponse,
    ChatbotSummaryResponse,
    AgentMessageRequest,
    SourceRef,
    AgentMessageResponse,
    ResearchSessionResponse,
)
from api.errors import ErrorCode, APIError, api_error_response
from core.ingest_preprocess import read_raster
from core.summary_builder import build_chatbot_summary
from core.agent_rag import retrieve, format_for_prompt
from core.warp_and_eval import export_control_points_csv
from core.dense_matcher import _check_loftr_available, LoFTRUnavailableError
from core.graph_renderer import (
    render_residual_scatter,
    render_residual_histogram,
    render_crater_histogram,
    render_confidence_gauge,
)

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

# ---------------------------------------------------------------------------
# Structured Error Taxonomy Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return api_error_response(exc.status_code, exc.error_code, exc.message, exc.job_id)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    job_id = None
    path_parts = [p for p in request.url.path.split("/") if p]
    if "jobs" in path_parts:
        idx = path_parts.index("jobs")
        if idx + 1 < len(path_parts) and path_parts[idx + 1] != "register":
            job_id = path_parts[idx + 1]

    code = ErrorCode.INTERNAL_ERROR
    msg = str(detail)
    extra = {}
    if isinstance(detail, dict):
        extra = {k: v for k, v in detail.items() if k not in ("error_code", "message", "job_id")}
        if "error_code" in detail:
            try:
                code = ErrorCode(detail["error_code"])
            except ValueError:
                code = detail["error_code"]
            msg = detail.get("message", msg)
    elif exc.status_code == 404:
        code = ErrorCode.JOB_NOT_FOUND
    elif exc.status_code == 400:
        if "not found" in msg.lower() or "image" in msg.lower() or "path" in msg.lower():
            code = ErrorCode.INVALID_INPUT_PATH
        elif "kind" in msg.lower():
            code = ErrorCode.INVALID_KIND
        elif "completed" in msg.lower() or "not done" in msg.lower():
            code = ErrorCode.JOB_NOT_DONE
        elif "loftr" in msg.lower():
            code = ErrorCode.LOFTR_UNAVAILABLE
        else:
            code = ErrorCode.INTERNAL_ERROR

    content = {
        "error_code": code.value if isinstance(code, ErrorCode) else str(code),
        "message": msg,
        "job_id": job_id,
        **extra,
    }
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"INTERNAL_ERROR on {request.url.path}: {exc}\n{traceback.format_exc()}")
    job_id = None
    path_parts = [p for p in request.url.path.split("/") if p]
    if "jobs" in path_parts:
        idx = path_parts.index("jobs")
        if idx + 1 < len(path_parts):
            job_id = path_parts[idx + 1]
    return api_error_response(500, ErrorCode.INTERNAL_ERROR, str(exc), job_id)


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


def _execute_pipeline_task(job_id: str, img_a_path: str, img_b_path: str, matching_method: str = "classical", mode: str = "standard"):
    """Background runner for LunaMatchPipeline."""
    try:
        pipeline = LunaMatchPipeline(job_id, img_a_path, img_b_path, matching_method=matching_method, mode=mode)
        result = pipeline.run()
        logger.info(f"Job {job_id} finished with status: {result.get('status')}")
    except Exception as e:
        logger.error(f"Execution error for job {job_id}: {e}\n{traceback.format_exc()}")


@app.post("/register", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def register_images(request: Request, background_tasks: BackgroundTasks):
    """
    Initiate a new registration pipeline run between Chandrayaan-2 moving imagery
    and reference lunar imagery. Supports both JSON body and multipart file uploads.
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        job_id = str(form.get("job_id") or f"job_{uuid.uuid4().hex[:10]}")
        mode = str(form.get("mode") or "standard")
        matching_method = str(form.get("matching_method") or "classical")

        img_a_upload = form.get("img_a")
        img_b_upload = form.get("img_b")

        if not img_a_upload or not img_b_upload:
            raise APIError(status_code=400, error_code=ErrorCode.INVALID_INPUT_PATH, message="Both 'img_a' and 'img_b' files must be uploaded.", job_id=job_id)

        input_dir = Path("data") / "jobs" / job_id / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        filename_a = getattr(img_a_upload, "filename", "img_a.tif") or "img_a.tif"
        filename_b = getattr(img_b_upload, "filename", "img_b.tif") or "img_b.tif"

        path_a = str(input_dir / filename_a)
        path_b = str(input_dir / filename_b)

        if hasattr(img_a_upload, "read"):
            content_a = await img_a_upload.read()
        elif hasattr(img_a_upload, "file"):
            content_a = img_a_upload.file.read()
        else:
            content_a = bytes(img_a_upload)

        with open(path_a, "wb") as f_a:
            f_a.write(content_a)

        if hasattr(img_b_upload, "read"):
            content_b = await img_b_upload.read()
        elif hasattr(img_b_upload, "file"):
            content_b = img_b_upload.file.read()
        else:
            content_b = bytes(img_b_upload)

        with open(path_b, "wb") as f_b:
            f_b.write(content_b)

        img_a_path = path_a
        img_b_path = path_b
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        job_id = body.get("job_id") or f"job_{uuid.uuid4().hex[:10]}"
        img_a_path = body.get("img_a_path")
        img_b_path = body.get("img_b_path")
        matching_method = body.get("matching_method", "classical")
        mode = body.get("mode", "standard")

        if not img_a_path or not os.path.exists(img_a_path):
            raise APIError(status_code=400, error_code=ErrorCode.INVALID_INPUT_PATH, message=f"Source image not found: {img_a_path}", job_id=job_id)
        if not img_b_path or not os.path.exists(img_b_path):
            raise APIError(status_code=400, error_code=ErrorCode.INVALID_INPUT_PATH, message=f"Reference image not found: {img_b_path}", job_id=job_id)

    if matching_method.lower() == "loftr":
        try:
            _check_loftr_available()
        except LoFTRUnavailableError as e_loftr:
            raise APIError(status_code=400, error_code=ErrorCode.LOFTR_UNAVAILABLE, message=str(e_loftr), job_id=job_id)

    # Initialize job directory structure and PENDING status
    pipeline = LunaMatchPipeline(job_id, img_a_path, img_b_path, matching_method=matching_method, mode=mode)

    # Launch execution asynchronously
    background_tasks.add_task(_execute_pipeline_task, job_id, img_a_path, img_b_path, matching_method, mode)

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
        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message=f"Job ID not found: {job_id}", job_id=job_id)

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
        raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Failed to parse job status: {e}", job_id=job_id)


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    """Retrieve quantitative validation metrics and output files for a completed job."""
    job_dir = Path("data") / "jobs" / job_id
    status_file = job_dir / "status.json"
    metrics_file = job_dir / "output" / "metrics.json"

    if not status_file.exists():
        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message=f"Job ID not found: {job_id}", job_id=job_id)

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
            transform_readable=metrics.get("transform_readable"),
            output_files=output_files,
        )
    except Exception as e:
        raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Error reading metrics: {e}", job_id=job_id)


@app.get("/jobs/{job_id}/summary", response_model=SummaryResponse)
def get_job_summary(job_id: str):
    """
    Deterministic Quality Assessment & Explanation Summary Endpoint.
    Assembles geodetic metrics, plain-language confidence classification,
    diagnostic reasoning, input metadata, and relative visual artifact paths.
    """
    try:
        summary = build_chatbot_summary(job_id)
        return summary
    except FileNotFoundError:
        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message=f"Job not found: {job_id}", job_id=job_id)
    except Exception as e:
        logger.error(f"Failed to build summary for {job_id}: {e}\n{traceback.format_exc()}")
        raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Failed to generate summary: {e}", job_id=job_id)


@app.get("/jobs/{job_id}/export")
def export_job_data(job_id: str, kind: str = "control_points"):
    """
    Export job artifacts as downloadable files:
      - kind='control_points' : returns CSV of verified + refined matches (x1, y1, x2, y2, confidence)
    """
    job_dir = Path("data") / "jobs" / job_id
    if not job_dir.exists():
        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message=f"Job not found: {job_id}", job_id=job_id)

    if kind == "control_points":
        csv_path = job_dir / "output" / "control_points.csv"
        if not csv_path.exists():
            matches_file = job_dir / "intermediate" / "matches_verified.npy"
            if not matches_file.exists():
                matches_file = job_dir / "intermediate" / "matches_raw.npy"

            if matches_file.exists():
                try:
                    matches = np.load(matches_file)
                    export_control_points_csv(matches, csv_path)
                except Exception as e_exp:
                    raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Failed to generate control points CSV: {e_exp}", job_id=job_id)
            else:
                raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message="No control points found for this job", job_id=job_id)

        return FileResponse(
            str(csv_path),
            media_type="text/csv",
            filename=f"{job_id}_control_points.csv",
        )

    raise APIError(
        status_code=400,
        error_code=ErrorCode.INVALID_KIND,
        message=f"Unsupported export kind: '{kind}'. Supported kinds: ['control_points']",
        job_id=job_id,
    )


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
        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message=f"Job not found: {job_id}", job_id=job_id)

    valid_kinds = ("registered", "residual", "checkerboard", "tiepoints", "craters")
    if kind not in valid_kinds:
        raise APIError(status_code=400, error_code=ErrorCode.INVALID_KIND, message=f"Unsupported preview kind: '{kind}'. Supported kinds: {list(valid_kinds)}", job_id=job_id)

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
        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message="Residual map preview not yet generated.", job_id=job_id)

    elif kind == "checkerboard":
        if job_status != JobState.DONE.value:
            raise APIError(status_code=400, error_code=ErrorCode.JOB_NOT_DONE, message=f"Job is not completed (current status: {job_status})", job_id=job_id)

        cb_cache_path = job_dir / "output" / "preview_checkerboard.png"
        if cb_cache_path.exists():
            return FileResponse(str(cb_cache_path), media_type="image/png")

        path_a, _ = _find_job_inputs(job_dir)
        if not path_a or not os.path.exists(path_a):
            raise APIError(status_code=404, error_code=ErrorCode.INVALID_INPUT_PATH, message="Source image input path not recorded or found for checkerboard.", job_id=job_id)

        try:
            raw_a, _ = read_raster(path_a)
        except Exception as e:
            raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Failed to read source image: {e}", job_id=job_id)

        registered_path = job_dir / "output" / "registered.tif"
        if registered_path.exists() and _HAS_RASTERIO:
            with rasterio.open(str(registered_path)) as s:
                registered_img = s.read(1).astype(np.float64)
        elif (job_dir / "output" / "registered.npy").exists():
            registered_img = np.load(str(job_dir / "output" / "registered.npy"))
        else:
            raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message="Registered output raster not found.", job_id=job_id)

        checkerboard = create_checkerboard(raw_a, registered_img, tile_size=36)
        cb_u8 = (np.clip(checkerboard, 0, 1) * 255.0).astype(np.uint8)
        cv2.imwrite(str(cb_cache_path), cb_u8)
        return FileResponse(str(cb_cache_path), media_type="image/png")

    elif kind == "tiepoints":
        if job_status != JobState.DONE.value:
            raise APIError(status_code=400, error_code=ErrorCode.JOB_NOT_DONE, message=f"Job is not completed (current status: {job_status})", job_id=job_id)

        tp_cache_path = job_dir / "output" / "preview_tiepoints.png"
        if tp_cache_path.exists():
            return FileResponse(str(tp_cache_path), media_type="image/png")

        path_a, path_b = _find_job_inputs(job_dir)
        if not path_a or not path_b or not os.path.exists(path_a) or not os.path.exists(path_b):
            raise APIError(status_code=404, error_code=ErrorCode.INVALID_INPUT_PATH, message="Input images not recorded or found on disk.", job_id=job_id)

        matches_path = job_dir / "intermediate" / "matches_verified.npy"
        if not matches_path.exists():
            matches_path = job_dir / "intermediate" / "matches_raw.npy"
        if not matches_path.exists():
            raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message="Match correspondences not found.", job_id=job_id)

        try:
            raw_a, _ = read_raster(path_a)
            raw_b, _ = read_raster(path_b)
            matches = np.load(str(matches_path))
        except Exception as e:
            raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Failed to read data for tie-points overlay: {e}", job_id=job_id)

        tp_canvas = draw_tie_points(raw_a, raw_b, matches)
        cv2.imwrite(str(tp_cache_path), tp_canvas)
        return FileResponse(str(tp_cache_path), media_type="image/png")

    elif kind == "craters":
        crater_cache_path = job_dir / "output" / "preview_craters.png"
        if crater_cache_path.exists():
            return FileResponse(str(crater_cache_path), media_type="image/png")
        crater_a_path = job_dir / "output" / "craters_a.png"
        if crater_a_path.exists():
            return FileResponse(str(crater_a_path), media_type="image/png")

        craters_json = job_dir / "intermediate" / "craters_a.json"
        path_a, _ = _find_job_inputs(job_dir)
        if craters_json.exists() and path_a and os.path.exists(path_a):
            try:
                from core.crater_detection import render_crater_overlay
                with open(craters_json, "r") as cf:
                    craters = json.load(cf)
                raw_a, _ = read_raster(path_a)
                canvas = render_crater_overlay(raw_a, craters)
                cv2.imwrite(str(crater_cache_path), canvas)
                return FileResponse(str(crater_cache_path), media_type="image/png")
            except Exception as e:
                raise APIError(status_code=500, error_code=ErrorCode.INTERNAL_ERROR, message=f"Failed to generate crater preview: {e}", job_id=job_id)

        raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message="Crater overlay preview not found or not yet generated.", job_id=job_id)

    # Default: registered output
    tif_path = job_dir / "output" / "registered.tif"
    if tif_path.exists():
        return FileResponse(str(tif_path), media_type="image/tiff")

    npy_path = job_dir / "output" / "registered.npy"
    if npy_path.exists():
        return FileResponse(str(npy_path), media_type="application/octet-stream")

    raise APIError(status_code=404, error_code=ErrorCode.JOB_NOT_FOUND, message=f"Preview kind '{kind}' not found or output missing.", job_id=job_id)


# ---------------------------------------------------------------------------
# Checkpoint 5: /jobs/{job_id}/graphs endpoint
# ---------------------------------------------------------------------------

VALID_GRAPH_KINDS = ["residual_scatter", "residual_histogram", "crater_histogram", "confidence_gauge", "sfd"]


@app.get("/jobs/{job_id}/graphs")
def get_job_graphs(job_id: str, kind: str = "residual_scatter"):
    """
    Serve rendered chart PNG artifacts for scientific visualization.

    Supported kinds:
      - kind='residual_scatter'   : Residual error scatter plot (reprojected vs reference)
      - kind='residual_histogram' : Histogram of per-point residual errors
      - kind='crater_histogram'   : Crater diameter histogram (bar chart by size class)
      - kind='confidence_gauge'   : Registration confidence quality gauge
      - kind='sfd'                : Crater Size-Frequency Distribution (log-log power law plot)

    All graphs are cached as PNG to output/ on first render and served from disk thereafter.
    """
    if kind not in VALID_GRAPH_KINDS:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": ErrorCode.INVALID_KIND.value,
                "message": f"Unsupported graph kind: '{kind}'. Supported kinds: {VALID_GRAPH_KINDS}",
                "valid_values": VALID_GRAPH_KINDS,
            },
        )

    job_dir = Path("data") / "jobs" / job_id
    status_file = job_dir / "status.json"

    if not job_dir.exists():
        raise APIError(
            status_code=404,
            error_code=ErrorCode.JOB_NOT_FOUND,
            message=f"Job not found: {job_id}",
            job_id=job_id,
        )

    if not status_file.exists():
        if kind == "sfd" and (job_dir / "intermediate" / "craters_a.json").exists():
            job_status = JobState.DONE.value
        else:
            raise APIError(
                status_code=404,
                error_code=ErrorCode.JOB_NOT_FOUND,
                message=f"Job status not found for: {job_id}",
                job_id=job_id,
            )
    else:
        try:
            with open(status_file, "r") as f:
                job_status = json.load(f).get("status", "UNKNOWN")
        except Exception:
            job_status = "UNKNOWN"

    if job_status != JobState.DONE.value:
        raise APIError(
            status_code=409,
            error_code=ErrorCode.JOB_NOT_DONE,
            message=f"Job is not completed (current status: {job_status})",
            job_id=job_id,
        )

    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"graph_{kind}.png"

    if cache_path.exists():
        return FileResponse(str(cache_path), media_type="image/png")

    inter_dir = job_dir / "intermediate"

    if kind in ("residual_scatter", "residual_histogram"):
        matches_path = inter_dir / "matches_refined.npy"
        if not matches_path.exists():
            matches_path = inter_dir / "matches_verified.npy"

        refined = np.empty((0, 4))
        if matches_path.exists():
            try:
                refined = np.load(str(matches_path))
            except Exception:
                refined = np.empty((0, 4))

        if refined.size > 0 and refined.ndim == 2 and refined.shape[1] >= 4:
            src_pts = refined[:, 0:2]
            dst_pts = refined[:, 2:4]
            res_mags = np.linalg.norm(dst_pts - src_pts, axis=1)
        else:
            src_pts = np.empty((0, 2))
            dst_pts = np.empty((0, 2))
            res_mags = np.empty((0,))

        if kind == "residual_scatter":
            render_residual_scatter(src_pts, dst_pts, str(cache_path))
        else:
            render_residual_histogram(res_mags, str(cache_path))

    elif kind == "crater_histogram":
        craters_path = inter_dir / "craters_a.json"
        craters = []
        if craters_path.exists():
            try:
                with open(craters_path, "r") as cf:
                    craters = json.load(cf)
            except Exception:
                craters = []
        render_crater_histogram(craters, str(cache_path))

    elif kind == "confidence_gauge":
        grade = "F"
        confidence_label = "unknown"
        summary_path = output_dir / "summary.json"
        metrics_path = output_dir / "metrics.json"

        if summary_path.exists():
            try:
                with open(summary_path, "r") as sf:
                    sdata = json.load(sf)
                qa = sdata.get("quality_assessment", {})
                grade = qa.get("grade", grade)
                confidence_label = qa.get("confidence_label", confidence_label)
            except Exception:
                pass
        elif metrics_path.exists():
            try:
                with open(metrics_path, "r") as mf:
                    mdata = json.load(mf)
                qa = mdata.get("quality_assessment", {})
                grade = qa.get("grade", grade)
                confidence_label = qa.get("confidence_label", confidence_label)
            except Exception:
                pass

        if grade == "F" and confidence_label == "unknown":
            try:
                summary = build_chatbot_summary(job_id)
                qa = summary.get("quality_assessment", {})
                grade = qa.get("grade", grade)
                confidence_label = qa.get("confidence_label", confidence_label)
            except Exception:
                pass

        render_confidence_gauge(grade, confidence_label, str(cache_path))

    elif kind == "sfd":
        craters_a_path = inter_dir / "craters_a.json"
        craters_b_path = inter_dir / "craters_b.json"
        craters = []
        for p in [craters_a_path, craters_b_path]:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        craters.extend(json.load(f))
                except Exception:
                    pass

        from core.crater_detection import compute_sfd
        sfd = compute_sfd(craters, n_bins=12)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")

            bin_centers = sfd.get("bin_centers_m", [])
            bin_counts = sfd.get("bin_counts", [])
            slope = sfd.get("power_law_slope")
            intercept = sfd.get("power_law_intercept")
            r2 = sfd.get("sfd_r2")
            n_craters = sfd.get("n_craters", 0)

            if bin_centers and bin_counts and any(c > 0 for c in bin_counts):
                valid_mask = [c > 0 for c in bin_counts]
                bc_valid = [bin_centers[i] for i, v in enumerate(valid_mask) if v]
                bn_valid = [bin_counts[i] for i, v in enumerate(valid_mask) if v]

                ax.loglog(
                    bc_valid, bn_valid,
                    'o-', color='#00d4ff', linewidth=2, markersize=6,
                    label=f"N(>D) — {n_craters} craters"
                )

                if slope is not None and intercept is not None and len(bc_valid) >= 3:
                    import numpy as _np
                    d_range = _np.logspace(
                        _np.log10(min(bc_valid)), _np.log10(max(bc_valid)), 50
                    )
                    n_fit = 10 ** (slope * _np.log10(d_range) + intercept)
                    ax.loglog(
                        d_range, n_fit, '--', color='#ff6b6b', linewidth=1.8,
                        label=f"Power law: N ∝ D^{slope:.2f} (R²={r2:.2f})"
                    )

                ax.set_xlabel("Crater Diameter D (m)", color='white', fontsize=11)
                ax.set_ylabel("Cumulative Count N(>D)", color='white', fontsize=11)
                ax.set_title(
                    f"Crater Size-Frequency Distribution (SFD) — Job {job_id[:12]}",
                    color='white', fontsize=12, fontweight='bold'
                )
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('gray')
                ax.spines['left'].set_color('gray')
                ax.legend(loc='upper right', facecolor='#1a1a2e', labelcolor='white', fontsize=9)
                ax.grid(True, which='both', linestyle='--', alpha=0.3, color='gray')
            else:
                ax.text(
                    0.5, 0.5, f"No crater data available\n(n_craters={n_craters})",
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray', fontsize=12
                )
                ax.set_xlabel("Crater Diameter (m)", color='white')
                ax.set_ylabel("Cumulative Count", color='white')

            plt.tight_layout(pad=1.2)
            plt.savefig(str(cache_path), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
        except Exception as e_sfd:
            logger.error(f"SFD graph render failed: {e_sfd}")

    if cache_path.exists():
        return FileResponse(str(cache_path), media_type="image/png")

    raise APIError(
        status_code=500,
        error_code=ErrorCode.INTERNAL_ERROR,
        message=f"Failed to generate graph '{kind}'",
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# AI Agent Explanation Layer & Research Companion
# ---------------------------------------------------------------------------

# Result Explainer: keyed by job_id
# {job_id: {summary: dict, history: list[dict], created_at: datetime}}
explain_sessions: dict[str, dict] = {}

# Research Companion: keyed by session_id
# {session_id: {history: list[dict], created_at: datetime}}
research_sessions: dict[str, dict] = {}

# Pre-fetched RAG cache for common questions
_rag_prefetch_cache: dict[str, list[dict]] = {}

GENERATIVE_OVERRIDE_WORDS = [
    r'\bexplain\b',
    r'\bwhy\b',
    r'\bdescribe\b',
    r'\btell me about\b',
    r'\bwhat does\b',
    r'\bwhat do\b',
    r'\bsuitable\b',
    r'\bimportant\b',
    r'\bsignifican',
    r'\bcaused?\b',
    r'\bmeaning\b',
    r'\binterpret\b',
    r'\bhow does\b',
    r'\bcan i use\b',
    r'\bshould i\b',
]

FAST_PATH_PATTERNS = [
    (r"\brmse\b", "rmse_px"),
    (r"\binlier", "inlier_ratio"),
    (r"\bsdi\b|\bspatial.{0,15}dispers", "sdi"),
    (r"\bgrade\b|\breliable\b|\bconfidence\b", "grade"),
    (r"\bssim\b", "ssim"),
    (r"\bncc\b", "ncc"),
    (r"\bmae\b|\bmean.{0,10}abs", "mae_px"),
    (r"\binliers?\b|\bmatches\b|\btie.?point", "n_inliers"),
    (r"\btransform\b|\bhomography\b|\baffine\b|\btps\b", "transform_type"),
    (r"\belapsed\b|\btime\b|\bfast\b|\bslow\b|\blatency\b|\bspeed\b", "elapsed_s"),
]


def _try_fast_path(query: str, summary: dict) -> Optional[str]:
    """
    Check if query matches a known metric pattern.
    If yes: return a formatted string answer using summary data directly.
    If no: return None (falls through to generative path).
    Use FAST_PATH_PATTERNS regex list. Case-insensitive.
    For matched field: extract value from summary["metrics"] or
    summary["quality_assessment"] and format a one-sentence answer.
    """
    query_lower = query.lower()
    for override in GENERATIVE_OVERRIDE_WORDS:
        if re.search(override, query_lower):
            return None  # force generative path

    metrics = summary.get("metrics", {})
    qa = summary.get("quality_assessment", {})

    for pattern, field in FAST_PATH_PATTERNS:
        if re.search(pattern, query_lower):
            if field == "rmse_px":
                rmse = metrics.get("rmse_px")
                if rmse is not None:
                    subpixel = " (sub-pixel accurate — below the 0.5px threshold)" if float(rmse) < 0.5 else ""
                    return f"The reprojection RMSE is {float(rmse):.4f} px{subpixel}."
                return "The reprojection RMSE is not available."
            elif field in ("inlier_ratio", "n_inliers"):
                n_inliers = metrics.get("n_inliers", 0)
                n_total = metrics.get("n_total", 0)
                ratio = metrics.get("inlier_ratio")
                if ratio is not None:
                    return f"{n_inliers} out of {n_total} candidates verified ({float(ratio)*100:.1f}% inlier ratio)."
                return f"{n_inliers} out of {n_total} candidate matches verified."
            elif field == "sdi":
                sdi = metrics.get("sdi")
                if sdi is not None:
                    spread = "well-spread across the scene" if float(sdi) >= 0.6 else ("moderately spread" if float(sdi) >= 0.3 else "clustered in a localized region")
                    return f"The Spatial Dispersion Index (SDI) is {float(sdi):.4f} ({spread})."
                return "Spatial Dispersion Index (SDI) is not available."
            elif field == "grade":
                grade = qa.get("grade", "N/A")
                conf = qa.get("confidence_label", "unknown")
                reason = qa.get("reasoning", "")
                return f"Grade {grade} — {conf}. {reason}".strip()
            elif field == "ssim":
                ssim = metrics.get("ssim")
                if ssim is not None:
                    return f"The Structural Similarity Index (SSIM) is {float(ssim):.4f}."
                return "SSIM metric is not available."
            elif field == "ncc":
                ncc = metrics.get("ncc")
                if ncc is not None:
                    return f"The Normalized Cross-Correlation (NCC) is {float(ncc):.4f}."
                return "NCC metric is not available."
            elif field == "mae_px":
                mae = metrics.get("mae_px")
                if mae is not None:
                    return f"The Mean Absolute Error (MAE) is {float(mae):.4f} px."
                return "MAE metric is not available."
            elif field == "transform_type":
                tf = metrics.get("transform_type") or "homography"
                cond = metrics.get("condition_number")
                cond_str = f" with condition number {float(cond):.2f}" if cond is not None else ""
                return f"Geometric transformation fitted: {tf}{cond_str}."
            elif field == "elapsed_s":
                elapsed = metrics.get("elapsed_s", 0.0)
                return f"Registration completed in {float(elapsed):.2f} seconds."
    return None


def _call_groq_with_rag(
    query: str,
    summary: Optional[dict],
    history: List[dict],
    rag_context: str,
) -> Tuple[str, float]:
    """
    Build system prompt grounding the LLM in:
      1. rag_context (retrieved knowledge chunks)
      2. summary metrics (if available — None for research companion)
    Call Groq (via openai client or direct request with GROQ_API_KEY).
    Return (reply_text, latency_ms).
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "GROQ_API_KEY_MISSING",
                "message": "GROQ_API_KEY environment variable is not set.",
            },
        )

    t0 = time.time()
    metrics_str = f"Registration metrics:\n{json.dumps(summary['metrics'], indent=2)}\n" if summary and "metrics" in summary else ""
    quality_str = f"Quality: {summary['quality_assessment']['reasoning']}\n" if summary and "quality_assessment" in summary and "reasoning" in summary["quality_assessment"] else ""

    system_prompt = (
        "You are LUNA-MATCH, a scientific assistant for lunar image registration. "
        "Answer based only on the provided context and metrics. Do not invent numbers.\n\n"
        f"{metrics_str}"
        f"{quality_str}"
        f"Scientific knowledge context:\n{rag_context}\n\n"
        "Keep answers under 120 words unless asked for more detail."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=groq_api_key,
        )
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=350,
            temperature=0.2,
        )
        reply_text = response.choices[0].message.content or ""
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Groq API call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error_code": "GROQ_API_ERROR", "message": f"LLM inference failed: {str(e)}"},
        )

    latency_ms = (time.time() - t0) * 1000.0
    return reply_text, latency_ms


@app.post("/agent/explain/{job_id}/start")
async def explain_start(job_id: str):
    try:
        summary = build_chatbot_summary(job_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found.", "job_id": job_id},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "INTERNAL_ERROR", "message": str(e), "job_id": job_id},
        )

    job_status = summary.get("status")
    if job_status != "DONE":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": "JOB_NOT_DONE",
                "message": f"Job {job_id} is not completed (current status: {job_status}).",
                "job_id": job_id,
            },
        )

    explain_sessions[job_id] = {
        "summary": summary,
        "history": [],
        "created_at": datetime.now(),
    }

    def _prefetch():
        try:
            chunks_1 = retrieve("explain registration result", top_k=4)
            chunks_2 = retrieve("is this reliable", top_k=4)
            _rag_prefetch_cache[job_id] = {
                "explain registration result": chunks_1,
                "is this reliable": chunks_2,
            }
        except Exception as ex:
            logger.warning(f"RAG prefetch failed for {job_id}: {ex}")

    _executor.submit(_prefetch)

    return {"job_id": job_id, "context_ready": True}


@app.post("/agent/explain/{job_id}/message", response_model=AgentMessageResponse)
async def explain_message(job_id: str, req: AgentMessageRequest):
    if job_id not in explain_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": f"Call POST /agent/explain/{job_id}/start first",
                "job_id": job_id,
            },
        )

    session = explain_sessions[job_id]
    query = req.query.strip()
    t0 = time.time()

    fast_answer = _try_fast_path(query, session["summary"])
    if fast_answer:
        latency_ms = (time.time() - t0) * 1000.0
        session["history"].append({"role": "user", "content": query})
        session["history"].append({"role": "assistant", "content": fast_answer})
        if len(session["history"]) > 20:
            session["history"] = session["history"][-20:]
        return AgentMessageResponse(
            text_response=fast_answer,
            used_fast_path=True,
            latency_ms=latency_ms,
            sources=[],
        )

    # Generative path
    prefetch_dict = _rag_prefetch_cache.get(job_id, {})
    if query in prefetch_dict:
        chunks = prefetch_dict[query]
    else:
        chunks = retrieve(query, top_k=4)

    prompt_context = format_for_prompt(chunks)
    reply_text, latency_ms = _call_groq_with_rag(query, session["summary"], session["history"], prompt_context)

    session["history"].append({"role": "user", "content": query})
    session["history"].append({"role": "assistant", "content": reply_text})
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    sources = [
        SourceRef(
            source_file=c["source_file"],
            section_title=c["section_title"],
            score=float(c.get("score", 0.0)),
        )
        for c in chunks
    ]

    return AgentMessageResponse(
        text_response=reply_text,
        used_fast_path=False,
        latency_ms=latency_ms,
        sources=sources,
    )


@app.post("/research/session", response_model=ResearchSessionResponse)
async def create_research_session():
    now = datetime.now()
    cutoff = now - timedelta(minutes=30)
    expired = [sid for sid, sdata in list(research_sessions.items()) if sdata.get("created_at", now) < cutoff]
    for sid in expired:
        research_sessions.pop(sid, None)

    session_id = str(uuid.uuid4())
    research_sessions[session_id] = {
        "history": [],
        "created_at": now,
    }
    return ResearchSessionResponse(session_id=session_id)


@app.post("/research/{session_id}/message", response_model=AgentMessageResponse)
async def research_message(session_id: str, req: AgentMessageRequest):
    if session_id not in research_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "JOB_NOT_FOUND",
                "message": "Session expired or not found. Call POST /research/session to start.",
            },
        )

    session = research_sessions[session_id]
    query = req.query.strip()

    chunks = retrieve(query, top_k=4)
    prompt_context = format_for_prompt(chunks)
    reply_text, latency_ms = _call_groq_with_rag(query, None, session["history"], prompt_context)

    session["history"].append({"role": "user", "content": query})
    session["history"].append({"role": "assistant", "content": reply_text})
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]
    session["created_at"] = datetime.now()

    sources = [
        SourceRef(
            source_file=c["source_file"],
            section_title=c["section_title"],
            score=float(c.get("score", 0.0)),
        )
        for c in chunks
    ]

    return AgentMessageResponse(
        text_response=reply_text,
        used_fast_path=False,
        latency_ms=latency_ms,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# Compatibility & Analysis Routes
# ---------------------------------------------------------------------------

class ChatMessageRequest(BaseModel):
    job_id: str
    message: str
    session_id: Optional[str] = None


@app.post("/chat")
async def chat_endpoint(req: ChatMessageRequest):
    """Interactive text chat endpoint for a registration job."""
    job_id = req.job_id
    if job_id not in explain_sessions:
        try:
            summary = build_chatbot_summary(job_id)
            explain_sessions[job_id] = {
                "summary": summary,
                "history": [],
                "created_at": datetime.now(),
            }
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail={"error_code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found."})

    session = explain_sessions[job_id]
    fast = _try_fast_path(req.message, session["summary"])
    if fast:
        session["history"].append({"role": "user", "content": req.message})
        session["history"].append({"role": "assistant", "content": fast})
        return {"response": fast, "used_fast_path": True, "job_id": job_id}

    chunks = retrieve(req.message, top_k=4)
    reply, latency_ms = _call_groq_with_rag(req.message, session["summary"], session["history"], format_for_prompt(chunks))
    session["history"].append({"role": "user", "content": req.message})
    session["history"].append({"role": "assistant", "content": reply})
    return {"response": reply, "used_fast_path": False, "latency_ms": latency_ms, "job_id": job_id}


@app.get("/chat/{job_id}/history")
def get_chat_history(job_id: str):
    """Retrieve chat turn history for a given job session."""
    session = explain_sessions.get(job_id)
    if not session:
        return {"job_id": job_id, "history": []}
    return {"job_id": job_id, "history": session.get("history", [])}


@app.post("/analyze")
def analyze_endpoint(job_id: str):
    """Metric-based analysis endpoint conforming to summary schema."""
    try:
        return build_chatbot_summary(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail={"error_code": "JOB_NOT_FOUND", "message": f"Job {job_id} not found."})

