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
import traceback
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.orchestrator import LunaMatchPipeline, JobState
from api.schemas import RegisterRequest, JobStatusResponse, JobResultResponse, SummaryResponse, ChatbotSummaryResponse
from api.errors import ErrorCode, APIError, api_error_response
from core.ingest_preprocess import read_raster
from core.summary_builder import build_chatbot_summary
from core.warp_and_eval import export_control_points_csv
from core.dense_matcher import _check_loftr_available, LoFTRUnavailableError

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
                code = ErrorCode.INTERNAL_ERROR
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
def register_images(req: RegisterRequest, background_tasks: BackgroundTasks):
    """
    Initiate a new registration pipeline run between Chandrayaan-2 moving imagery
    and reference lunar imagery.
    """
    job_id = req.job_id or f"job_{uuid.uuid4().hex[:10]}"

    if not os.path.exists(req.img_a_path):
        raise APIError(status_code=400, error_code=ErrorCode.INVALID_INPUT_PATH, message=f"Source image not found: {req.img_a_path}", job_id=job_id)
    if not os.path.exists(req.img_b_path):
        raise APIError(status_code=400, error_code=ErrorCode.INVALID_INPUT_PATH, message=f"Reference image not found: {req.img_b_path}", job_id=job_id)

    if req.matching_method.lower() == "loftr":
        try:
            _check_loftr_available()
        except LoFTRUnavailableError as e_loftr:
            raise APIError(status_code=400, error_code=ErrorCode.LOFTR_UNAVAILABLE, message=str(e_loftr), job_id=job_id)

    # Initialize job directory structure and PENDING status
    pipeline = LunaMatchPipeline(job_id, req.img_a_path, req.img_b_path, matching_method=req.matching_method, mode=req.mode)

    # Launch execution asynchronously
    background_tasks.add_task(_execute_pipeline_task, job_id, req.img_a_path, req.img_b_path, req.matching_method, req.mode)

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

VALID_GRAPH_KINDS = ["sfd", "residual_scatter", "residual_histogram", "crater_histogram", "confidence_gauge"]


@app.get("/jobs/{job_id}/graphs")
def get_job_graphs(job_id: str, kind: str = "sfd"):
    """
    Serve rendered chart PNG artifacts for scientific visualization.

    Supported kinds:
      - kind='sfd'                : Crater Size-Frequency Distribution (log-log power law plot)
      - kind='residual_scatter'   : Residual error scatter plot (reprojected vs reference)
      - kind='residual_histogram' : Histogram of per-point residual errors
      - kind='crater_histogram'   : Crater diameter histogram (bar chart by size class)
      - kind='confidence_gauge'   : Registration confidence quality gauge

    All graphs are cached as PNG to output/ on first render and served from disk thereafter.
    """
    if kind not in VALID_GRAPH_KINDS:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_KIND",
                "message": f"Unknown graph kind: '{kind}'",
                "valid_values": VALID_GRAPH_KINDS,
            }
        )

    job_dir = Path("data") / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- SFD (Crater Size-Frequency Distribution) ----
    if kind == "sfd":
        sfd_cache = output_dir / "graph_sfd.png"
        if sfd_cache.exists():
            return FileResponse(str(sfd_cache), media_type="image/png")

        # Load crater data
        craters_a_path = job_dir / "intermediate" / "craters_a.json"
        craters_b_path = job_dir / "intermediate" / "craters_b.json"

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

                # Power law fit overlay
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

                # Physical age annotation
                if slope is not None:
                    age_str = ("Older terrain (b≥2.5)" if slope <= -2.5
                               else "Younger terrain (b<2.0)" if slope > -2.0
                               else "Moderate age terrain")
                    ax.text(
                        0.02, 0.05, age_str,
                        transform=ax.transAxes, color='#ffdd57', fontsize=9, style='italic'
                    )
            else:
                ax.text(
                    0.5, 0.5, f"No crater data available\n(n_craters={n_craters})",
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray', fontsize=12
                )
                ax.set_xlabel("Crater Diameter (m)", color='white')
                ax.set_ylabel("Cumulative Count", color='white')

            plt.tight_layout(pad=1.2)
            plt.savefig(str(sfd_cache), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return FileResponse(str(sfd_cache), media_type="image/png")

        except ImportError:
            # matplotlib not available — return SFD data as JSON fallback
            return JSONResponse(content=sfd)
        except Exception as e:
            logger.error(f"SFD graph render failed for {job_id}: {e}")
            raise HTTPException(status_code=500, detail=f"SFD graph generation failed: {e}")

    # ---- Residual Scatter ----
    elif kind == "residual_scatter":
        cache = output_dir / "graph_residual_scatter.png"
        if cache.exists():
            return FileResponse(str(cache), media_type="image/png")

        matches_path = job_dir / "intermediate" / "matches_refined.npy"
        if not matches_path.exists():
            matches_path = job_dir / "intermediate" / "matches_verified.npy"
        if not matches_path.exists():
            raise HTTPException(status_code=404, detail="Match data not found for residual scatter.")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            matches = _np.load(str(matches_path))
            pts_a = matches[:, :2]
            pts_b = matches[:, 2:4]
            residuals = _np.linalg.norm(pts_a - pts_b, axis=1)

            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            sc = ax.scatter(pts_a[:, 0], pts_a[:, 1], c=residuals, cmap='plasma',
                            s=20, alpha=0.8, vmin=0, vmax=_np.percentile(residuals, 95))
            plt.colorbar(sc, ax=ax, label="Residual (px)").ax.yaxis.label.set_color('white')
            ax.set_xlabel("x (px)", color='white')
            ax.set_ylabel("y (px)", color='white')
            ax.set_title(f"Reprojection Residual Scatter — Job {job_id[:12]}",
                         color='white', fontweight='bold')
            ax.tick_params(colors='white')
            plt.tight_layout()
            plt.savefig(str(cache), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return FileResponse(str(cache), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Residual scatter generation failed: {e}")

    # ---- Residual Histogram ----
    elif kind == "residual_histogram":
        cache = output_dir / "graph_residual_histogram.png"
        if cache.exists():
            return FileResponse(str(cache), media_type="image/png")

        matches_path = job_dir / "intermediate" / "matches_refined.npy"
        if not matches_path.exists():
            matches_path = job_dir / "intermediate" / "matches_verified.npy"
        if not matches_path.exists():
            raise HTTPException(status_code=404, detail="Match data not found for histogram.")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            matches = _np.load(str(matches_path))
            residuals = _np.linalg.norm(matches[:, :2] - matches[:, 2:4], axis=1)
            rmse = float(_np.sqrt(_np.mean(residuals**2)))

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            ax.hist(residuals, bins=30, color='#00d4ff', edgecolor='#1a1a2e', alpha=0.85)
            ax.axvline(rmse, color='#ff6b6b', linestyle='--', linewidth=2,
                       label=f"RMSE = {rmse:.3f}px")
            ax.set_xlabel("Residual Error (px)", color='white')
            ax.set_ylabel("Count", color='white')
            ax.set_title(f"Residual Error Distribution — {len(matches)} inliers",
                         color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1a1a2e', labelcolor='white')
            plt.tight_layout()
            plt.savefig(str(cache), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return FileResponse(str(cache), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Residual histogram failed: {e}")

    # ---- Crater Histogram ----
    elif kind == "crater_histogram":
        cache = output_dir / "graph_crater_histogram.png"
        if cache.exists():
            return FileResponse(str(cache), media_type="image/png")

        craters_path = job_dir / "intermediate" / "craters_a.json"
        if not craters_path.exists():
            raise HTTPException(status_code=404, detail="Crater data not found.")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from core.crater_detection import bucket_diameter_histogram

            with open(craters_path, "r") as f:
                craters = json.load(f)

            buckets = bucket_diameter_histogram(craters)

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            colors = ['#00d4ff', '#4cc9f0', '#7209b7', '#f72585']
            bars = ax.bar(list(buckets.keys()), list(buckets.values()),
                          color=colors, edgecolor='#1a1a2e')
            ax.set_xlabel("Diameter Class", color='white')
            ax.set_ylabel("Count", color='white')
            ax.set_title(f"Crater Diameter Histogram — {len(craters)} craters",
                         color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for bar, val in zip(bars, buckets.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', color='white', fontsize=9)
            plt.tight_layout()
            plt.savefig(str(cache), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return FileResponse(str(cache), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Crater histogram failed: {e}")

    # ---- Confidence Gauge ----
    elif kind == "confidence_gauge":
        cache = output_dir / "graph_confidence_gauge.png"
        if cache.exists():
            return FileResponse(str(cache), media_type="image/png")

        metrics_path = job_dir / "output" / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Metrics not found for confidence gauge.")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            rmse = metrics.get("rmse_px", 999)
            sdi = metrics.get("sdi", 0)
            inlier_ratio = metrics.get("inlier_ratio", 0)
            # Compute 0-100 confidence score
            score = min(100, max(0,
                100 - min(rmse, 5) * 15 + sdi * 20 + inlier_ratio * 20
            ))

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#16213e")
            theta = _np.linspace(0, _np.pi, 200)
            ax.plot(theta, [1]*200, color='#2d2d4e', linewidth=30)
            fill_theta = _np.linspace(0, _np.pi * score / 100, 200)
            color = '#ff6b6b' if score < 40 else '#ffdd57' if score < 70 else '#51cf66'
            ax.plot(fill_theta, [1]*len(fill_theta), color=color, linewidth=30)
            ax.set_ylim(0, 1.5)
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(-1)
            ax.axis('off')
            ax.text(0, 0.25, f"{score:.0f}",
                    ha='center', va='center', fontsize=36, color='white', fontweight='bold',
                    transform=ax.transData)
            ax.set_title(f"Registration Confidence\nRMSE={rmse:.3f}px | SDI={sdi:.2f}",
                         color='white', fontsize=10, pad=20)
            plt.tight_layout()
            plt.savefig(str(cache), dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return FileResponse(str(cache), media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Confidence gauge failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unhandled graph kind: {kind}")

