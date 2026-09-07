"""
api/main.py
===========
FastAPI application for LUNA-MATCH Planetary Image Registration Workbench.

Exposes 4 core endpoints:
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
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pipeline.orchestrator import LunaMatchPipeline, JobState
from api.schemas import RegisterRequest, JobStatusResponse, JobResultResponse

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


@app.get("/jobs/{job_id}/preview")
def get_job_preview(job_id: str, kind: str = "registered"):
    """
    Serve preview images for frontend visualization:
      - kind='registered' : the warped registered GeoTIFF/image
      - kind='residual'   : the 2D residual error heatmap
    """
    job_dir = Path("data") / "jobs" / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if kind == "residual":
        preview_path = job_dir / "output" / "residual_map.png"
        if preview_path.exists():
            return FileResponse(str(preview_path), media_type="image/png")
        raise HTTPException(status_code=404, detail="Residual map preview not yet generated.")

    # Default: registered output
    tif_path = job_dir / "output" / "registered.tif"
    if tif_path.exists():
        return FileResponse(str(tif_path), media_type="image/tiff")

    npy_path = job_dir / "output" / "registered.npy"
    if npy_path.exists():
        return FileResponse(str(npy_path), media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail="Registration output preview not found.")
