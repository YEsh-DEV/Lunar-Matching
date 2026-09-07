"""
api/schemas.py
==============
Pydantic schemas for the LUNA-MATCH Lunar Image Registration API.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Payload to initiate a new registration pipeline run."""
    img_a_path: str = Field(..., description="Absolute path or URI to moving/source image (Chandrayaan-2 OHRC/TMC-2/IIRS)")
    img_b_path: str = Field(..., description="Absolute path or URI to reference image (LRO NAC / SELENE)")
    job_id: Optional[str] = Field(None, description="Optional custom job identifier; UUID generated if omitted")


class JobStatusResponse(BaseModel):
    """Current state and execution progress of a job."""
    job_id: str
    status: str
    elapsed_s: Optional[float] = None
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    """Final metrics and registered artifact paths upon completion."""
    job_id: str
    status: str
    rmse_px: Optional[float] = None
    inlier_ratio: Optional[float] = None
    sdi: Optional[float] = None
    n_inliers: Optional[int] = None
    n_total: Optional[int] = None
    elapsed_s: Optional[float] = None
    transform: Optional[str] = None
    error: Optional[str] = None
    output_files: Optional[Dict[str, str]] = None
