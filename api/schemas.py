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


class QualityAssessmentSchema(BaseModel):
    confidence_label: str
    grade: str
    reasoning: str
    warnings: List[str] = []


class MetricsSchema(BaseModel):
    rmse_px: Optional[float] = None
    inlier_ratio: Optional[float] = None
    sdi: Optional[float] = None
    transform_type: Optional[str] = None
    n_inliers: int = 0
    n_total: int = 0
    elapsed_s: float = 0.0


class InputMetadataSchema(BaseModel):
    img_a_path: Optional[str] = None
    img_b_path: Optional[str] = None
    gsd_a_m_per_px: float = 1.0
    gsd_b_m_per_px: float = 1.0
    scale_disparity_ratio: float = 1.0
    pixel_dimension_ratio: float = 1.0
    solar_correction_applied: bool = False


class ArtifactsSchema(BaseModel):
    registered_geotiff: Optional[str] = None
    checkerboard_png: Optional[str] = None
    tiepoints_png: Optional[str] = None
    residual_map_png: Optional[str] = None


class ChatbotSummaryResponse(BaseModel):
    """Unified Chatbot/VLM/RAG payload conforming to Schema v1.1."""
    schema_version: str = "1.1"
    job_id: str
    status: str
    quality_assessment: QualityAssessmentSchema
    metrics: MetricsSchema
    input_metadata: InputMetadataSchema
    artifacts: ArtifactsSchema

