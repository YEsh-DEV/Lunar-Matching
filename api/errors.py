"""
api/errors.py
=============
Structured error taxonomy for the LUNA-MATCH Planetary Image Registration Workbench.
"""

from enum import Enum
from typing import Optional, Dict, Any
from fastapi.responses import JSONResponse


class ErrorCode(str, Enum):
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_DONE = "JOB_NOT_DONE"
    OVERLAP_TOO_LOW = "OVERLAP_TOO_LOW"
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    GEOMETRIC_DEGENERACY = "GEOMETRIC_DEGENERACY"
    ILL_CONDITIONED_UNRECOVERABLE = "ILL_CONDITIONED_UNRECOVERABLE"
    LOFTR_UNAVAILABLE = "LOFTR_UNAVAILABLE"
    INVALID_INPUT_PATH = "INVALID_INPUT_PATH"
    INVALID_KIND = "INVALID_KIND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIError(Exception):
    """
    Structured API Exception conforming to LUNA-MATCH error contract:
    {"error_code": str, "message": str, "job_id": str|null}
    """
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        job_id: Optional[str] = None,
    ):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.job_id = job_id
        super().__init__(message)


def format_error_payload(
    error_code: ErrorCode,
    message: str,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Format standard non-2xx error dictionary."""
    code_str = error_code.value if isinstance(error_code, ErrorCode) else str(error_code)
    return {
        "error_code": code_str,
        "message": message,
        "job_id": job_id,
    }


def api_error_response(
    status_code: int,
    error_code: ErrorCode,
    message: str,
    job_id: Optional[str] = None,
) -> JSONResponse:
    """Generate FastAPI JSONResponse conforming to LUNA-MATCH error contract."""
    return JSONResponse(
        status_code=status_code,
        content=format_error_payload(error_code, message, job_id),
    )
