"""
tests/test_summary_builder.py
==============================
Unit and integration tests for core/summary_builder.py and the
GET /jobs/{job_id}/summary endpoint in api/main.py.

Tests cover:
  1. Chatbot summary schema contract shape & schema_version
  2. Confidence label & grade threshold evaluations (high, moderate, low, failed, synthetic fallback)
  3. Robust handling of missing artifacts (none crash, gracefully returned as None)
  4. Non-existent job raises FileNotFoundError
  5. FastAPI endpoint integration (HTTP 200 and HTTP 404 responses)
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from core.summary_builder import build_chatbot_summary, classify_registration_quality, SCHEMA_VERSION
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def tmp_job_dir():
    """Create a temporary job directory structure."""
    td = tempfile.mkdtemp()
    job_id = "test_summary_job_001"
    job_dir = Path(td) / job_id
    (job_dir / "input").mkdir(parents=True)
    (job_dir / "intermediate").mkdir(parents=True)
    (job_dir / "output").mkdir(parents=True)

    # status.json
    with open(job_dir / "status.json", "w") as f:
        json.dump({"job_id": job_id, "status": "DONE", "elapsed_s": 2.5}, f)

    # metrics.json
    with open(job_dir / "output" / "metrics.json", "w") as f:
        json.dump({
            "status": "DONE",
            "rmse_px": 0.4215,
            "inlier_ratio": 0.852,
            "sdi": 0.6543,
            "n_inliers": 14,
            "n_total": 20,
            "elapsed_s": 2.5,
            "transform": "homography",
            "is_synthetic_fallback": False,
        }, f)

    # input_paths.json
    with open(job_dir / "input" / "input_paths.json", "w") as f:
        json.dump({
            "img_a_path": "data/samples/image_1.tif",
            "img_b_path": "data/samples/image_2.tif",
        }, f)

    # metadata.json
    with open(job_dir / "input" / "metadata.json", "w") as f:
        json.dump({
            "img_a_path": "data/samples/image_1.tif",
            "img_b_path": "data/samples/image_2.tif",
            "image_a": {"gsd": "131.6", "incidence_angle": "45.0", "emission_angle": "5.0", "phase_angle": "40.0"},
            "image_b": {"gsd": "555.6", "incidence_angle": "45.0", "emission_angle": "5.0", "phase_angle": "40.0"},
        }, f)

    # Create dummy artifact files
    (job_dir / "output" / "registered.tif").touch()
    (job_dir / "output" / "residual_map.png").touch()
    (job_dir / "output" / "preview_checkerboard.png").touch()
    (job_dir / "output" / "preview_tiepoints.png").touch()

    yield td, job_id

    shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 1: Contract Shape & Schema Version
# ---------------------------------------------------------------------------

def test_summary_contract_shape(tmp_job_dir):
    td, job_id = tmp_job_dir
    summary = build_chatbot_summary(job_id, base_jobs_dir=td)

    assert summary["schema_version"] == "1.1"
    assert summary["job_id"] == job_id
    assert summary["status"] == "DONE"

    # Quality Assessment block
    qa = summary["quality_assessment"]
    assert "confidence_label" in qa
    assert "grade" in qa
    assert "reasoning" in qa
    assert isinstance(qa["warnings"], list)

    # Metrics block
    m = summary["metrics"]
    assert m["rmse_px"] == 0.4215
    assert m["inlier_ratio"] == 0.852
    assert m["sdi"] == 0.6543
    assert m["n_inliers"] == 14
    assert m["n_total"] == 20
    assert m["transform_type"] == "homography"
    assert m["elapsed_s"] == 2.5

    # Input Metadata block
    im = summary["input_metadata"]
    assert im["gsd_a_m_per_px"] == 131.6
    assert im["gsd_b_m_per_px"] == 555.6
    assert round(im["scale_disparity_ratio"], 2) == 4.22
    assert "pixel_dimension_ratio" in im
    assert im["pixel_dimension_ratio"] >= 1.0
    assert im["solar_correction_applied"] is True

    # Artifacts block
    art = summary["artifacts"]
    assert art["registered_geotiff"] is not None
    assert art["residual_map_png"] is not None
    assert art["checkerboard_png"] is not None
    assert art["tiepoints_png"] is not None


# ---------------------------------------------------------------------------
# Test 2: Confidence Label & Grade Thresholds
# ---------------------------------------------------------------------------

def test_confidence_thresholds():
    # 1. High Confidence
    metrics_high = {
        "status": "DONE",
        "rmse_px": 0.35,
        "inlier_ratio": 0.45,
        "n_inliers": 12,
        "sdi": 0.50,
        "transform": "homography",
        "is_synthetic_fallback": False,
    }
    meta_high = {"scale_disparity_ratio": 1.5, "solar_correction_applied": True}
    label, grade, reason, warnings = classify_registration_quality(metrics_high, meta_high)
    assert label == "high confidence"
    assert grade == "A"
    assert len(warnings) == 0

    # 2. Moderate Confidence
    metrics_mod = {
        "status": "DONE",
        "rmse_px": 1.15,
        "inlier_ratio": 0.22,
        "n_inliers": 6,
        "sdi": 0.35,
        "transform": "homography",
        "is_synthetic_fallback": False,
    }
    meta_mod = {"scale_disparity_ratio": 1.2, "solar_correction_applied": False}
    label, grade, reason, warnings = classify_registration_quality(metrics_mod, meta_mod)
    assert label == "moderate confidence"
    assert grade == "B"
    assert any("Solar ephemeris" in w for w in warnings)

    # 3. Low Confidence (High Scale Disparity)
    metrics_low_scale = {
        "status": "DONE",
        "rmse_px": 1.80,
        "inlier_ratio": 0.12,
        "n_inliers": 4,
        "sdi": 0.22,
        "transform": "homography",
        "is_synthetic_fallback": False,
    }
    meta_low_scale = {"scale_disparity_ratio": 4.5, "solar_correction_applied": False}
    label, grade, reason, warnings = classify_registration_quality(metrics_low_scale, meta_low_scale)
    assert "scale disparity may exceed matcher capability" in label
    assert grade == "C"

    # 4. Low Confidence (High Reprojection Error)
    metrics_low_rmse = {
        "status": "DONE",
        "rmse_px": 2.85,
        "inlier_ratio": 0.35,
        "n_inliers": 12,
        "sdi": 0.40,
        "transform": "homography",
        "is_synthetic_fallback": False,
    }
    meta_low_rmse = {"scale_disparity_ratio": 1.0, "solar_correction_applied": True}
    label, grade, reason, warnings = classify_registration_quality(metrics_low_rmse, meta_low_rmse)
    assert "high reprojection residual error" in label
    assert grade == "C"

    # 5. Synthetic Fallback Active
    metrics_fallback = {
        "status": "DONE",
        "rmse_px": 0.5,
        "inlier_ratio": 0.9,
        "n_inliers": 20,
        "sdi": 0.8,
        "transform": "homography",
        "is_synthetic_fallback": True,
    }
    label, grade, reason, warnings = classify_registration_quality(metrics_fallback, meta_high)
    assert "synthetic fallback active" in label
    assert grade == "D"

    # 6. Failed Job
    metrics_failed = {
        "status": "FAILED",
        "stage": "matching",
        "error": "insufficient real matches: got 3, need >= 8",
    }
    label, grade, reason, warnings = classify_registration_quality(metrics_failed, meta_high)
    assert label == "failed"
    assert grade == "F"


# ---------------------------------------------------------------------------
# Test 3: Missing Artifact Handling
# ---------------------------------------------------------------------------

def test_missing_artifact_handling():
    """Summary builder must not crash if visual artifact files are absent."""
    td = tempfile.mkdtemp()
    job_id = "test_empty_artifacts_job"
    job_dir = Path(td) / job_id
    (job_dir / "output").mkdir(parents=True)
    (job_dir / "input").mkdir(parents=True)
    (job_dir / "intermediate").mkdir(parents=True)

    with open(job_dir / "status.json", "w") as f:
        json.dump({"job_id": job_id, "status": "DONE"}, f)
    with open(job_dir / "output" / "metrics.json", "w") as f:
        json.dump({"status": "DONE", "rmse_px": 1.0, "n_inliers": 5, "n_total": 10}, f)

    try:
        summary = build_chatbot_summary(job_id, base_jobs_dir=td)
        assert summary["status"] == "DONE"
        assert summary["artifacts"]["registered_geotiff"] is None
        assert summary["artifacts"]["residual_map_png"] is None
        assert summary["artifacts"]["checkerboard_png"] is None
        assert summary["artifacts"]["tiepoints_png"] is None
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Non-existent Job Raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_nonexistent_job_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        build_chatbot_summary("completely_nonexistent_job_xyz_99999")


# ---------------------------------------------------------------------------
# Test 5: API GET /jobs/{job_id}/summary Endpoint
# ---------------------------------------------------------------------------

def test_api_get_job_summary_endpoint(client):
    # Test 404 for missing job
    res_404 = client.get("/jobs/missing_job_abc_123/summary")
    assert res_404.status_code == 404

    # Test real bench job if exists
    if Path("data/jobs/bench_pair_2").exists():
        res_200 = client.get("/jobs/bench_pair_2/summary")
        assert res_200.status_code == 200
        data = res_200.json()
        assert data["schema_version"] == "1.1"
        assert data["job_id"] == "bench_pair_2"
        assert data["quality_assessment"]["confidence_label"] == "moderate confidence"
        assert data["input_metadata"]["pixel_dimension_ratio"] == 1.11
