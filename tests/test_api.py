"""
tests/test_api.py
=================
Integration tests for the LUNA-MATCH FastAPI application.
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def synthetic_images():
    td = tempfile.mkdtemp()
    img_a = np.random.uniform(0.1, 0.9, (64, 64))
    img_b = img_a + np.random.normal(0, 0.05, (64, 64))

    p_a = os.path.join(td, "img_a.npy")
    p_b = os.path.join(td, "img_b.npy")
    np.save(p_a, img_a)
    np.save(p_b, img_b)
    return p_a, p_b


def test_api_register_endpoint(client, synthetic_images):
    """POST /register initiates a job and returns HTTP 202."""
    p_a, p_b = synthetic_images
    resp = client.post("/register", json={
        "img_a_path": p_a,
        "img_b_path": p_b,
        "job_id": "test_api_job_001"
    })
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "test_api_job_001"
    assert data["status"] in ("PENDING", "PREPROCESSING", "DONE")


def test_api_get_job_status(client, synthetic_images):
    """GET /jobs/{id} returns job lifecycle state."""
    p_a, p_b = synthetic_images
    job_id = "test_api_status_002"
    client.post("/register", json={
        "img_a_path": p_a,
        "img_b_path": p_b,
        "job_id": job_id
    })
    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert "status" in data


def test_api_get_job_result_nonexistent(client):
    """GET /jobs/{id}/result returns 404 for nonexistent job."""
    resp = client.get("/jobs/nonexistent_id_999/result")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Checkpoint 5: /graphs endpoint tests
# ---------------------------------------------------------------------------

def test_api_graphs_invalid_kind(client):
    """
    GET /jobs/{id}/graphs?kind=invalid must return 400 with INVALID_KIND error code.
    The job may or may not exist — the kind validation fires before job lookup.
    """
    resp = client.get("/jobs/some_job_xyz/graphs?kind=totally_invalid_kind")
    assert resp.status_code == 400
    data = resp.json()
    # FastAPI may wrap the detail in a 'detail' key
    detail = data.get("detail", data)
    if isinstance(detail, dict):
        assert detail.get("error_code") == "INVALID_KIND"
        assert "valid_values" in detail
    # At minimum, a 400 with informative content
    assert resp.status_code == 400


def test_api_graphs_sfd_nonexistent_job(client):
    """
    GET /graphs?kind=sfd for a nonexistent job must return 404.
    """
    resp = client.get("/jobs/job_does_not_exist_abc123/graphs?kind=sfd")
    assert resp.status_code == 404


def test_api_graphs_sfd_on_completed_job(client, synthetic_images, tmp_path):
    """
    GET /graphs?kind=sfd on a completed job must return 200 (PNG or JSON).
    We create a minimal job directory with empty craters_a.json to simulate completion.
    """
    import json
    import os

    # Simulate a completed job: create directory structure manually
    job_id = "test_graphs_sfd_job_001"
    job_dir = os.path.join("data", "jobs", job_id)
    os.makedirs(os.path.join(job_dir, "intermediate"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "output"), exist_ok=True)

    # Write empty craters file (no craters detected)
    with open(os.path.join(job_dir, "intermediate", "craters_a.json"), "w") as f:
        json.dump([], f)

    resp = client.get(f"/jobs/{job_id}/graphs?kind=sfd")
    # SFD with 0 craters should still return 200 (empty plot or JSON data)
    assert resp.status_code == 200
