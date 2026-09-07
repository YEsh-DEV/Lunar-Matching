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
