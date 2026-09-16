"""
tests/test_agent.py
===================
Tests for the AI Agent explanation layer and research companion endpoints in api/main.py.
"""

import os
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app, explain_sessions, _rag_prefetch_cache, research_sessions
from core.agent_rag import retrieve


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def completed_job_dir():
    job_id = "test_agent_job_done_fixture_001"
    job_dir = Path("data") / "jobs" / job_id
    output_dir = job_dir / "output"
    input_dir = job_dir / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    status_data = {
        "job_id": job_id,
        "status": "DONE",
        "elapsed_s": 0.85,
    }
    with open(job_dir / "status.json", "w") as f:
        json.dump(status_data, f)

    metrics_data = {
        "status": "DONE",
        "rmse_px": 0.3284,
        "inlier_ratio": 0.625,
        "sdi": 0.68,
        "n_inliers": 25,
        "n_total": 40,
        "elapsed_s": 0.85,
        "transform_type": "homography",
        "condition_number": 42.5,
        "ssim": 0.72,
        "ncc": 0.81,
        "mae_px": 0.28,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f)

    meta_data = {
        "img_a_path": "fake_a.tif",
        "img_b_path": "fake_b.tif",
        "scale_disparity_ratio": 1.2,
        "solar_correction_applied": True,
    }
    with open(input_dir / "metadata.json", "w") as f:
        json.dump(meta_data, f)

    yield job_id

    shutil.rmtree(job_dir, ignore_errors=True)
    explain_sessions.pop(job_id, None)
    _rag_prefetch_cache.pop(job_id, None)


def test_fast_path_rmse_query(client, completed_job_dir):
    # Start explain session
    resp_start = client.post(f"/agent/explain/{completed_job_dir}/start")
    assert resp_start.status_code == 200

    # Query RMSE
    resp = client.post(
        f"/agent/explain/{completed_job_dir}/message",
        json={"query": "what is the rmse"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["used_fast_path"] is True
    assert "rmse" in data["text_response"].lower()
    assert data["latency_ms"] < 100


def test_fast_path_grade_query(client, completed_job_dir):
    client.post(f"/agent/explain/{completed_job_dir}/start")
    resp = client.post(
        f"/agent/explain/{completed_job_dir}/message",
        json={"query": "is this reliable"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["used_fast_path"] is True
    assert "grade" in data["text_response"].lower()


def test_generative_path_fires_without_groq_key(client, completed_job_dir):
    client.post(f"/agent/explain/{completed_job_dir}/start")
    with patch.dict(os.environ, {}, clear=True):
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]
        resp = client.post(
            f"/agent/explain/{completed_job_dir}/message",
            json={"query": "describe the crater distribution"}
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data.get("error_code") == "GROQ_API_KEY_MISSING"


def test_research_session_creates_unique_ids(client):
    r1 = client.post("/research/session")
    r2 = client.post("/research/session")
    assert r1.status_code == 200
    assert r2.status_code == 200
    sid1 = r1.json()["session_id"]
    sid2 = r2.json()["session_id"]
    assert sid1 != sid2


def test_research_session_not_found_returns_404(client):
    resp = client.post(
        "/research/nonexistent-session-id-9999/message",
        json={"query": "what is lunar crater morphology"}
    )
    assert resp.status_code == 404


def test_rag_retrieve_returns_top_k():
    results = retrieve("RMSE inlier ratio", top_k=4)
    assert len(results) == 4
    for r in results:
        assert "text" in r
        assert "source_file" in r
        assert "section_title" in r


def test_explain_start_job_not_found(client):
    resp = client.post("/agent/explain/bad_job/start")
    assert resp.status_code == 404
    data = resp.json()
    assert data.get("error_code") == "JOB_NOT_FOUND"


def test_history_capped_at_10_turns(client, completed_job_dir):
    client.post(f"/agent/explain/{completed_job_dir}/start")
    for i in range(12):
        resp = client.post(
            f"/agent/explain/{completed_job_dir}/message",
            json={"query": f"what is the rmse turn {i}"}
        )
        assert resp.status_code == 200

    history = explain_sessions[completed_job_dir]["history"]
    assert len(history) <= 20


def test_explain_query_bypasses_fast_path(client, completed_job_dir):
    client.post(f"/agent/explain/{completed_job_dir}/start")
    resp = client.post(
        f"/agent/explain/{completed_job_dir}/message",
        json={"query": "explain why the pipeline chose homography instead of TPS"}
    )
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("used_fast_path") is False
    else:
        assert resp.status_code == 503
        data = resp.json()
        assert data.get("error_code") == "GROQ_API_KEY_MISSING"


def test_why_query_bypasses_fast_path(client, completed_job_dir):
    client.post(f"/agent/explain/{completed_job_dir}/start")
    resp = client.post(
        f"/agent/explain/{completed_job_dir}/message",
        json={"query": "why is the inlier ratio important"}
    )
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("used_fast_path") is False
    else:
        assert resp.status_code == 503
        data = resp.json()
        assert data.get("error_code") == "GROQ_API_KEY_MISSING"
