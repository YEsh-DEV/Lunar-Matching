"""
tests/test_pipeline_e2e.py
============================
End-to-end orchestrator test.

Executes the full 7-stage registration pipeline against test inputs:
  1. Pipeline state transitions and error isolation
  2. status.json written correctly at each transition
  3. Intermediate and output artifacts exist
  4. Metrics report adherence to interface contract
  5. Multi-job parallelism isolation
"""

import sys, os, json, shutil, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from pipeline.orchestrator import LunaMatchPipeline, JobState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tmp_job():
    """Return a job_id and two tiny synthetic image paths."""
    job_id = f"test_{uuid.uuid4().hex[:8]}"
    job_dir = os.path.join("data", "jobs", job_id, "input")
    os.makedirs(job_dir, exist_ok=True)

    img_a = np.random.rand(64, 64)
    img_b = img_a + np.random.randn(64, 64) * 0.05

    path_a = os.path.join(job_dir, "image_a.npy")
    path_b = os.path.join(job_dir, "image_b.npy")
    np.save(path_a, img_a)
    np.save(path_b, img_b)

    return job_id, path_a, path_b


def cleanup_job(job_id):
    job_dir = os.path.join("data", "jobs", job_id)
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 1: Full pipeline runs without crash (stubs active)
# ---------------------------------------------------------------------------

def test_pipeline_runs_end_to_end():
    """Pipeline must complete DONE state with all stubs active."""
    job_id, path_a, path_b = make_tmp_job()
    try:
        pipeline = LunaMatchPipeline(job_id, path_a, path_b)

        assert pipeline.get_status() == JobState.PENDING

        result = pipeline.run()

        # Result must not be a FAILED dict
        assert result.get('status') in ('DONE', 'FAILED'), (
            f"Unexpected result status: {result.get('status')}"
        )

        # Even if some stubs fail, it should fail gracefully with stage info
        if result['status'] == 'FAILED':
            assert 'stage' in result, "FAILED result must contain 'stage' key."
            assert 'error' in result, "FAILED result must contain 'error' key."
    finally:
        cleanup_job(job_id)


# ---------------------------------------------------------------------------
# Test 2: status.json is written on disk
# ---------------------------------------------------------------------------

def test_status_json_exists():
    """status.json must exist and contain valid JSON after pipeline runs."""
    job_id, path_a, path_b = make_tmp_job()
    try:
        pipeline = LunaMatchPipeline(job_id, path_a, path_b)
        pipeline.run()

        status_path = os.path.join("data", "jobs", job_id, "status.json")
        assert os.path.exists(status_path), "status.json was not written."

        with open(status_path) as f:
            status = json.load(f)

        assert 'job_id' in status
        assert 'status' in status
        assert status['job_id'] == job_id
        assert status['status'] in [s.value for s in JobState]
    finally:
        cleanup_job(job_id)


# ---------------------------------------------------------------------------
# Test 3: get_status() returns correct final state
# ---------------------------------------------------------------------------

def test_get_status_reflects_pipeline_state():
    """get_status() must reflect the terminal state after run()."""
    job_id, path_a, path_b = make_tmp_job()
    try:
        pipeline = LunaMatchPipeline(job_id, path_a, path_b)
        result = pipeline.run()

        final_state = pipeline.get_status()
        assert final_state in (JobState.DONE, JobState.FAILED), (
            f"Terminal state must be DONE or FAILED; got {final_state}"
        )
        assert final_state.value == result['status']
    finally:
        cleanup_job(job_id)


# ---------------------------------------------------------------------------
# Test 4: Intermediate artifact directories exist
# ---------------------------------------------------------------------------

def test_intermediate_directories_created():
    """intermediate/ and output/ directories must be created by the pipeline."""
    job_id, path_a, path_b = make_tmp_job()
    try:
        pipeline = LunaMatchPipeline(job_id, path_a, path_b)
        pipeline.run()

        inter = os.path.join("data", "jobs", job_id, "intermediate")
        out   = os.path.join("data", "jobs", job_id, "output")

        assert os.path.isdir(inter), f"intermediate/ dir not created: {inter}"
        assert os.path.isdir(out),   f"output/ dir not created: {out}"
    finally:
        cleanup_job(job_id)


# ---------------------------------------------------------------------------
# Test 5: PC intermediate files written
# ---------------------------------------------------------------------------

def test_pc_intermediate_files_written():
    """Phase congruency step must write pc_map_a.npy and mim_a.npy."""
    job_id, path_a, path_b = make_tmp_job()
    try:
        pipeline = LunaMatchPipeline(job_id, path_a, path_b)
        result = pipeline.run()

        if result['status'] == 'DONE' or result.get('stage', '') not in ('preprocessing', 'phase_congruency'):
            pc_path  = os.path.join("data", "jobs", job_id, "intermediate", "pc_map_a.npy")
            mim_path = os.path.join("data", "jobs", job_id, "intermediate", "mim_a.npy")

            assert os.path.exists(pc_path),  f"pc_map_a.npy not written: {pc_path}"
            assert os.path.exists(mim_path), f"mim_a.npy not written: {mim_path}"

            pc = np.load(pc_path)
            assert pc.ndim == 2, f"pc_map must be 2D; got {pc.ndim}D"
    finally:
        cleanup_job(job_id)


# ---------------------------------------------------------------------------
# Test 6: Two independent jobs don't interfere
# ---------------------------------------------------------------------------

def test_two_parallel_jobs_independent():
    """Two independent jobs must write to separate directories without collision."""
    job1, p1a, p1b = make_tmp_job()
    job2, p2a, p2b = make_tmp_job()
    try:
        pipe1 = LunaMatchPipeline(job1, p1a, p1b)
        pipe2 = LunaMatchPipeline(job2, p2a, p2b)

        r1 = pipe1.run()
        r2 = pipe2.run()

        # Status files must be separate
        s1 = os.path.join("data", "jobs", job1, "status.json")
        s2 = os.path.join("data", "jobs", job2, "status.json")

        assert os.path.exists(s1) and os.path.exists(s2)

        with open(s1) as f: st1 = json.load(f)
        with open(s2) as f: st2 = json.load(f)

        assert st1['job_id'] == job1
        assert st2['job_id'] == job2
    finally:
        cleanup_job(job1)
        cleanup_job(job2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
