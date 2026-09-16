"""
tests/test_overlap_check.py
============================
Unit and integration tests for Stage 0 Footprint Overlap Pre-Check (core/overlap_check.py).

Covers:
  1. Geospatial O(1) bounding-box IoU computation (disjoint, partial, identical).
  2. Downsampled NCC peak location and magnitude proxy for non-georeferenced pairs.
  3. overlap_gate threshold enforcement and OverlapTooLowError.
  4. End-to-end fast failure in LunaMatchPipeline: non-overlapping pair fails fast
     with OVERLAP_TOO_LOW without running Phase Congruency FFT (Stage 2).
"""

import os
import shutil
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytest

from core.overlap_check import (
    estimate_overlap,
    overlap_gate,
    OverlapTooLowError,
    _bounding_box_iou,
)
from pipeline.orchestrator import LunaMatchPipeline, JobState


@dataclass
class DummyAffine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    is_identity: bool = False

    def __mul__(self, pt):
        return (self.c + self.a * pt[0] + self.b * pt[1],
                self.f + self.d * pt[0] + self.e * pt[1])


@dataclass
class DummyMetadata:
    crs: str = "EPSG:32631"
    transform: Any = None
    shape: tuple = (100, 100)


def test_bounding_box_iou_exact():
    # Box A: [0, 0, 10, 10] -> area 100
    # Box B: [5, 0, 15, 10] -> area 100, intersection [5, 0, 10, 10] -> area 50
    # Union = 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    box_a = (0.0, 0.0, 10.0, 10.0)
    box_b = (5.0, 0.0, 15.0, 10.0)
    iou = _bounding_box_iou(box_a, box_b)
    assert abs(iou - 1.0 / 3.0) < 1e-5

    # Completely disjoint
    box_c = (20.0, 20.0, 30.0, 30.0)
    assert _bounding_box_iou(box_a, box_c) == 0.0


def test_geospatial_overlap_disjoint():
    img_a = np.zeros((100, 100))
    img_b = np.zeros((100, 100))

    # Raster A at (0, 0), Raster B at (1000, 1000)
    meta_a = DummyMetadata(transform=DummyAffine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0))
    meta_b = DummyMetadata(transform=DummyAffine(1.0, 0.0, 1000.0, 0.0, -1.0, 1100.0))

    overlap = estimate_overlap(img_a, img_b, meta_a, meta_b)
    assert overlap == 0.0

    with pytest.raises(OverlapTooLowError) as exc_info:
        overlap_gate(overlap, min_required=0.15)
    assert "OVERLAP_TOO_LOW" in str(exc_info.value)


def test_ncc_overlap_identical_and_disjoint():
    rng = np.random.default_rng(42)
    img_a = rng.uniform(0, 255, (128, 128))

    # Identical images without georeferencing
    overlap_self = estimate_overlap(img_a, img_a, None, None)
    assert overlap_self > 0.90
    overlap_gate(overlap_self, min_required=0.15)  # must not raise

    # Unrelated independent random noise images
    img_unrelated = rng.uniform(0, 255, (128, 128))
    overlap_noise = estimate_overlap(img_a, img_unrelated, None, None)
    assert overlap_noise < 0.15

    with pytest.raises(OverlapTooLowError):
        overlap_gate(overlap_noise, min_required=0.15)


def test_pipeline_rejects_synthetic_non_overlapping_pair():
    """
    End-to-end integration test: LunaMatchPipeline must reject a synthetic
    non-overlapping pair in Stage 0 before Stage 2 (Phase Congruency FFT) runs.
    """
    job_id = f"test_overlap_reject_{uuid.uuid4().hex[:8]}"
    job_dir = Path("data") / "jobs" / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    try:
        rng = np.random.default_rng(999)
        # Create two completely independent noise fields
        img_a = rng.normal(128, 20, (128, 128)).clip(0, 255)
        img_b = rng.normal(50, 10, (128, 128)).clip(0, 255)

        path_a = str(input_dir / "img_a.npy")
        path_b = str(input_dir / "img_b.npy")
        np.save(path_a, img_a)
        np.save(path_b, img_b)

        pipeline = LunaMatchPipeline(job_id, path_a, path_b)
        result = pipeline.run()

        # Check that pipeline aborted with OVERLAP_TOO_LOW
        assert result["status"] == "FAILED"
        assert result.get("error_code") == "OVERLAP_TOO_LOW"
        assert pipeline.get_status() == JobState.FAILED

        # Verify Stage 2 Phase Congruency was NEVER executed (no pc_map artifact created)
        pc_map_path = pipeline.paths["pc_map_a"]
        assert not pc_map_path.exists(), "Phase congruency should not execute when overlap check fails!"

    finally:
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
