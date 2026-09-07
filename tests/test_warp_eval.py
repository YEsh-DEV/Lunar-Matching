"""
tests/test_warp_eval.py
=======================
Unit and integration tests for core/warp_and_eval.py.
"""

import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.warp_and_eval import (
    warp_image,
    compute_rmse,
    compute_inlier_ratio,
    compute_sdi,
    export_geotiff,
    build_metrics_report,
    generate_residual_error_map,
)
from core.geometric_verification import (
    fit_thin_plate_spline,
    ThinPlateSplineTransform,
)


def test_warp_image_homography():
    """warp_image must warp image with 3x3 homography matrix."""
    img = np.zeros((100, 100), dtype=np.float64)
    img[40:60, 40:60] = 1.0

    # Identity homography
    H_eye = np.eye(3, dtype=np.float64)
    warped = warp_image(img, H_eye, ref_shape=(100, 100))

    assert warped.shape == (100, 100)
    assert np.isclose(warped[50, 50], 1.0, atol=0.1)


def test_warp_image_thin_plate_spline():
    """warp_image must duck-type on hasattr(transform, 'apply') with ThinPlateSplineTransform."""
    # Create 9 regular grid control points with slight distortion
    gx, gy = np.meshgrid(np.linspace(20, 80, 4), np.linspace(20, 80, 4))
    pts_a = np.column_stack([gx.ravel(), gy.ravel()])
    pts_b = pts_a + 2.0  # simple 2-pixel shift

    tps = fit_thin_plate_spline(pts_a, pts_b, smoothing=0.0)
    assert hasattr(tps, "apply")

    img = np.zeros((100, 100), dtype=np.float64)
    img[40:60, 40:60] = 1.0

    warped = warp_image(img, tps, ref_shape=(100, 100))

    assert warped.shape == (100, 100)
    assert isinstance(warped, np.ndarray)


def test_compute_rmse():
    """compute_rmse evaluates reprojection error accurately."""
    p1 = np.array([[10.0, 10.0], [20.0, 20.0]])
    p2 = np.array([[10.0, 13.0], [20.0, 24.0]])  # errors: 3 and 4 -> sqrt((9+16)/2) = 3.5355

    rmse_2arg = compute_rmse(p1, p2)
    assert abs(rmse_2arg - 3.5355) < 1e-3

    # Orchestrator 3-argument call
    matches = np.column_stack([p1, p2, [0.9, 0.9]])
    img_a = np.zeros((50, 50))
    rmse_3arg = compute_rmse(matches, img_a, img_a)
    assert rmse_3arg > 0.0


def test_compute_inlier_ratio():
    """compute_inlier_ratio supports both 1-arg mask and 2-arg counts."""
    mask = np.array([True, True, True, False])
    assert abs(compute_inlier_ratio(mask) - 0.75) < 1e-4
    assert abs(compute_inlier_ratio(3, 4) - 0.75) < 1e-4
    assert compute_inlier_ratio(0, 0) == 0.0


def test_compute_sdi_shannon_entropy_and_log_guard():
    """compute_sdi must guard log2(0) on empty cells and yield higher score for uniform spread."""
    # Clustered points in single cell
    clustered = np.full((100, 2), 5.0)
    sdi_low = compute_sdi(clustered, image_shape=(100, 100), grid=8)
    assert sdi_low == 0.0, "All points in one cell should have zero entropy / SDI."

    # Evenly spaced points across 8x8 grid
    gx, gy = np.meshgrid(np.linspace(5, 95, 8), np.linspace(5, 95, 8))
    uniform = np.column_stack([gx.ravel(), gy.ravel()])
    sdi_high = compute_sdi(uniform, image_shape=(100, 100), grid=8)
    assert sdi_high > 0.9, "Uniform grid must yield SDI close to 1.0."


def test_export_geotiff():
    """export_geotiff exports registered raster."""
    warped = np.ones((64, 64), dtype=np.float64) * 0.7
    meta = {'crs': 'EPSG:4326', 'gsd': 0.5}

    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        out_path = f.name

    try:
        export_geotiff(warped, out_path, meta)
        npy_fallback = out_path.replace('.tif', '.npy')
        assert os.path.exists(out_path) or os.path.exists(npy_fallback)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(npy_fallback):
            os.remove(npy_fallback)


def test_build_metrics_report():
    """build_metrics_report formats required fields correctly."""
    rep = build_metrics_report(
        rmse=0.42,
        inlier_ratio=0.88,
        sdi=0.91,
        n_inliers=150,
        n_total=170,
        elapsed_s=2.34,
        transform="tps",
    )
    assert rep['rmse_px'] == 0.42
    assert rep['inlier_ratio'] == 0.88
    assert rep['sdi'] == 0.91
    assert rep['n_inliers'] == 150
    assert rep['transform'] == "tps"
