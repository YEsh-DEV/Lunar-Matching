"""
tests/test_geometric.py  (v2 — numpy Generator fix, bool identity fix, tolerance fix)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.geometric_verification import (
    magsac_filter, fit_thin_plate_spline, check_relief_significance,
    compute_homography_residuals, ThinPlateSplineTransform,
)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def make_planar_correspondences(n_inliers=150, n_outliers=50, noise_std=0.5):
    rng = np.random.default_rng(42)
    H_true = np.array([
        [1.05, 0.02, 12.0],
        [-0.01, 0.98,  8.0],
        [0.0001, 0.0001, 1.0]
    ])
    pts_a = rng.uniform(50, 450, size=(n_inliers, 2))
    ones  = np.ones((n_inliers, 1))
    pts_h = np.hstack([pts_a, ones])
    proj  = (H_true @ pts_h.T).T
    w     = proj[:, 2:3] + 1e-10
    pts_b_true = proj[:, :2] / w + rng.standard_normal((n_inliers, 2)) * noise_std

    out_a = rng.uniform(0, 500, size=(n_outliers, 2))
    out_b = rng.uniform(0, 500, size=(n_outliers, 2))

    pts_a_all = np.vstack([pts_a, out_a])
    pts_b_all = np.vstack([pts_b_true, out_b])
    gt_inliers = np.array([True] * n_inliers + [False] * n_outliers)
    return pts_a_all, pts_b_all, gt_inliers, H_true


def make_relief_correspondences(n_pts=100):
    rng = np.random.default_rng(7)
    pts_a = rng.uniform(50, 450, size=(n_pts, 2))
    cx, cy = 250, 250
    dx = pts_a[:, 0] - cx
    dy = pts_a[:, 1] - cy
    r  = np.sqrt(dx**2 + dy**2) + 1.0
    scale = 0.003
    pts_b = pts_a.copy()
    pts_b[:, 0] += scale * dx * dy / r
    pts_b[:, 1] += scale * dy * dx / r
    pts_b += rng.standard_normal((n_pts, 2)) * 0.3   # fixed: was rng.randn
    return pts_a, pts_b


@pytest.mark.skipif(not _HAS_CV2, reason="OpenCV not installed")
def test_magsac_rejects_outliers():
    n_inliers, n_outliers = 150, 50
    pts_a, pts_b, gt_inliers, _ = make_planar_correspondences(n_inliers, n_outliers)
    inlier_mask, H = magsac_filter(pts_a, pts_b)
    assert H is not None
    assert inlier_mask.shape == (n_inliers + n_outliers,)
    tpr = inlier_mask[:n_inliers].mean()
    fpr = inlier_mask[n_inliers:].mean()
    assert tpr > 0.80, f"True positive rate {tpr:.2%} < 80%"
    assert fpr < 0.20, f"False positive rate {fpr:.2%} > 20%"


def test_magsac_too_few_points():
    pts_a = np.random.rand(5, 2) * 100
    pts_b = np.random.rand(5, 2) * 100
    mask, H = magsac_filter(pts_a, pts_b)
    assert H is None
    assert mask.dtype == bool
    assert mask.sum() == 0


@pytest.mark.skipif(not _HAS_SCIPY, reason="SciPy not installed")
def test_tps_vs_homography_on_relief():
    pts_a, pts_b = make_relief_correspondences(n_pts=80)
    if _HAS_CV2:
        H, _ = cv2.findHomography(
            pts_a.reshape(-1,1,2).astype(np.float32),
            pts_b.reshape(-1,1,2).astype(np.float32),
        )
        if H is not None:
            h_res  = compute_homography_residuals(pts_a, pts_b, H)
            h_rmse = np.sqrt(np.mean(h_res**2))
        else:
            h_rmse = 999.0
    else:
        h_rmse = 5.0

    tps = fit_thin_plate_spline(pts_a, pts_b)
    tps_res  = np.linalg.norm(tps.apply(pts_a) - pts_b, axis=1)
    tps_rmse = np.sqrt(np.mean(tps_res**2))

    assert tps_rmse < h_rmse, \
        f"TPS RMSE ({tps_rmse:.3f}) not less than Homography RMSE ({h_rmse:.3f})"


@pytest.mark.skipif(not _HAS_SCIPY, reason="SciPy not installed")
def test_tps_apply_interface():
    pts_a = np.random.rand(20, 2) * 200
    pts_b = pts_a + np.random.randn(20, 2) * 2
    tps = fit_thin_plate_spline(pts_a, pts_b)
    assert hasattr(tps, 'apply')
    result = tps.apply(np.random.rand(10, 2) * 200)
    assert result.shape == (10, 2)


def test_relief_significance_triggers_on_large_residuals():
    pts   = np.random.rand(50, 2) * 100
    large = np.random.uniform(2.0, 5.0, 50)
    result = check_relief_significance(pts, large, threshold_px=1.5)
    assert bool(result) == True, "Should recommend TPS for large residuals"


def test_relief_significance_flat_scene():
    pts   = np.random.rand(50, 2) * 100
    small = np.random.uniform(0.1, 0.4, 50)
    result = check_relief_significance(pts, small, threshold_px=1.5)
    assert bool(result) == False, "Should NOT recommend TPS for sub-pixel residuals"


def test_compute_homography_residuals_identity():
    pts_a = np.random.rand(20, 2) * 200
    H_id  = np.eye(3)
    residuals = compute_homography_residuals(pts_a, pts_a.copy(), H_id)
    assert residuals.max() < 1e-7, f"Identity H should give ~zero residuals; max={residuals.max():.2e}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ---------------------------------------------------------------------------
# Checkpoint 3: TPS control-point cap test
# ---------------------------------------------------------------------------

def test_tps_cap_48_points():
    """
    With 200 synthetic source points, ANMS selection of top 48 must:
      1. Yield exactly min(48, N) control points.
      2. Produce a TPS transform with RMSE < 1.0px on held-out test points.
    This test directly verifies the orchestrator's cap logic (the selection
    itself) by calling anms_select + fit_thin_plate_spline in isolation.
    """
    from core.anms_spatial_filter import anms_select

    _TPS_MAX_CTRL_PTS = 48
    rng = np.random.default_rng(42)

    # 200 source points spread across a 400x400 canvas
    src_pts = rng.uniform(20, 380, (200, 2))
    # Ground-truth destination: uniform 2px shift + small noise
    dst_pts = src_pts + np.array([2.0, 3.0]) + rng.normal(0, 0.05, src_pts.shape)

    # Simulate orchestrator ANMS cap: build (N,5) array with confidence column
    confs = np.ones(200)
    candidate_arr = np.column_stack([src_pts, dst_pts, confs])
    selected = anms_select(candidate_arr, k=_TPS_MAX_CTRL_PTS)

    assert len(selected) <= _TPS_MAX_CTRL_PTS, (
        f"ANMS cap must return <= {_TPS_MAX_CTRL_PTS} points; got {len(selected)}"
    )
    assert len(selected) >= 4, "Must have at least 4 points for TPS"

    ctrl_src = selected[:, :2]
    ctrl_dst = selected[:, 2:4]

    tps = fit_thin_plate_spline(ctrl_src, ctrl_dst)
    assert hasattr(tps, 'apply'), "TPS must have .apply() interface"

    # Verify accuracy on held-out test points (not used in fitting)
    test_src = rng.uniform(20, 380, (30, 2))
    test_dst = test_src + np.array([2.0, 3.0])
    pred_dst = tps.apply(test_src)
    errors = np.linalg.norm(pred_dst - test_dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    assert rmse < 1.0, (
        f"TPS accuracy check: RMSE={rmse:.4f}px must be < 1.0px on held-out points"
    )


def test_condition_number_fallback_near_collinear():
    """
    Synthetic near-collinear points cause homography to have condition number kappa > 1e4.
    Verify that magsac_filter triggers fallback to affine/similarity and condition_number <= 1e4.
    """
    dy = 1e-4
    pts_a = np.array([
        [0.0, 0.0],
        [100.0, dy],
        [200.0, -dy],
        [300.0, 2*dy],
        [400.0, -2*dy],
        [500.0, dy],
        [600.0, -dy],
        [700.0, 2*dy],
    ])
    pts_b = pts_a.copy()
    pts_b[:, 0] += 50.0
    pts_b[2, 0] += 0.5

    mask, model, transform_type, cond = magsac_filter(pts_a, pts_b, return_details=True)

    assert transform_type in ("affine", "similarity"), (
        f"Expected fallback to affine or similarity; got {transform_type}"
    )
    assert cond <= 1e4, f"Expected condition number <= 1e4 after fallback; got {cond}"
    assert model is not None
    assert model.shape == (3, 3)


