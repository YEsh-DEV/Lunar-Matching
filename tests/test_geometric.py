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
