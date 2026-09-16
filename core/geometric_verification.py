"""
core/geometric_verification.py
===============================
Robust Geometric Verification Module.

Implements:
  - MAGSAC++ outlier filtering via OpenCV's cv2.USAC_MAGSAC
  - Thin Plate Spline (TPS) non-rigid transform via scipy RBFInterpolator
  - Homography residual analysis to decide if TPS is warranted

References:
  Barath, D. et al. (2020). MAGSAC++: A Fast, Reliable and Accurate Robust Estimator.
  Bookstein, F. (1989). Principal Warps: Thin-Plate Splines.
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Thin Plate Spline wrapper (interface contract: .apply(points) method)
# ---------------------------------------------------------------------------

class ThinPlateSplineTransform:
    """
    Thin Plate Spline non-rigid transform.

    TPS: f(x,y) = a1 + ax*x + ay*y + sum_i wi * U(||(x,y)-(xi,yi)||)
         U(r) = r^2 * ln(r)   (thin_plate_spline kernel in SciPy)

    Fits separate RBF interpolators for x and y output channels.
    Exposes .apply(points) -> points interface required by the contract.
    """

    def __init__(self, rbf_x, rbf_y):
        self._rbf_x = rbf_x
        self._rbf_y = rbf_y

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the TPS warp to a set of source points.

        Parameters
        ----------
        points : (N, 2) float array of (x, y) coordinates in image A

        Returns
        -------
        warped : (N, 2) float array of corresponding (x, y) in image B
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points must be (N,2) array; got shape {points.shape}")

        warped_x = self._rbf_x(points)
        warped_y = self._rbf_y(points)
        return np.column_stack([warped_x, warped_y])


# ---------------------------------------------------------------------------
# 1. MAGSAC++ Filter
# ---------------------------------------------------------------------------

_MIN_MATCH_POINTS = 8   # minimum for reliable homography estimation

def compute_condition_number(matrix: Optional[np.ndarray]) -> float:
    """
    Compute condition number kappa = sigma_max / sigma_min via SVD.
    For affine / similarity transforms (2x3 or 3x3 with [0, 0, 1] bottom row),
    evaluates the condition number of the linear 2x2 transformation block.
    For projective homographies, evaluates the full 3x3 matrix.
    Returns float('inf') for singular or None matrices.
    """
    if matrix is None:
        return float("inf")
    try:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.shape == (2, 3):
            s = np.linalg.svd(mat[:, :2], compute_uv=False)
        elif mat.shape == (3, 3) and abs(mat[2, 0]) < 1e-7 and abs(mat[2, 1]) < 1e-7 and abs(mat[2, 2] - 1.0) < 1e-3:
            s = np.linalg.svd(mat[:2, :2], compute_uv=False)
        else:
            s = np.linalg.svd(mat, compute_uv=False)

        sigma_max = float(s[0])
        sigma_min = float(s[-1])
        if sigma_min <= 1e-15:
            return 1e15
        return float(sigma_max / sigma_min)
    except Exception:
        return float("inf")


def refit_with_condition_fallback(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    H: np.ndarray,
    kappa_threshold: float = 1e4,
) -> Tuple[np.ndarray, str, float]:
    """
    Check condition number kappa = sigma_max / sigma_min via SVD.
    If kappa > 1e4:
      1. Refit as affine (cv2.estimateAffine2D or cv2.estimateAffinePartial2D).
      2. If still degenerate (kappa > 1e4), refit as similarity (4-DOF via cv2.estimateAffinePartial2D).
    Returns (transform_3x3, transform_type, condition_number).
    """
    kappa = compute_condition_number(H)
    if kappa <= kappa_threshold:
        return H, "homography", kappa

    logger.warning(
        f"Homography condition number kappa={kappa:.2e} exceeds threshold {kappa_threshold:.0e}. "
        "Triggering model fallback hierarchy..."
    )

    if not _HAS_CV2 or len(pts_a) < 3:
        return H, "homography", kappa

    src = pts_a.reshape(-1, 1, 2).astype(np.float64)
    dst = pts_b.reshape(-1, 1, 2).astype(np.float64)

    # Step 1: Attempt affine (6-DOF)
    M_aff = None
    kappa_aff = float("inf")
    try:
        M_aff, _ = cv2.estimateAffine2D(src, dst)
        if M_aff is not None:
            H_aff = np.eye(3, dtype=np.float64)
            H_aff[:2, :] = M_aff
            kappa_aff = compute_condition_number(H_aff)
            if kappa_aff <= kappa_threshold:
                logger.info(f"Fallback to affine successful: kappa={kappa_aff:.2e}")
                return H_aff, "affine", kappa_aff
    except Exception as e_aff:
        logger.debug(f"estimateAffine2D failed: {e_aff}")

    # Step 2: Still degenerate or failed: refit as similarity (4-DOF) via cv2.estimateAffinePartial2D
    logger.warning(
        f"Affine refit still degenerate (kappa={kappa_aff:.2e} > {kappa_threshold:.0e}). "
        "Falling back to similarity (4-DOF) via cv2.estimateAffinePartial2D..."
    )
    try:
        M_sim, _ = cv2.estimateAffinePartial2D(src, dst)
        if M_sim is not None:
            H_sim = np.eye(3, dtype=np.float64)
            H_sim[:2, :] = M_sim
            kappa_sim = compute_condition_number(H_sim)
            logger.info(f"Fallback to similarity successful: kappa={kappa_sim:.2e}")
            return H_sim, "similarity", kappa_sim
    except Exception as e_sim:
        logger.warning(f"estimateAffinePartial2D failed: {e_sim}")

    # If all refits fail, return original H
    return H, "homography", kappa


def magsac_filter(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    model: str = 'homography',
    confidence: float = 0.999,
    max_iterations: int = 10000,
    reprojection_threshold: float = 3.0,
    return_details: bool = False,
) -> tuple:
    """
    Robust geometric verification using MAGSAC++.

    Parameters
    ----------
    pts_a : (N, 2) float array of keypoint coords in image A
    pts_b : (N, 2) float array of keypoint coords in image B
    model : 'homography' (default; cv2.USAC_MAGSAC)
    confidence : RANSAC confidence parameter
    max_iterations : maximum RANSAC/MAGSAC iterations
    reprojection_threshold : fallback threshold (MAGSAC++ marginalizes this)
    return_details : if True, returns (inlier_mask, H, transform_type, condition_number)

    Returns
    -------
    inlier_mask : (N,) bool array -- True = geometric inlier
    H_matrix    : (3, 3) float64 transform matrix, or None if estimation fails
    """
    N = len(pts_a)

    # --- Degenerate case guards ---
    if N < _MIN_MATCH_POINTS:
        logger.warning(
            f"MAGSAC: only {N} points supplied; minimum is {_MIN_MATCH_POINTS}. "
            "Returning all-False inlier mask."
        )
        if return_details:
            return np.zeros(N, dtype=bool), None, "none", float("inf")
        return np.zeros(N, dtype=bool), None

    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for MAGSAC++ geometric verification.")

    # Reshape for OpenCV: (N, 1, 2)
    src = pts_a.reshape(-1, 1, 2).astype(np.float64)
    dst = pts_b.reshape(-1, 1, 2).astype(np.float64)

    try:
        if model == 'homography':
            H, mask = cv2.findHomography(
                src, dst,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=reprojection_threshold,
                confidence=confidence,
                maxIters=max_iterations,
            )
        else:
            raise ValueError(f"Unsupported model type: {model}")

    except cv2.error as e:
        logger.error(f"MAGSAC++ failed with OpenCV error: {e}")
        if return_details:
            return np.zeros(N, dtype=bool), None, "none", float("inf")
        return np.zeros(N, dtype=bool), None

    # Handle failure modes
    if H is None or mask is None:
        logger.warning(
            "MAGSAC++ returned None (all-outlier failure or degenerate config). "
            "Returning all-False inlier mask."
        )
        if return_details:
            return np.zeros(N, dtype=bool), None, "none", float("inf")
        return np.zeros(N, dtype=bool), None

    inlier_mask = mask.ravel().astype(bool)

    # Condition number check via SVD and fallback hierarchy
    kappa = compute_condition_number(H)
    transform_type = "homography"
    condition_number = kappa

    if kappa > 1e4:
        inliers_a = pts_a[inlier_mask]
        inliers_b = pts_b[inlier_mask]
        if len(inliers_a) >= 3:
            H, transform_type, condition_number = refit_with_condition_fallback(
                inliers_a, inliers_b, H, kappa_threshold=1e4
            )

    n_inliers = inlier_mask.sum()
    logger.info(
        f"MAGSAC++: {N} candidates → {n_inliers} inliers "
        f"({100.0 * n_inliers / N:.1f}% inlier ratio) | type={transform_type} kappa={condition_number:.2e}"
    )

    if return_details:
        return inlier_mask, H.astype(np.float64), transform_type, condition_number
    return inlier_mask, H.astype(np.float64)


# ---------------------------------------------------------------------------
# 2. Thin Plate Spline Fitting
# ---------------------------------------------------------------------------

def fit_thin_plate_spline(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    smoothing: float = 0.0,
) -> object:
    """
    Fit a Thin Plate Spline transform from src_pts to dst_pts.

    TPS kernel: U(r) = r^2 * ln(r) (scipy 'thin_plate_spline').
    Fits two independent RBFInterpolators: one for x-output, one for y-output.

    Parameters
    ----------
    src_pts : (N, 2) float -- source pixel coordinates (x, y) in image A
    dst_pts : (N, 2) float -- destination pixel coordinates (x, y) in image B
    smoothing : regularization strength (0.0 = exact interpolation)

    Returns
    -------
    ThinPlateSplineTransform object with .apply(points) method
    """
    if len(src_pts) < 4:
        raise ValueError(
            f"TPS requires at least 4 control points; got {len(src_pts)}"
        )

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy is required for Thin Plate Spline (TPS) fitting.")

    src = src_pts.astype(np.float64)
    dst = dst_pts.astype(np.float64)

    # Separate RBF for x and y channels
    rbf_x = RBFInterpolator(
        src, dst[:, 0],
        kernel='thin_plate_spline',
        smoothing=smoothing,
    )
    rbf_y = RBFInterpolator(
        src, dst[:, 1],
        kernel='thin_plate_spline',
        smoothing=smoothing,
    )

    logger.info(
        f"TPS fitted on {len(src_pts)} control points "
        f"(smoothing={smoothing})."
    )

    return ThinPlateSplineTransform(rbf_x, rbf_y)


# ---------------------------------------------------------------------------
# 3. Relief Significance Check
# ---------------------------------------------------------------------------

def check_relief_significance(
    inlier_pts: np.ndarray,
    homography_residuals: np.ndarray,
    threshold_px: float = 1.5,
    fraction_threshold: float = 0.3,
) -> bool:
    """
    Decide if non-rigid TPS warp is warranted over a flat homography.

    Rationale: If crater walls, central peaks, or rille edges introduce
    significant parallax displacement, the homography reprojection residuals
    will be systematically large at those affected points. TPS can absorb
    this non-rigid displacement; flat homography cannot.

    Criterion: if >fraction_threshold of inliers have residual > threshold_px,
    TPS is warranted.

    Parameters
    ----------
    inlier_pts            : (N, 2) inlier point coordinates (unused here, for future spatial analysis)
    homography_residuals  : (N,) per-point reprojection residuals in pixels
    threshold_px          : residual magnitude considered "significant"
    fraction_threshold    : fraction of points exceeding threshold to trigger TPS

    Returns
    -------
    bool : True if TPS non-rigid warp is recommended
    """
    if len(homography_residuals) == 0:
        return False

    residuals = np.abs(homography_residuals)
    fraction_exceeding = np.mean(residuals > threshold_px)

    tps_warranted = fraction_exceeding > fraction_threshold

    logger.info(
        f"Relief check: {fraction_exceeding:.1%} of inliers exceed {threshold_px}px "
        f"residual → TPS {'recommended' if tps_warranted else 'not needed'}."
    )

    return tps_warranted


# ---------------------------------------------------------------------------
# Convenience: compute homography reprojection residuals
# ---------------------------------------------------------------------------

def compute_homography_residuals(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Compute per-point reprojection errors for a homography H.

    error_i = ||H * pts_a[i] - pts_b[i]||_2

    Parameters
    ----------
    pts_a : (N, 2) -- source points
    pts_b : (N, 2) -- destination points
    H     : (3, 3) homography matrix

    Returns
    -------
    residuals : (N,) float -- Euclidean reprojection error in pixels
    """
    N = len(pts_a)
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([pts_a.astype(np.float64), ones])   # (N, 3)
    proj  = (H @ pts_h.T).T                                # (N, 3)

    # De-homogenize
    w = proj[:, 2:3] + 1e-10
    proj_eucl = proj[:, :2] / w                             # (N, 2)

    residuals = np.linalg.norm(proj_eucl - pts_b, axis=1)
    return residuals
