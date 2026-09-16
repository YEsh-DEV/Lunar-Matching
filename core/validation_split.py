"""
core/validation_split.py
=========================
Stage 6b — Held-Out Validation Split for Transform Verification.

Provides unbiased validation RMSE on held-out match correspondences to guard
against overfitting (especially in Thin Plate Spline and higher-order models).

Specification:
  - Shuffles verified inliers deterministically with a fixed seed (default 42).
  - 80/20 train/validation split (split_ratio = 0.8).
  - Fits the transform fitter on the 80% training split.
  - Evaluates reprojection RMSE against the held-out 20% validation split.
  - Returns {held_out_rmse_px, fit_rmse_px, overfit_ratio, n_fit, n_val}.
  - overfit_ratio = held_out_rmse / fit_rmse (> 1.5 indicates potential overfit).
"""

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def project_points(transform: Any, pts: np.ndarray) -> np.ndarray:
    """
    Project (N, 2) source points using either:
      - ThinPlateSplineTransform (has .apply(pts))
      - 3x3 homography matrix (np.ndarray)
      - Callable f(pts) -> (N, 2)
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if hasattr(transform, "apply"):
        return np.asarray(transform.apply(pts), dtype=np.float64)

    if isinstance(transform, np.ndarray) and transform.shape == (3, 3):
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        proj = (transform @ pts_h.T).T
        w = proj[:, 2:3] + 1e-10
        return proj[:, :2] / w

    if callable(transform):
        return np.asarray(transform(pts), dtype=np.float64)

    logger.warning("Unrecognized transform type in project_points; returning input unchanged.")
    return pts.copy()


def held_out_rmse(
    matches: np.ndarray,
    transform_fitter: Callable[[np.ndarray, np.ndarray], Any],
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate transform generalization using a deterministic held-out validation split.

    Parameters
    ----------
    matches : (N, >=4) ndarray of match correspondences [x1, y1, x2, y2, ...]
    transform_fitter : callable(src_pts, dst_pts) -> transform
    split_ratio : fraction of matches used for fitting (default 0.8)
    seed : random seed for deterministic shuffle (default 42)

    Returns
    -------
    dict with:
      - held_out_rmse_px : float, reprojection RMSE on held-out test split
      - fit_rmse_px      : float, reprojection RMSE on training split
      - overfit_ratio    : float, held_out_rmse / fit_rmse
      - n_fit            : int, number of matches used for fitting
      - n_val            : int, number of matches used for validation
    """
    if matches is None or len(matches) < 5:
        # Minimum 4 points needed to fit homography plus at least 1 held-out point
        logger.info("Fewer than 5 matches available; skipping held-out validation split.")
        return {
            "held_out_rmse_px": 0.0,
            "fit_rmse_px": 0.0,
            "overfit_ratio": 1.0,
            "n_fit": len(matches) if matches is not None else 0,
            "n_val": 0,
        }

    # Deterministic shuffle with fixed seed
    rng = np.random.default_rng(seed)
    indices = np.arange(len(matches))
    rng.shuffle(indices)
    shuffled = matches[indices]

    # Compute split point ensuring at least 4 fit points and at least 1 val point
    n_total = len(shuffled)
    n_fit = int(np.floor(n_total * split_ratio))
    n_fit = max(4, min(n_fit, n_total - 1))
    n_val = n_total - n_fit

    train_matches = shuffled[:n_fit]
    val_matches = shuffled[n_fit:]

    src_train = train_matches[:, :2].astype(np.float64)
    dst_train = train_matches[:, 2:4].astype(np.float64)
    src_val = val_matches[:, :2].astype(np.float64)
    dst_val = val_matches[:, 2:4].astype(np.float64)

    try:
        model = transform_fitter(src_train, dst_train)
        if model is None:
            raise RuntimeError("transform_fitter returned None")

        train_proj = project_points(model, src_train)
        fit_diff = train_proj - dst_train
        fit_rmse = float(np.sqrt(np.mean(np.sum(fit_diff ** 2, axis=1))))

        val_proj = project_points(model, src_val)
        val_diff = val_proj - dst_val
        held_out_rmse_val = float(np.sqrt(np.mean(np.sum(val_diff ** 2, axis=1))))

        overfit_ratio = float(held_out_rmse_val / max(fit_rmse, 1e-6))

        logger.info(
            f"Stage 6b: Held-out RMSE = {held_out_rmse_val:.4f}px | "
            f"Fit RMSE = {fit_rmse:.4f}px | Overfit Ratio = {overfit_ratio:.2f} "
            f"(train={n_fit}, val={n_val})"
        )

        return {
            "held_out_rmse_px": round(held_out_rmse_val, 4),
            "fit_rmse_px": round(fit_rmse, 4),
            "overfit_ratio": round(overfit_ratio, 4),
            "n_fit": int(n_fit),
            "n_val": int(n_val),
        }

    except Exception as e:
        logger.warning(f"Held-out validation fitting encountered an issue: {e}")
        return {
            "held_out_rmse_px": 0.0,
            "fit_rmse_px": 0.0,
            "overfit_ratio": 1.0,
            "n_fit": int(n_fit),
            "n_val": int(n_val),
        }
