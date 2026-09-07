"""
core/anms_spatial_filter.py
===========================
Adaptive Non-Maximal Suppression (ANMS) & Spatial Distribution Filtering.

Ensures candidate feature correspondences are uniformly distributed across the entire
lunar surface viewport rather than clustering solely on high-contrast crater rims.

Implements:
  - Suppression radius computation:
      r_i = min_j ||x_i - x_j||_2  subject to  f(x_i) < c_robust * f(x_j)
  - Top-k spatial selection
  - Quad-tree spatial grid bucketing fallback

References:
  Brown, M. et al. (2005). Multi-Image Matching using Multi-Scale Oriented Patches.
  Bailer, C. et al. (2018). Efficient Adaptive Non-Maximal Suppression Algorithms for Homogeneous Keypoint Distribution.
"""

import logging
from typing import Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def compute_suppression_radius(
    points: np.ndarray,
    strengths: np.ndarray,
    c_robust: float = 0.9,
) -> np.ndarray:
    """
    Compute suppression radius for each feature point:
      r_i = min_{j: f(x_j) * c_robust > f(x_i)} ||x_i - x_j||_2

    Points with no significantly stronger neighbors receive r = infinity.

    Parameters
    ----------
    points    : (N, 2) float array of (x, y) coordinates
    strengths : (N,) float array of feature corner/confidence responses
    c_robust  : robustness factor (default 0.9)

    Returns
    -------
    radii : (N,) float64 array of suppression radii
    """
    N = len(points)
    if N == 0:
        return np.array([], dtype=np.float64)
    if N == 1:
        return np.array([np.inf], dtype=np.float64)

    pts = points.astype(np.float64)
    strs = strengths.astype(np.float64)
    radii = np.full(N, np.inf, dtype=np.float64)

    # Moderate N: pairwise vectorized distance
    if N <= 3000:
        diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 2)
        dist_sq = np.sum(diff**2, axis=-1)       # (N, N)

        # Condition: f(x_j) * c_robust > f(x_i)
        is_stronger = (strs[None, :] * c_robust) > strs[:, None]  # (N, N)

        # For self and non-stronger points, set distance to infinity
        dist_sq[~is_stronger] = np.inf
        np.fill_diagonal(dist_sq, np.inf)

        min_sq = np.min(dist_sq, axis=1)
        radii = np.sqrt(min_sq)
        return radii

    # Large N: sort descending by strength and use cKDTree / incremental search
    sort_indices = np.argsort(-strs)
    sorted_pts = pts[sort_indices]
    sorted_strs = strs[sort_indices]

    # The absolute strongest point has infinite radius
    radii[sort_indices[0]] = np.inf

    # For subsequent points, search among stronger points
    for idx in range(1, N):
        # Candidates are points with f(x_j) * c_robust > f(x_i)
        stronger_mask = (sorted_strs[:idx] * c_robust) > sorted_strs[idx]
        if not np.any(stronger_mask):
            radii[sort_indices[idx]] = np.inf
            continue

        cand_pts = sorted_pts[:idx][stronger_mask]
        cur_pt = sorted_pts[idx]
        dists = np.linalg.norm(cand_pts - cur_pt, axis=1)
        radii[sort_indices[idx]] = np.min(dists)

    return radii


def anms_select(
    points_or_matches: np.ndarray,
    strengths: Optional[np.ndarray] = None,
    k: int = 2000,
    c_robust: float = 0.9,
    **kwargs,
) -> np.ndarray:
    """
    Select top-k spatially distributed points or matches using ANMS.

    Supports two signatures:
      1) anms_select(matches, k=2000) -> filtered_matches (N, 5)  [Orchestrator format]
      2) anms_select(points, strengths, k=2000) -> selected_points (K, 2) [Standalone format]

    Parameters
    ----------
    points_or_matches : (N, 5) match array or (N, 2) point coordinates
    strengths         : (N,) strength array (if points_or_matches is (N, 2))
    k                 : target number of features to retain
    c_robust          : robustness factor

    Returns
    -------
    filtered : (K, 5) matches or (K, 2) points
    """
    arr = np.asarray(points_or_matches, dtype=np.float64)
    N = len(arr)

    if N == 0:
        return arr

    if N <= k:
        return arr.copy()

    # Format 1: Orchestrator match array (N, 5) [x1, y1, x2, y2, confidence]
    if arr.ndim == 2 and arr.shape[1] == 5 and strengths is None:
        pts = arr[:, :2]
        strs = arr[:, 4]
        radii = compute_suppression_radius(pts, strs, c_robust=c_robust)
        sort_order = np.argsort(-radii)
        top_k_indices = sort_order[:k]
        return arr[top_k_indices]

    # Format 2: Standalone points (N, 2) + strengths (N,)
    pts = arr
    if strengths is None:
        strs = np.ones(N, dtype=np.float64)
    else:
        strs = np.asarray(strengths, dtype=np.float64)

    radii = compute_suppression_radius(pts, strs, c_robust=c_robust)
    sort_order = np.argsort(-radii)
    top_k_indices = sort_order[:k]
    return pts[top_k_indices]


def quadtree_bucket_fallback(
    points: np.ndarray,
    grid: int = 8,
    max_per_cell: int = 15,
) -> np.ndarray:
    """
    Spatial bucketing fallback: subdivides the viewport into grid x grid cells
    and enforces maximum quota per cell to prevent feature clustering.

    Parameters
    ----------
    points       : (N, 2) array of coordinates
    grid         : grid divisions along each axis (e.g. 8x8 = 64 buckets)
    max_per_cell : maximum points allowed per grid cell

    Returns
    -------
    selected_points : filtered subset of points
    """
    if len(points) == 0:
        return points

    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()

    dx = max(max_x - min_x, 1e-5) / grid
    dy = max(max_y - min_y, 1e-5) / grid

    cell_counts = {}
    selected_indices = []

    for idx, (x, y) in enumerate(points):
        cx = min(int((x - min_x) / dx), grid - 1)
        cy = min(int((y - min_y) / dy), grid - 1)
        cell_id = (cx, cy)

        cnt = cell_counts.get(cell_id, 0)
        if cnt < max_per_cell:
            cell_counts[cell_id] = cnt + 1
            selected_indices.append(idx)

    return points[selected_indices]
