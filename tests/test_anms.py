"""
tests/test_anms.py
==================
Unit and integration tests for core/anms_spatial_filter.py.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.anms_spatial_filter import (
    compute_suppression_radius,
    anms_select,
    quadtree_bucket_fallback,
)


def test_suppression_radius_properties():
    """Points with greatest strength have infinite radius; nearby weaker points have small radii."""
    # Point 0: strong at (10, 10), Point 1: weak at (11, 10), Point 2: medium at (100, 100)
    pts = np.array([[10.0, 10.0], [11.0, 10.0], [100.0, 100.0]])
    strengths = np.array([1.0, 0.2, 0.8])

    radii = compute_suppression_radius(pts, strengths, c_robust=0.9)

    # Point 0 is absolute maximum -> inf
    assert np.isinf(radii[0])
    # Point 1 is suppressed by Point 0 at distance 1.0
    assert abs(radii[1] - 1.0) < 1e-4
    # Point 2 is suppressed by Point 0 at Euclidean distance ~127.28
    assert radii[2] > 50.0


def test_anms_select_orchestrator_contract():
    """anms_select must accept (N, 5) match array and return (K, 5) float64."""
    rng = np.random.default_rng(42)
    N = 500
    x1 = rng.uniform(0, 500, N)
    y1 = rng.uniform(0, 500, N)
    x2 = x1 + rng.normal(0, 1.0, N)
    y2 = y1 + rng.normal(0, 1.0, N)
    conf = rng.uniform(0.1, 1.0, N)

    matches_raw = np.column_stack([x1, y1, x2, y2, conf]).astype(np.float64)

    filtered = anms_select(matches_raw, k=100)

    assert isinstance(filtered, np.ndarray)
    assert filtered.dtype == np.float64
    assert filtered.shape == (100, 5)


def test_anms_improves_spatial_dispersion():
    """ANMS must produce points that are farther apart than naive top-k selection."""
    rng = np.random.default_rng(123)
    # Cluster 80 points tightly at (50, 50); peak is at index 0 with strength 1.0, others 0.2-0.7
    cluster_pts = rng.normal(50, 2.0, (80, 2))
    cluster_strs = rng.uniform(0.2, 0.7, 80)
    cluster_strs[0] = 1.0  # Cluster peak

    # Spread 80 points uniformly across [100, 500] with medium-high strength (0.8)
    spread_pts = rng.uniform(100, 500, (80, 2))
    spread_strs = rng.uniform(0.75, 0.85, 80)

    all_pts = np.vstack([cluster_pts, spread_pts])
    all_strs = np.concatenate([cluster_strs, spread_strs])

    selected = anms_select(all_pts, all_strs, k=40)

    # ANMS should suppress the cluster (only the peak survives) and pick distributed points
    in_cluster = (selected[:, 0] < 80) & (selected[:, 1] < 80)
    assert in_cluster.sum() <= 2, "ANMS must suppress dense clusters in favor of spatial coverage."


def test_quadtree_bucket_fallback():
    """quadtree_bucket_fallback enforces capacity limit per grid cell."""
    # 50 points in the exact same cell
    pts = np.full((50, 2), 25.0)
    filtered = quadtree_bucket_fallback(pts, grid=8, max_per_cell=5)

    assert len(filtered) == 5
