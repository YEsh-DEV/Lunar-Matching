"""
tests/test_structural_matching.py
==================================
Unit and Integration Tests for RIFT-Style Phase Congruency / MIM Structural Matching.

Covers:
  - (N, 5) interface contract compliance for run_structural_matching
  - Handling of uniform/degenerate inputs without exception
  - Execution and correspondence verification on linear verified test pair
  - Empirical evaluation on harsh illumination stress pair
  - Dispatch mechanism in run_dense_matching(method='classical'|'structural')
  - MIM descriptor normalization and dimensional properties
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

# Ensure repository root is on path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from core.dense_matcher import run_structural_matching, run_dense_matching
from core.phase_congruency_mim import mim_descriptor, compute_mim


def test_structural_matching_contract_shape():
    """Verify that run_structural_matching strictly returns an (N, 5) float64 array."""
    rng = np.random.default_rng(42)
    img_a = rng.uniform(0, 255, (128, 128)).astype(np.float64)
    img_b = rng.uniform(0, 255, (128, 128)).astype(np.float64)

    matches = run_structural_matching(img_a, img_b)

    assert isinstance(matches, np.ndarray), "Output must be a numpy ndarray"
    assert matches.ndim == 2, f"Expected 2D array, got ndim={matches.ndim}"
    assert matches.shape[1] == 5, f"Expected 5 columns [x1, y1, x2, y2, conf], got {matches.shape[1]}"
    assert matches.dtype == np.float64, f"Expected float64 dtype, got {matches.dtype}"


def test_structural_matching_empty_on_flat_image():
    """Flat uniform images should gracefully return an empty (0, 5) array."""
    img_a = np.zeros((100, 100), dtype=np.float64)
    img_b = np.zeros((100, 100), dtype=np.float64)

    matches = run_structural_matching(img_a, img_b)
    assert matches.shape == (0, 5)


def test_mim_descriptor_properties():
    """Verify MIM descriptor dimension, normalization, and bounds."""
    mim = np.random.randint(0, 6, (120, 120), dtype=np.uint8)
    keypoints = np.array([[30.0, 30.0], [50.0, 60.0], [80.0, 80.0]], dtype=np.float32)

    desc = mim_descriptor(mim, keypoints, grid_size=6, num_orientation_bins=6, patch_radius=20)

    # 6 * 6 * 6 = 216 dimensions
    assert desc.shape == (3, 216)
    assert desc.dtype == np.float32

    # Verify L2 normalization
    norms = np.linalg.norm(desc, axis=1)
    for norm in norms:
        assert abs(norm - 1.0) < 1e-4, f"Descriptor should be L2-normalised to 1.0; got {norm}"

    # Verify non-negativity
    assert np.all(desc >= 0.0), "Histogram bin values must be non-negative"


def test_dispatch_mechanism():
    """Verify run_dense_matching dispatches correctly based on method argument."""
    rng = np.random.default_rng(123)
    img_a = rng.uniform(0, 255, (64, 64)).astype(np.float64)
    img_b = rng.uniform(0, 255, (64, 64)).astype(np.float64)

    res_struct = run_dense_matching(img_a, img_b, method="structural")
    assert res_struct.ndim == 2 and res_struct.shape[1] == 5

    res_class = run_dense_matching(img_a, img_b, method="classical")
    assert res_class.ndim == 2 and res_class.shape[1] == 5


def test_structural_matching_verified_pair():
    """Verify structural matching runs on real verified test pair without regression."""
    p_a = root_dir / "data" / "samples" / "verified_a.tif"
    p_b = root_dir / "data" / "samples" / "verified_b.tif"

    if not (p_a.exists() and p_b.exists()):
        pytest.skip("Verified test pair files not found on disk")

    with rasterio.open(p_a) as src_a:
        img_a = src_a.read(1)
    with rasterio.open(p_b) as src_b:
        img_b = src_b.read(1)

    matches = run_structural_matching(img_a, img_b)

    assert matches.ndim == 2
    assert matches.shape[1] == 5
    assert len(matches) > 0, "Structural matcher should find correspondences on verified test pair"
    # Verify confidences in [0.0, 1.0]
    assert np.all(matches[:, 4] >= 0.0) and np.all(matches[:, 4] <= 1.0)


def test_structural_matching_stress_pair():
    """Document measured behavior of classical vs structural matching on harsh illumination stress pair."""
    p_a = root_dir / "data" / "samples" / "stress_a.tif"
    p_b = root_dir / "data" / "samples" / "stress_b.tif"

    if not (p_a.exists() and p_b.exists()):
        pytest.skip("Illumination stress test pair files not found on disk")

    with rasterio.open(p_a) as src_a:
        img_a = src_a.read(1)
    with rasterio.open(p_b) as src_b:
        img_b = src_b.read(1)

    m_struct = run_structural_matching(img_a, img_b)
    m_class = run_dense_matching(img_a, img_b, method="classical")

    assert m_struct.ndim == 2 and m_struct.shape[1] == 5
    assert m_class.ndim == 2 and m_class.shape[1] == 5

    # Document empirical counts in test report
    print(f"\nStress pair correspondences: Classical SIFT={len(m_class)}, Structural MIM={len(m_struct)}")
    assert len(m_struct) > 0, "Structural matching should produce correspondences on stress pair"
    assert len(m_class) > 0, "Classical matching should produce correspondences on stress pair"
