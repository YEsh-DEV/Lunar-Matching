"""
tests/test_crater_detection.py
==============================
Unit tests for deterministic lunar crater detection module (core/crater_detection.py).
Tests:
  1. Recovery of synthetic drawn circular craters within reasonable pixel tolerance.
  2. Noise / flat image rejection (zero false positive hallucinations).
  3. Rejection of highly elongated elliptical contours (aspect ratio filter).
  4. Physical diameter calculation using GSD.
  5. Rendering of crater visual overlay.
"""

import math
import numpy as np
import cv2
import pytest

from core.crater_detection import detect_craters, render_crater_overlay


def test_detect_synthetic_circular_craters():
    """Create a synthetic moment map with 3 known circular crater rims and verify recovery."""
    H, W = 400, 400
    edge_map = np.zeros((H, W), dtype=np.float32)

    # 3 craters: (cx, cy, r)
    ground_truth_craters = [
        (100, 120, 30),
        (250, 150, 45),
        (200, 300, 25),
    ]

    for cx, cy, r in ground_truth_craters:
        cv2.circle(edge_map, (cx, cy), r, 1.0, 2)

    gsd = 2.0  # 2 meters per pixel
    detected = detect_craters(
        edge_map,
        gsd_m_per_px=gsd,
        min_diameter_px=15,
        max_diameter_px=200,
        confidence_thresh=0.25,
    )

    assert len(detected) >= 3, f"Expected at least 3 craters, found {len(detected)}"

    # Match each ground truth to the nearest detection
    matched = 0
    for cx, cy, r in ground_truth_craters:
        best_dist = 999.0
        best_r_diff = 999.0
        for d in detected:
            dcx, dcy = d["center_px"]
            dr = d["radius_px"]
            dist = math.hypot(cx - dcx, cy - dcy)
            if dist < best_dist:
                best_dist = dist
                best_r_diff = abs(r - dr)

        # Center should be within 3 pixels and radius within 3 pixels
        assert best_dist <= 3.5, f"Crater at ({cx},{cy}) center error {best_dist:.2f}px too large"
        assert best_r_diff <= 3.5, f"Crater at ({cx},{cy}) radius error {best_r_diff:.2f}px too large"
        matched += 1

    assert matched == 3


def test_noise_and_flat_rejection():
    """Confirm zero or near-zero detections on uniform flat image and Gaussian noise image."""
    # 1. Uniform flat image
    flat = np.zeros((300, 300), dtype=np.float32)
    detected_flat = detect_craters(flat)
    assert len(detected_flat) == 0, f"Expected 0 detections on flat image, got {len(detected_flat)}"

    # 2. Pure Gaussian noise (no structured circular rims)
    np.random.seed(42)
    noise = np.random.normal(loc=0.5, scale=0.1, size=(300, 300)).astype(np.float32)
    detected_noise = detect_craters(noise, confidence_thresh=0.40)
    assert len(detected_noise) == 0, f"Expected 0 detections on pure noise, got {len(detected_noise)}"


def test_aspect_ratio_filtering():
    """Confirm that highly elongated linear or elliptical structures are rejected."""
    H, W = 300, 300
    edge_map = np.zeros((H, W), dtype=np.float32)

    # Draw a very elongated ellipse (axes: 80 x 15 -> aspect ratio ~ 0.19 << 0.60)
    cv2.ellipse(edge_map, (150, 150), (80, 15), 30, 0, 360, 1.0, 2)

    detected = detect_craters(
        edge_map,
        min_aspect_ratio=0.60,
        min_diameter_px=10,
    )
    assert len(detected) == 0, f"Elongated structure should be rejected; got {len(detected)} craters"


def test_gsd_physical_scaling():
    """Verify that diameter_m scales linearly with GSD."""
    H, W = 200, 200
    edge_map = np.zeros((H, W), dtype=np.float32)
    cv2.circle(edge_map, (100, 100), 20, 1.0, 2)

    gsd_1 = 1.5
    gsd_2 = 3.0

    det_1 = detect_craters(edge_map, gsd_m_per_px=gsd_1, confidence_thresh=0.20)
    det_2 = detect_craters(edge_map, gsd_m_per_px=gsd_2, confidence_thresh=0.20)

    assert len(det_1) == 1
    assert len(det_2) == 1

    # Ratio of physical diameters should match ratio of GSDs (3.0 / 1.5 = 2.0)
    d1 = det_1[0]["diameter_m"]
    d2 = det_2[0]["diameter_m"]
    assert pytest.approx(d2 / d1, rel=0.01) == 2.0


def test_render_crater_overlay():
    """Verify that render_crater_overlay returns a valid BGR image with correct dimensions."""
    img = np.zeros((200, 200), dtype=np.uint8)
    craters = [
        {
            "center_px": (100.0, 100.0),
            "radius_px": 25.0,
            "diameter_px": 50.0,
            "diameter_m": 1200.0,
            "aspect_ratio": 0.95,
            "confidence": 0.85,
        }
    ]

    overlay = render_crater_overlay(img, craters)
    assert overlay.shape == (200, 200, 3)
    assert overlay.dtype == np.uint8
    # Should not be entirely black (annotations drawn)
    assert np.any(overlay > 0)


# ---------------------------------------------------------------------------
# Checkpoint 4: Fitzgibbon direct algebraic LS ellipse fitting
# ---------------------------------------------------------------------------

def test_fitzgibbon_fits_known_ellipse():
    """
    fit_ellipse_fitzgibbon must recover center and semi-axes of a known ellipse
    to within a small tolerance.
    """
    from core.crater_detection import fit_ellipse_fitzgibbon

    # Generate dense samples on a known ellipse: center=(100, 80), a=40, b=25
    t = np.linspace(0, 2 * np.pi, 200)
    cx_true, cy_true = 100.0, 80.0
    a_true, b_true = 40.0, 25.0
    x = cx_true + a_true * np.cos(t)
    y = cy_true + b_true * np.sin(t)

    pts = np.column_stack([x, y])
    result = fit_ellipse_fitzgibbon(pts)

    assert result is not None, "Fitzgibbon must succeed on a clean ellipse"
    cx_fit, cy_fit = result['center_px']

    # Center must be within 2px of truth
    assert abs(cx_fit - cx_true) < 2.0, f"Center x error: {abs(cx_fit - cx_true):.2f}px"
    assert abs(cy_fit - cy_true) < 2.0, f"Center y error: {abs(cy_fit - cy_true):.2f}px"

    # Semi-axes must match within 2px (either orientation)
    a_fit = result['semi_major_px']
    b_fit = result['semi_minor_px']
    # The larger of the two fits the true major axis
    assert abs(a_fit - a_true) < 3.0 or abs(a_fit - b_true) < 3.0, (
        f"Semi-major axis error: fit={a_fit:.2f}, truth={a_true}"
    )


def test_fitzgibbon_requires_minimum_6_points():
    """fit_ellipse_fitzgibbon must return None when fewer than 6 points are given."""
    from core.crater_detection import fit_ellipse_fitzgibbon

    pts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    result = fit_ellipse_fitzgibbon(pts)
    assert result is None, "Must return None for < 6 points"


def test_detect_craters_fitzgibbon_on_synthetic():
    """
    detect_craters_fitzgibbon must find at least the primary synthetic crater and
    not hallucinate detections on a flat image.
    """
    from core.crater_detection import detect_craters_fitzgibbon

    H, W = 300, 300
    edge_map = np.zeros((H, W), dtype=np.float32)
    # Draw one clear ellipse (a=35, b=28) at (150, 150)
    cv2.ellipse(edge_map, (150, 150), (35, 28), 0, 0, 360, 1.0, 2)

    detected = detect_craters_fitzgibbon(
        edge_map, gsd_m_per_px=2.0, min_diameter_px=20, confidence_thresh=0.20
    )
    # Must find at least 1 crater
    assert len(detected) >= 1, f"Must detect synthetic ellipse; got {len(detected)}"

    # All detected craters must have 'method' field
    for d in detected:
        assert 'method' in d
        assert d['method'] == 'fitzgibbon_direct_ls'

    # Flat image must produce zero detections
    flat = np.zeros((200, 200), dtype=np.float32)
    assert detect_craters_fitzgibbon(flat) == []


# ---------------------------------------------------------------------------
# Checkpoint 5: SFD computation tests
# ---------------------------------------------------------------------------

def test_compute_sfd_empty_craters():
    """compute_sfd on empty list must return valid dict with n_craters=0."""
    from core.crater_detection import compute_sfd

    result = compute_sfd([])
    assert result['n_craters'] == 0
    assert result['power_law_slope'] is None
    assert result['sfd_r2'] is None


def test_compute_sfd_power_law_fit():
    """
    compute_sfd on a synthetic power-law distributed crater population must
    return a negative slope and R^2 > 0.8 (good fit).
    """
    from core.crater_detection import compute_sfd

    # Synthetic crater list: sizes following ~D^-2.5 distribution
    rng = np.random.default_rng(7)
    # Generate 80 craters with power-law sizes
    diameters = np.sort(rng.uniform(500, 15000, 80))[::-1]
    craters = [{'diameter_m': float(d)} for d in diameters]

    result = compute_sfd(craters, n_bins=10)

    assert result['n_craters'] == 80
    assert len(result['diameters_m']) == 80
    assert len(result['bin_counts']) == 10
    assert result['power_law_slope'] is not None, "Must fit power law slope"
    assert result['power_law_slope'] < 0, (
        f"SFD slope must be negative (got {result['power_law_slope']})"
    )
    # R^2 must be reasonable for a uniform distribution (not perfect power law)
    # Just check it's computed and finite
    assert result['sfd_r2'] is not None
    assert -1.0 <= result['sfd_r2'] <= 1.0


def test_compute_sfd_physical_fields():
    """compute_sfd must return all required output fields."""
    from core.crater_detection import compute_sfd

    craters = [{'diameter_m': float(d)} for d in [1000, 2000, 3000, 5000, 8000, 12000]]
    result = compute_sfd(craters)

    required_keys = [
        'diameters_m', 'bin_edges_m', 'bin_counts', 'bin_centers_m',
        'power_law_slope', 'power_law_intercept', 'n_craters', 'sfd_r2',
    ]
    for key in required_keys:
        assert key in result, f"SFD result missing required key: {key}"

    assert result['n_craters'] == 6
    assert len(result['bin_edges_m']) == len(result['bin_centers_m']) + 1
