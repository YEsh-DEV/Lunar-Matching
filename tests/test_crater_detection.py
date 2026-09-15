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
