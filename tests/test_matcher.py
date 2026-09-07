"""
tests/test_matcher.py
=====================
Unit and integration tests for core/dense_matcher.py.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.dense_matcher import (
    run_dense_matching,
    select_matcher,
    load_roma_model,
)


def make_synthetic_crater_pair(size=(256, 256)):
    """Generate a synthetic crater image pair with known displacement."""
    rng = np.random.default_rng(42)
    H, W = size
    img_a = rng.uniform(0.2, 0.8, (H, W)).astype(np.float64)

    # Draw synthetic circular crater rim
    yy, xx = np.ogrid[:H, :W]
    crater_mask = ((xx - 128)**2 + (yy - 128)**2 < 40**2)
    img_a[crater_mask] = 0.1

    # Shift image B by dx=5, dy=3
    img_b = np.roll(img_a, shift=(3, 5), axis=(0, 1))
    img_b += rng.normal(0, 0.02, (H, W))
    return img_a, img_b


def test_dense_matcher_contract_shape_and_dtype():
    """Matches output MUST be (N, 5) float64: [x1, y1, x2, y2, confidence]."""
    img_a, img_b = make_synthetic_crater_pair()
    matches = run_dense_matching(img_a, img_b)

    assert isinstance(matches, np.ndarray)
    assert matches.dtype == np.float64
    assert matches.ndim == 2
    assert matches.shape[1] == 5, f"Expected (N, 5) contract format; got {matches.shape}"
    assert len(matches) > 0, "Dense matcher must return candidate correspondences."


def test_match_confidence_and_bounds():
    """Confidence scores must fall in [0.0, 1.0] and coordinates within image bounds."""
    img_a, img_b = make_synthetic_crater_pair((128, 128))
    matches = run_dense_matching(img_a, img_b)

    x1 = matches[:, 0]
    y1 = matches[:, 1]
    x2 = matches[:, 2]
    y2 = matches[:, 3]
    conf = matches[:, 4]

    assert (conf >= 0.0).all() and (conf <= 1.0).all()
    assert (x1 >= 0).all() and (x1 < 128).all()
    assert (y1 >= 0).all() and (y1 < 128).all()


def test_select_matcher():
    """select_matcher selects SIFT for extreme scale differences and LoFTR otherwise."""
    assert select_matcher(4.0) == 'sift'
    assert select_matcher(0.2) == 'sift'
    assert select_matcher(1.2) == 'loftr'


def test_roma_stub_raises_not_implemented():
    """RoMa model loader raises NotImplementedError as specified in hackathon contract."""
    with pytest.raises(NotImplementedError):
        load_roma_model()
