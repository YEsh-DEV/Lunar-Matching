"""
tests/test_phase_congruency.py  (v2)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from core.phase_congruency_mim import (
    phase_congruency, _phase_congruency_per_orientation,
    moment_analysis, compute_mim, extract_structural_features,
    log_gabor_filter_bank, compute_even_odd_responses,
)


def make_crater_image(size=128, radius=30, noise_std=0.0):
    y, x = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    img = np.where(dist < radius, 0.8, 0.1).astype(np.float64)
    if noise_std > 0:
        img += np.random.randn(size, size) * noise_std
        img = np.clip(img, 0.0, 1.0)
    return img


def test_brightness_invariance():
    """PC must be identical under uniform brightness shift."""
    img = make_crater_image(size=64)
    img_bright = np.clip(img + 0.4, 0.0, 1.0)
    pc1 = phase_congruency(img,        n_scales=2, n_orientations=4)
    pc2 = phase_congruency(img_bright, n_scales=2, n_orientations=4)
    corr = np.corrcoef(pc1.ravel(), pc2.ravel())[0, 1]
    assert corr > 0.97, f"Brightness invariance failed: corr={corr:.4f}"


def test_contrast_invariance():
    """PC must be near-identical under contrast scaling."""
    img = make_crater_image(size=64)
    pc1 = phase_congruency(img,         n_scales=2, n_orientations=4)
    pc2 = phase_congruency(img * 0.3,   n_scales=2, n_orientations=4)
    corr = np.corrcoef(pc1.ravel(), pc2.ravel())[0, 1]
    assert corr > 0.97, f"Contrast invariance failed: corr={corr:.4f}"


def test_pc_range():
    img = make_crater_image(size=64)
    pc  = phase_congruency(img, n_scales=2, n_orientations=4)
    assert pc.min() >= 0.0 and pc.max() <= 1.0


def test_pc_shape():
    img = make_crater_image(size=64)
    pc  = phase_congruency(img, n_scales=2, n_orientations=4)
    assert pc.shape == img.shape


def test_moment_analysis_edge_at_rim():
    img = make_crater_image(size=64, radius=20)
    pc_per_o, amp_per_o = _phase_congruency_per_orientation(img, n_scales=2, n_orientations=4)
    M_edges, m_corners  = moment_analysis(pc_per_o)
    y, x = np.mgrid[:64, :64]
    cx, cy = 32, 32
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    rim_mask   = (dist >= 18) & (dist <= 22)
    plain_mask = dist < 10
    assert M_edges[rim_mask].mean() > M_edges[plain_mask].mean(), \
        "Edge energy should be higher at rim than flat interior"


def test_mim_dtype_and_shape():
    img = make_crater_image(size=64)
    _, amp_per_o = _phase_congruency_per_orientation(img, n_scales=2, n_orientations=4)
    mim = compute_mim(amp_per_o)
    assert mim.shape == img.shape
    assert mim.dtype == np.uint8
    assert mim.max() < 4


def test_extract_structural_features_keys():
    img  = make_crater_image(size=64)
    out  = extract_structural_features(img, n_scales=2, n_orientations=4)
    assert {'pc_map', 'mim', 'edge_map', 'corner_map', 'amp_per_o'} == set(out.keys())


def test_log_gabor_dc_zero():
    """
    After ifftshift, the DC component (index [0,0]) of the FFT-convention filter
    must be zero. We verify this by checking the [0,0] element of each filter.
    """
    fb = log_gabor_filter_bank((64, 64), n_scales=2, n_orientations=4)
    for s, scale in enumerate(fb):
        for o, filt in enumerate(scale):
            # The DC energy of the filter (its effect on the image mean) is
            # sum(filt) / N, which equals filt[0,0] in the FFT convention.
            dc = filt[0, 0]
            assert abs(dc) < 1e-7, (
                f"Filter bank DC[{s},{o}]={dc:.2e} is not near-zero. "
                "Log-Gabor filters must zero DC to avoid mean-shift sensitivity."
            )


def test_pc_edge_stronger_than_interior():
    """PC values at the hard edge must exceed smooth interior regions."""
    img = make_crater_image(size=64, radius=20)
    pc  = phase_congruency(img, n_scales=2, n_orientations=4)
    y, x = np.mgrid[:64, :64]
    dist = np.sqrt((x - 32)**2 + (y - 32)**2)
    rim_pc     = pc[(dist >= 18) & (dist <= 22)].mean()
    interior_pc = pc[dist < 8].mean()
    assert rim_pc > interior_pc, \
        f"PC rim={rim_pc:.4f} should exceed interior={interior_pc:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
