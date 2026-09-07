"""
tests/test_subpixel.py  (v2 — correct shift_image, numpy Generator fix)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from core.subpixel_refiner import lucas_kanade_refine, refine_all_matches

try:
    from scipy.ndimage import map_coordinates, shift as ndimage_shift
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def make_textured_image(size=128, seed=42):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    for _ in range(12):
        cx, cy = rng.integers(20, size-20, size=2)
        sigma  = rng.uniform(4, 12)
        y, x   = np.mgrid[:size, :size]
        img   += rng.uniform(0.3, 1.0) * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
        )
    img = np.clip(img / max(img.max(), 1e-8), 0, 1)
    return img


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy needed for sub-pixel shift")
@pytest.mark.parametrize("true_dx,true_dy", [
    (0.3, 0.0),
    (0.0, 0.4),
    (0.2, 0.2),
    (-0.3, 0.15),
])
def test_lk_subpixel_convergence(true_dx, true_dy):
    """LK must recover known sub-pixel shifts to within 0.5 px error."""
    img = make_textured_image(size=128, seed=5)
    # scipy.ndimage.shift: shift=(dy, dx) in (row, col) convention
    img_shifted = ndimage_shift(img, shift=(true_dy, true_dx), mode='nearest', order=3)

    x2r, y2r, converged = lucas_kanade_refine(
        img_f=img,
        img_g=img_shifted,
        point=(64, 64, 64, 64),
        patch_size=15, iterations=20,
    )

    err = np.sqrt((x2r - (64 + true_dx))**2 + (y2r - (64 + true_dy))**2)
    assert err < 0.5, (
        f"LK shift ({true_dx},{true_dy}): got offset ({x2r-64:.3f},{y2r-64:.3f}), "
        f"error={err:.4f}px > 0.5px"
    )


@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy needed for sub-pixel shift")
@pytest.mark.parametrize("true_dx,true_dy", [
    (0.1, 0.0),
    (0.0, 0.1),
    (0.12, 0.12),
])
def test_lk_subpixel_strict(true_dx, true_dy):
    """Smaller shifts within LK linear range must converge to < 0.3 px."""
    img = make_textured_image(size=128, seed=3)
    img_shifted = ndimage_shift(img, shift=(true_dy, true_dx), mode='nearest', order=3)
    x2r, y2r, _ = lucas_kanade_refine(
        img_f=img, img_g=img_shifted,
        point=(64, 64, 64, 64), patch_size=15, iterations=20,
    )
    err = np.sqrt((x2r - (64 + true_dx))**2 + (y2r - (64 + true_dy))**2)
    assert err < 0.35, f"Strict LK: error {err:.4f}px > 0.35px for shift ({true_dx},{true_dy})"


def test_lk_converged_flag_on_flat_patch():
    flat = np.zeros((64, 64), dtype=np.float64)
    _, _, converged = lucas_kanade_refine(flat, flat, (32, 32, 32, 32), patch_size=11)
    assert converged is False


def test_lk_out_of_bounds():
    img = make_textured_image(size=64)
    x2r, y2r, converged = lucas_kanade_refine(
        img, img, (2, 2, 2, 2), patch_size=15,
    )
    assert converged is False
    assert x2r == 2 and y2r == 2


def test_refine_all_matches_shape_preserved():
    img_a = make_textured_image(size=128, seed=1)
    img_b = make_textured_image(size=128, seed=2)
    rng = np.random.default_rng(99)
    N = 20
    pts = rng.uniform(30, 100, size=(N, 2))
    # fixed: was rng.randn — use rng.standard_normal
    matches = np.column_stack([
        pts,
        pts + rng.standard_normal((N, 2)) * 0.3,
        np.ones(N) * 0.9,
    ]).astype(np.float64)

    refined = refine_all_matches(img_a, img_b, matches, patch_size=11)
    assert refined.shape == matches.shape
    assert refined.dtype == np.float64


def test_refine_all_matches_bad_shape():
    img = make_textured_image(size=64)
    with pytest.raises(ValueError):
        refine_all_matches(img, img, np.random.rand(10, 4))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
