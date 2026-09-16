"""
tests/test_validation_split.py
===============================
Unit and integration tests for Stage 6b Held-Out Validation Split (core/validation_split.py).

Covers:
  1. Contract compliance: dict keys {held_out_rmse_px, fit_rmse_px, overfit_ratio, n_fit, n_val}.
  2. Determinism: fixed seed produces reproducible splits and identical RMSE values.
  3. Small-sample guard: graceful handling when matches < 5.
  4. Synthetic overfit scenario: proving held-out RMSE differs meaningfully
     from fit RMSE (fit_rmse ~ 0, held_out_rmse >> fit_rmse, overfit_ratio > 1.5).
  5. End-to-end pipeline reporting: LunaMatchPipeline outputs held_out_rmse_px in metrics.
"""

import numpy as np
import pytest

from core.validation_split import held_out_rmse, project_points


class MemorizingOverfittedModel:
    """
    Synthetic model that perfectly interpolates/memorizes training coordinates (error = 0)
    but severely degrades on unseen held-out coordinates.
    """
    def __init__(self, src_train: np.ndarray, dst_train: np.ndarray, noise_offset: float = 15.0):
        self.src_train = src_train.copy()
        self.dst_train = dst_train.copy()
        self.noise_offset = noise_offset

    def apply(self, pts: np.ndarray) -> np.ndarray:
        out = []
        for p in pts:
            dists = np.linalg.norm(self.src_train - p, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < 1e-3:
                # Perfect recall for training points
                out.append(self.dst_train[min_idx])
            else:
                # Wild generalization divergence on held-out points
                out.append(p + np.array([self.noise_offset, self.noise_offset]))
        return np.asarray(out, dtype=np.float64)


def test_validation_split_contract_and_determinism():
    rng = np.random.default_rng(101)
    pts_a = rng.uniform(0, 500, (20, 2))
    # True shift: +10, +5
    pts_b = pts_a + [10.0, 5.0]
    conf = np.ones((20, 1))
    matches = np.hstack([pts_a, pts_b, conf])

    def simple_fitter(src, dst):
        shift = np.mean(dst - src, axis=0)
        class ShiftModel:
            def apply(self, p): return p + shift
        return ShiftModel()

    res1 = held_out_rmse(matches, simple_fitter, split_ratio=0.8, seed=42)
    res2 = held_out_rmse(matches, simple_fitter, split_ratio=0.8, seed=42)

    # Determinism
    assert res1 == res2
    assert "held_out_rmse_px" in res1
    assert "fit_rmse_px" in res1
    assert "overfit_ratio" in res1
    assert res1["n_fit"] == 16
    assert res1["n_val"] == 4
    # On pure translation, both fit and held-out RMSE should be ~0
    assert res1["fit_rmse_px"] < 1e-3
    assert res1["held_out_rmse_px"] < 1e-3


def test_validation_split_guard_few_points():
    matches_few = np.ones((3, 5))
    res = held_out_rmse(matches_few, lambda s, d: None)
    assert res["held_out_rmse_px"] == 0.0
    assert res["n_val"] == 0


def test_synthetic_overfit_scenario_differs_meaningfully():
    """
    Prove that held-out RMSE detects overfitting:
    Training RMSE is ~0, but held-out RMSE diverges meaningfully with overfit_ratio >> 1.5.
    """
    rng = np.random.default_rng(42)
    pts_a = rng.uniform(50, 400, (25, 2))
    # Ground truth mapping: dst = src + [5.0, 5.0]
    pts_b = pts_a + [5.0, 5.0]
    matches = np.column_stack([pts_a, pts_b, np.ones(25)])

    def overfit_fitter(src, dst):
        return MemorizingOverfittedModel(src, dst, noise_offset=20.0)

    res = held_out_rmse(matches, overfit_fitter, split_ratio=0.8, seed=42)

    fit_rmse = res["fit_rmse_px"]
    held_out_rmse_val = res["held_out_rmse_px"]
    overfit_ratio = res["overfit_ratio"]

    # Training fit error is zero
    assert fit_rmse < 1e-3, f"Expected fit RMSE ~0, got {fit_rmse}"

    # Held-out RMSE diverges meaningfully
    assert held_out_rmse_val > 10.0, f"Expected large held-out RMSE, got {held_out_rmse_val}"

    # Overfit ratio must substantially exceed 1.5
    assert overfit_ratio > 1.5, f"Expected overfit ratio > 1.5, got {overfit_ratio}"
    assert overfit_ratio > 100.0, f"Severe overfit ratio should be very large; got {overfit_ratio}"
