"""
tests/test_graph_renderer.py
=============================
Unit tests for core/graph_renderer.py:
Verifies all 4 rendering functions handle empty/degenerate inputs gracefully
and generate valid PNG image files on disk.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from core.graph_renderer import (
    render_residual_scatter,
    render_residual_histogram,
    render_crater_histogram,
    render_confidence_gauge,
)


def _assert_valid_png(path: Path) -> None:
    assert path.exists(), f"Expected file {path} was not created."
    assert path.stat().st_size > 0, f"Expected non-empty file {path}."
    with Image.open(str(path)) as img:
        assert img.format == "PNG", f"Expected PNG format, got {img.format}"
        assert img.size[0] > 0 and img.size[1] > 0


def test_residual_scatter_empty_input(tmp_path):
    """Empty arrays -> file is created, no raise, valid PNG."""
    out_file = tmp_path / "scatter_empty.png"
    render_residual_scatter(np.empty((0, 2)), np.empty((0, 2)), str(out_file))
    _assert_valid_png(out_file)

    # Also test None input
    out_file_none = tmp_path / "scatter_none.png"
    render_residual_scatter(None, None, str(out_file_none))
    _assert_valid_png(out_file_none)


def test_residual_scatter_real_data(tmp_path):
    """20 random points -> file created, valid PNG."""
    out_file = tmp_path / "scatter_real.png"
    rng = np.random.default_rng(42)
    pts_ref = rng.uniform(50, 450, (20, 2))
    pts_rep = pts_ref + rng.normal(0, 0.5, (20, 2))
    render_residual_scatter(pts_rep, pts_ref, str(out_file))
    _assert_valid_png(out_file)


def test_residual_histogram_empty(tmp_path):
    """Empty -> file created, no raise, valid PNG."""
    out_file = tmp_path / "hist_empty.png"
    render_residual_histogram(np.empty(0), str(out_file))
    _assert_valid_png(out_file)

    # Also test None
    out_file_none = tmp_path / "hist_none.png"
    render_residual_histogram(None, str(out_file_none))
    _assert_valid_png(out_file_none)


def test_residual_histogram_real(tmp_path):
    """50 random residuals -> file created, valid PNG."""
    out_file = tmp_path / "hist_real.png"
    rng = np.random.default_rng(42)
    residuals = np.abs(rng.normal(0.4, 0.2, 50))
    render_residual_histogram(residuals, str(out_file))
    _assert_valid_png(out_file)


def test_crater_histogram_empty(tmp_path):
    """[] -> file created, no raise, valid PNG."""
    out_file = tmp_path / "crater_empty.png"
    render_crater_histogram([], str(out_file))
    _assert_valid_png(out_file)

    # Also test None
    out_file_none = tmp_path / "crater_none.png"
    render_crater_histogram(None, str(out_file_none))
    _assert_valid_png(out_file_none)


def test_crater_histogram_real(tmp_path):
    """List of 5 craters with diameter_m -> file created, valid PNG."""
    out_file = tmp_path / "crater_real.png"
    craters = [
        {"diameter_m": 450.0, "x": 100, "y": 100},
        {"diameter_m": 1200.0, "x": 150, "y": 200},
        {"diameter_m": 2500.0, "x": 250, "y": 300},
        {"diameter_m": 5000.0, "x": 300, "y": 400},
        {"diameter_m": 15000.0, "x": 400, "y": 450},
    ]
    render_crater_histogram(craters, str(out_file))
    _assert_valid_png(out_file)


def test_confidence_gauge_all_grades(tmp_path):
    """For each grade A/B/C/D/F -> file created, valid PNG."""
    grades_and_labels = [
        ("A", "high confidence"),
        ("B", "moderate confidence"),
        ("C", "low confidence — sparse inliers"),
        ("D", "unreliable — synthetic fallback active"),
        ("F", "failed"),
    ]
    for grade, label in grades_and_labels:
        out_file = tmp_path / f"gauge_{grade}.png"
        render_confidence_gauge(grade, label, str(out_file))
        _assert_valid_png(out_file)
