"""
core/graph_renderer.py
======================
Render scientific chart and quality assessment artifacts for LUNA-MATCH.
All functions write directly to a PNG file using Matplotlib's non-interactive "Agg" backend.
Guaranteed to never raise on empty/degenerate input and never leak figure memory.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt

from core.crater_detection import bucket_diameter_histogram


def render_residual_scatter(
    reprojected_pts: Optional[np.ndarray],
    reference_pts: Optional[np.ndarray],
    out_path: Union[str, Path],
) -> None:
    """
    Scatter plot of reprojection residuals.
    X axis: reference point index. Y axis: residual magnitude in px.
    If reprojected_pts or reference_pts is empty/None: render a
    placeholder chart with text "No data — registration produced
    0 inliers." Never raise on empty input.
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig) — always, even on empty input.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        has_data = False
        if reprojected_pts is not None and reference_pts is not None:
            rep = np.asarray(reprojected_pts)
            ref = np.asarray(reference_pts)
            if rep.size > 0 and ref.size > 0 and len(rep) == len(ref):
                has_data = True
                if rep.ndim == 1 and ref.ndim == 1:
                    residuals = np.abs(ref - rep)
                else:
                    residuals = np.linalg.norm(ref[:, :2] - rep[:, :2], axis=1)

        if has_data and len(residuals) > 0:
            indices = np.arange(len(residuals))
            ax.scatter(
                indices,
                residuals,
                color="#00d4ff",
                s=36,
                alpha=0.85,
                edgecolors="#1a1a2e",
                linewidths=0.5,
                label=f"Points (n={len(residuals)})",
            )
            mean_res = float(np.mean(residuals))
            ax.axhline(
                mean_res,
                color="#ff6b6b",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean = {mean_res:.3f} px",
            )
            ax.set_xlabel("Reference Point Index", color="white", fontsize=11)
            ax.set_ylabel("Residual Magnitude (px)", color="white", fontsize=11)
            ax.set_title("Reprojection Residual Scatter", color="white", fontsize=12, fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("gray")
            ax.grid(True, linestyle="--", alpha=0.3, color="gray")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", loc="upper right")
        else:
            ax.text(
                0.5,
                0.5,
                "No data — registration produced\n0 inliers.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("gray")
            ax.set_title("Reprojection Residual Scatter", color="white", fontsize=12, fontweight="bold")

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def render_residual_histogram(
    residual_magnitudes: Optional[Union[np.ndarray, List[float]]],
    out_path: Union[str, Path],
) -> None:
    """
    Histogram of per-point residual magnitudes.
    20 bins, x-label "Residual (px)", y-label "Count".
    Vertical dashed line at 0.5px labeled "sub-pixel threshold".
    Empty input -> placeholder chart, never raise.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        has_data = False
        if residual_magnitudes is not None:
            res = np.asarray(residual_magnitudes).ravel()
            if res.size > 0:
                has_data = True

        if has_data:
            ax.hist(
                res,
                bins=20,
                color="#00d4ff",
                edgecolor="#1a1a2e",
                alpha=0.85,
                label=f"Residuals (n={len(res)})",
            )
            ax.axvline(
                0.5,
                color="#ff6b6b",
                linestyle="--",
                linewidth=2.0,
                label="sub-pixel threshold",
            )
            ax.set_xlabel("Residual (px)", color="white", fontsize=11)
            ax.set_ylabel("Count", color="white", fontsize=11)
            ax.set_title("Per-Point Residual Magnitude Distribution", color="white", fontsize=12, fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("gray")
            ax.grid(True, linestyle="--", alpha=0.3, color="gray")
            ax.legend(facecolor="#1a1a2e", labelcolor="white", loc="upper right")
        else:
            ax.text(
                0.5,
                0.5,
                "No data — registration produced\n0 inliers.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("gray")
            ax.set_title("Per-Point Residual Magnitude Distribution", color="white", fontsize=12, fontweight="bold")

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def render_crater_histogram(
    craters: Optional[List[Dict[str, Any]]],
    out_path: Union[str, Path],
) -> None:
    """
    Bar chart of crater diameter buckets.
    Reuse bucket_diameter_histogram() from core/crater_detection.py
    for the bucketing — do NOT recompute bucketing logic here.
    Buckets: '<1km','1-3km','3-10km','>10km'.
    Empty craters list -> placeholder chart "No craters detected."
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    try:
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        if craters is not None and len(craters) > 0:
            buckets = bucket_diameter_histogram(craters)
            bucket_keys = ["<1km", "1-3km", "3-10km", ">10km"]
            counts = [buckets.get(k, 0) for k in bucket_keys]
            colors = ["#00d4ff", "#4cc9f0", "#7209b7", "#f72585"]

            bars = ax.bar(bucket_keys, counts, color=colors, edgecolor="#1a1a2e", linewidth=1.2)
            ax.set_xlabel("Crater Diameter Class", color="white", fontsize=11)
            ax.set_ylabel("Count", color="white", fontsize=11)
            ax.set_title(f"Crater Diameter Distribution (n={len(craters)})", color="white", fontsize=12, fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("gray")
            ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")

            for bar, val in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.15,
                    str(val),
                    ha="center",
                    va="bottom",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No craters detected.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("gray")
            ax.set_title("Crater Diameter Distribution", color="white", fontsize=12, fontweight="bold")

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def render_confidence_gauge(
    grade: str,
    confidence_label: str,
    out_path: Union[str, Path],
) -> None:
    """
    Simple colored bar or gauge showing grade (A/B/C/D/F).
    Color map: A=green, B=yellowgreen, C=orange, D=orangered, F=red.
    Title: grade + " — " + confidence_label.
    This function ONLY renders a grade already computed by
    summary_builder.py — it never derives or changes the grade itself.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    color_map: Dict[str, str] = {
        "A": "green",
        "B": "yellowgreen",
        "C": "orange",
        "D": "orangered",
        "F": "red",
    }

    normalized_grade = (grade or "F").strip().upper()
    active_color = color_map.get(normalized_grade, "gray")
    title_text = f"{normalized_grade} — {confidence_label}"

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    try:
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        grades = ["A", "B", "C", "D", "F"]
        y_pos = np.zeros(len(grades))
        x_pos = np.arange(len(grades))

        # Render horizontal segments for each grade
        for idx, g in enumerate(grades):
            is_active = (g == normalized_grade)
            bar_color = color_map.get(g, "gray") if is_active else "#2d2d4e"
            alpha = 1.0 if is_active else 0.4
            edge_color = "white" if is_active else "gray"
            edge_width = 2.0 if is_active else 0.5

            ax.bar(
                idx,
                0.6,
                bottom=0.2,
                color=bar_color,
                alpha=alpha,
                edgecolor=edge_color,
                linewidth=edge_width,
                width=0.85,
            )
            ax.text(
                idx,
                0.5,
                g,
                ha="center",
                va="center",
                color="white",
                fontsize=16,
                fontweight="bold" if is_active else "normal",
            )

        ax.set_xlim(-0.6, len(grades) - 0.4)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(title_text, color="white", fontsize=13, fontweight="bold", pad=15)

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
