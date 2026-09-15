"""
core/crater_detection.py
========================
Classical, Deterministic Lunar Crater Detection from Phase Congruency Edge Maps.

Uses the maximum-moment edge map (M_psi) already computed during Phase Congruency
feature extraction (which highlights crater rims illumination-invariantly).
Detects circular and elliptical crater structures via contour extraction, ellipse fitting,
aspect ratio filtering, circularity checks, and metric scaling via sensor GSD.

Zero machine learning dependencies, deterministic, fast, and physically interpretable.
"""

import math
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def detect_craters(
    edge_moment_map: np.ndarray,
    gsd_m_per_px: float = 1.0,
    min_diameter_px: int = 8,
    max_diameter_px: int = 500,
    min_aspect_ratio: float = 0.60,
    confidence_thresh: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Detect approximately circular and elliptical crater structures from an edge moment map.

    Parameters
    ----------
    edge_moment_map : (H, W) float64/float32 array of edge moment energies (M_psi)
    gsd_m_per_px    : Ground Sampling Distance in meters/pixel for physical diameter scaling
    min_diameter_px : minimum crater diameter to consider in pixels
    max_diameter_px : maximum crater diameter to consider in pixels
    min_aspect_ratio: minimum minor/major axis ratio (craters near nadir are close to circular)
    confidence_thresh: minimum combined confidence score [0.0, 1.0]

    Returns
    -------
    craters : list of dicts, each describing a detected crater:
        {
            "center_px": (x, y),
            "radius_px": float,
            "diameter_px": float,
            "diameter_m": float,
            "aspect_ratio": float,
            "confidence": float,
        }
    """
    if edge_moment_map is None or edge_moment_map.size == 0:
        return []

    H, W = edge_moment_map.shape[:2]
    if H < min_diameter_px or W < min_diameter_px:
        return []

    # 1. Bounds and Flat/Noise Field Rejection
    e_min, e_max = float(np.nanmin(edge_moment_map)), float(np.nanmax(edge_moment_map))
    if e_max <= e_min or math.isnan(e_max) or (e_max - e_min) < 1e-6:
        return []

    # Noise rejection via Peak-to-Average Power Ratio (PAPR):
    # Real edge moment maps have PAPR >= 20-60; pure Gaussian noise has PAPR < 3.0
    mean_val = float(np.nanmean(edge_moment_map))
    if mean_val > 0:
        papr = e_max / mean_val
        if papr < 4.0:
            return []

    # Percentile-based normalization into uint8 [0, 255]
    p99 = float(np.percentile(edge_moment_map, 99.0))
    if p99 <= 1e-8:
        norm_edge = ((edge_moment_map - e_min) / (e_max - e_min) * 255.0).astype(np.uint8)
    else:
        norm_edge = np.clip(edge_moment_map / p99 * 255.0, 0, 255).astype(np.uint8)

    effective_max_dia = min(max_diameter_px, int(min(H, W) * 0.85))
    min_r = max(min_diameter_px // 2, 3)
    max_r = effective_max_dia // 2

    candidates = []

    # 2. Stage 1: Classical Hough Circle Transform
    # Invariant to directional illumination gradients; accumulates circular rim evidence
    circles = cv2.HoughCircles(
        norm_edge, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(min_r * 2, 14),
        param1=50, param2=24, minRadius=min_r, maxRadius=max_r
    )
    if circles is not None and len(circles[0]) > 0:
        thetas = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        for (cx, cy, r) in circles[0]:
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            dia = float(r * 2.0)
            if dia < min_diameter_px or dia > effective_max_dia:
                continue

            sx = np.clip(np.round(cx + r * np.cos(thetas)).astype(int), 0, W - 1)
            sy = np.clip(np.round(cy + r * np.sin(thetas)).astype(int), 0, H - 1)
            rim_mean = float(np.mean(norm_edge[sy, sx])) / 255.0

            inner_r = max(r * 0.5, 1.0)
            inx = np.clip(np.round(cx + inner_r * np.cos(thetas)).astype(int), 0, W - 1)
            iny = np.clip(np.round(cy + inner_r * np.sin(thetas)).astype(int), 0, H - 1)
            in_mean = float(np.mean(norm_edge[iny, inx])) / 255.0

            contrast = max(0.0, rim_mean - in_mean)
            conf = float(np.clip(0.50 * rim_mean + 0.50 * (contrast / max(rim_mean, 1e-4)), 0.05, 0.99))

            if conf >= confidence_thresh:
                candidates.append({
                    "center_px": (round(float(cx), 2), round(float(cy), 2)),
                    "radius_px": round(float(r), 2),
                    "diameter_px": round(float(dia), 2),
                    "diameter_m": round(float(dia * gsd_m_per_px), 2),
                    "aspect_ratio": 1.0,
                    "confidence": round(float(conf), 3),
                })

    # 3. Stage 2: Contour Fitting on thresholded edge moment map
    thresh_val, _ = cv2.threshold(norm_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eff_thresh = max(int(thresh_val * 0.70), 30)
    _, binary = cv2.threshold(norm_edge, eff_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if len(cnt) < 5:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw >= W - 8 or bh >= H - 8:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < math.pi * min_diameter_px * 0.7 or perim > math.pi * effective_max_dia * 2.0:
            continue
        try:
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
        except Exception:
            continue
        d_maj, d_min = max(d1, d2), min(d1, d2)
        if d_maj <= 0 or d_min <= 0:
            continue
        dia = (d_maj + d_min) / 2.0
        ar = d_min / d_maj
        if dia < min_diameter_px or dia > effective_max_dia or ar < min_aspect_ratio:
            continue
        if not (0 <= cx < W and 0 <= cy < H):
            continue

        cnt_area = cv2.contourArea(cnt)
        ell_area = math.pi * (d_maj / 2.0) * (d_min / 2.0)
        area_ratio = min(cnt_area, ell_area) / max(cnt_area, ell_area, 1e-4)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, 1)
        mean_edge = float(np.mean(norm_edge[mask > 0])) / 255.0
        conf = float(np.clip(0.45 * area_ratio + 0.35 * mean_edge + 0.20 * ar, 0.05, 0.99))

        if conf >= confidence_thresh:
            candidates.append({
                "center_px": (round(float(cx), 2), round(float(cy), 2)),
                "radius_px": round(float(dia / 2.0), 2),
                "diameter_px": round(float(dia), 2),
                "diameter_m": round(float(dia * gsd_m_per_px), 2),
                "aspect_ratio": round(float(ar), 3),
                "confidence": round(float(conf), 3),
            })

    # 4. Non-Maximum Suppression (deduplicate overlapping/concentric candidates)
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    selected: List[Dict[str, Any]] = []

    for cand in candidates:
        cx1, cy1 = cand["center_px"]
        r1 = cand["radius_px"]
        is_duplicate = False

        for existing in selected:
            cx2, cy2 = existing["center_px"]
            r2 = existing["radius_px"]
            dist = math.hypot(cx1 - cx2, cy1 - cy2)

            overlap_radius = max(r1, r2)
            if dist < 0.40 * overlap_radius and abs(r1 - r2) < 0.50 * overlap_radius:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(cand)

    logger.info(f"Crater detection: found {len(selected)} verified craters (from {len(candidates)} candidates).")
    return selected


def render_crater_overlay(
    image: np.ndarray,
    craters: List[Dict[str, Any]],
    line_thickness: int = 2,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Render detected crater ellipses and physical diameter labels on a copy of the input image.

    Parameters
    ----------
    image          : (H, W) or (H, W, 3) image
    craters        : list of crater dicts from detect_craters()
    line_thickness : pixel thickness of the ellipse outlines
    font_scale     : text scale for labels

    Returns
    -------
    overlay : (H, W, 3) uint8 BGR annotated image
    """
    if image.dtype != np.uint8:
        im_min, im_max = float(np.nanmin(image)), float(np.nanmax(image))
        if im_max > im_min:
            u8 = ((image - im_min) / (im_max - im_min) * 255.0).astype(np.uint8)
        else:
            u8 = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        u8 = image.copy()

    if u8.ndim == 2:
        canvas = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    elif u8.ndim == 3 and u8.shape[2] == 1:
        canvas = cv2.cvtColor(u8[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        canvas = u8.copy()

    for c in craters:
        cx, cy = int(round(c["center_px"][0])), int(round(c["center_px"][1]))
        r = int(round(c["radius_px"]))
        conf = c.get("confidence", 0.5)

        # Color: Emerald green for high confidence, Cyan/Amber for moderate
        color = (0, 235, 100) if conf >= 0.55 else (255, 200, 0)

        # Draw circle/ellipse
        cv2.circle(canvas, (cx, cy), r, color, line_thickness, cv2.LINE_AA)
        # Center crosshair dot
        cv2.circle(canvas, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # Label: Diameter in meters (or km if >= 1000m)
        d_m = c.get("diameter_m", c["diameter_px"])
        if d_m >= 1000:
            label = f"{d_m/1000.0:.1f}km"
        else:
            label = f"{int(round(d_m))}m"

        # Text position: slightly above the circle
        tx = max(5, cx - 20)
        ty = max(15, cy - r - 4)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(canvas, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), (10, 10, 10), -1)
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
