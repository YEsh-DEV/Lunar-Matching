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
        px = np.clip(cnt[:, 0, 0], 0, W - 1)
        py = np.clip(cnt[:, 0, 1], 0, H - 1)
        mean_edge = float(np.mean(norm_edge[py, px])) / 255.0
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


# ---------------------------------------------------------------------------
# Checkpoint 4 — Fitzgibbon Direct Algebraic Least-Squares Ellipse Fitting
# ---------------------------------------------------------------------------

def fit_ellipse_fitzgibbon(contour_points: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Fit an ellipse to a set of 2D contour points using the Direct Algebraic
    Least-Squares method (Fitzgibbon, Pilu & Fisher, 1996).

    Solves: min ||D * a||^2  subject to  a^T C a = 1
    where D is the design matrix [x^2, xy, y^2, x, y, 1] and C is the
    constraint matrix enforcing ellipticity (4AC - B^2 > 0).

    Advantages over cv2.fitEllipse (Bookstein's method):
      - Guaranteed elliptic solution (never returns hyperbola/parabola)
      - More robust to noise and partial crater rim occlusion
      - Closed-form linear solution — deterministic and fast

    Parameters
    ----------
    contour_points : (N, 2) float array of (x, y) contour coordinates.
                     Minimum 6 points required.

    Returns
    -------
    dict with keys {center_px, semi_major_px, semi_minor_px, angle_deg, aspect_ratio}
    or None if fitting fails.

    References
    ----------
    Fitzgibbon, A., Pilu, M., & Fisher, R.B. (1996). Direct least-squares
    fitting of ellipses. CVPR.
    """
    pts = np.asarray(contour_points, dtype=np.float64)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if len(pts) < 6:
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    # Normalize for numerical stability
    x_c = x.mean()
    y_c = y.mean()
    scale = max(np.std(x), np.std(y), 1e-6)
    xn = (x - x_c) / scale
    yn = (y - y_c) / scale

    # Design matrix D: each row is [x^2, xy, y^2, x, y, 1]
    D = np.column_stack([xn**2, xn * yn, yn**2, xn, yn, np.ones_like(xn)])

    # Scatter matrix S = D^T D
    S = D.T @ D

    # Constraint matrix C for ellipse (4AC - B^2 > 0 constraint)
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 2] = 2.0
    C[2, 0] = 2.0
    C[1, 1] = -1.0

    # Solve generalized eigenproblem: S a = lambda C a
    try:
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(S + 1e-8 * np.eye(6), C))
    except np.linalg.LinAlgError:
        return None

    # Select eigenvector with smallest positive eigenvalue (elliptic constraint)
    # Eigenvalue must be real and positive
    real_mask = np.abs(eigenvalues.imag) < 1e-6
    pos_mask = eigenvalues.real > 0
    valid = real_mask & pos_mask

    if not np.any(valid):
        return None

    valid_eigs = eigenvalues.real[valid]
    valid_vecs = eigenvectors[:, valid]

    best_idx = np.argmin(valid_eigs)
    a = valid_vecs[:, best_idx].real

    # Convert from normalized to original coordinate frame
    # a = [A, B, C, D, E, F] in normalized: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    A, B, C_coef, D_coef, E_coef, F_coef = a

    # De-normalize (expand back from (x-x_c)/scale, (y-y_c)/scale)
    # Original: A*(xn)^2 + ... = 0 where xn = (x-x_c)/scale
    s = scale
    A_ = A / s**2
    B_ = B / s**2
    C_ = C_coef / s**2
    D_ = (D_coef - 2*A_*x_c - B_*y_c) / s
    E_ = (E_coef - 2*C_coef/s**2 * y_c * s - B_/s * x_c * s) / s
    # Simpler: directly compute center using the algebraic form
    # Center: (2CD - BE) / (B^2 - 4AC), (2AE - BD) / (B^2 - 4AC)
    denom = B_**2 - 4 * A_ * C_
    if abs(denom) < 1e-10:
        return None

    # Recompute D_, E_, F_ using the scale-expanded coefficients
    A_, B_, C_, D_, E_, F_ = (
        A / s**2,
        B / s**2,
        C_coef / s**2,
        (-2*A*x_c/s**2 - B*y_c/s**2 + D_coef/s),
        (-2*C_coef*y_c/s**2 - B*x_c/s**2 + E_coef/s),
        (A*x_c**2/s**2 + B*x_c*y_c/s**2 + C_coef*y_c**2/s**2
         - D_coef*x_c/s - E_coef*y_c/s + F_coef),
    )

    denom = B_**2 - 4 * A_ * C_
    if abs(denom) < 1e-10:
        return None

    cx = (2 * C_ * D_ - B_ * E_) / denom
    cy = (2 * A_ * E_ - B_ * D_) / denom

    # Semi-axes computation via eigenvalue of the shape matrix
    M = np.array([[A_, B_/2], [B_/2, C_]])
    eigvals_shape = np.linalg.eigvalsh(M)
    if np.any(eigvals_shape == 0):
        return None

    num = -(A_ * cy**2 + C_ * cx**2 - B_ * cx * cy + F_ - B_ * cx * cy)
    # Use standard formula
    num = -(F_ - A_ * cx**2 - B_ * cx * cy - C_ * cy**2)
    if num == 0:
        return None

    ev = np.linalg.eigvalsh(M)
    if np.any(ev <= 0) or num <= 0:
        return None

    try:
        semi_a = float(np.sqrt(num / ev[0]))
        semi_b = float(np.sqrt(num / ev[1]))
    except (ValueError, RuntimeWarning):
        return None

    if semi_a <= 0 or semi_b <= 0:
        return None

    d_major = max(semi_a, semi_b)
    d_minor = min(semi_a, semi_b)

    # Rotation angle of major axis
    if abs(B_) < 1e-10:
        theta = 0.0 if A_ < C_ else 90.0
    else:
        theta = float(0.5 * np.degrees(np.arctan2(B_, A_ - C_)))

    ar = d_minor / d_major if d_major > 1e-6 else 0.0

    return {
        'center_px': (round(float(cx), 2), round(float(cy), 2)),
        'semi_major_px': round(float(d_major), 2),
        'semi_minor_px': round(float(d_minor), 2),
        'angle_deg': round(float(theta), 2),
        'aspect_ratio': round(float(ar), 3),
        'method': 'fitzgibbon_direct_ls',
    }


def detect_craters_fitzgibbon(
    edge_moment_map: np.ndarray,
    gsd_m_per_px: float = 1.0,
    min_diameter_px: int = 8,
    max_diameter_px: int = 500,
    min_aspect_ratio: float = 0.60,
    confidence_thresh: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Fitzgibbon-enhanced crater detection.

    Runs ALONGSIDE (NOT replacing) the existing detect_craters().
    Uses the same contour extraction as the existing Stage 2b but fits ellipses
    via direct algebraic LS (Fitzgibbon) instead of OpenCV's iterative method.

    Intended use: call detect_craters() for the primary pipeline result, and
    detect_craters_fitzgibbon() as an additive secondary run for cross-validation.
    Only included in the output metrics if it measurably improves center accuracy.

    Parameters
    ----------
    Same as detect_craters().

    Returns
    -------
    list of crater dicts. Same schema as detect_craters() plus 'method' field.
    """
    if edge_moment_map is None or edge_moment_map.size == 0:
        return []

    H, W = edge_moment_map.shape[:2]
    if H < min_diameter_px or W < min_diameter_px:
        return []

    e_min, e_max = float(np.nanmin(edge_moment_map)), float(np.nanmax(edge_moment_map))
    if e_max <= e_min or math.isnan(e_max) or (e_max - e_min) < 1e-6:
        return []

    mean_val = float(np.nanmean(edge_moment_map))
    if mean_val > 0 and e_max / mean_val < 4.0:
        return []

    p99 = float(np.percentile(edge_moment_map, 99.0))
    if p99 <= 1e-8:
        norm_edge = ((edge_moment_map - e_min) / (e_max - e_min) * 255.0).astype(np.uint8)
    else:
        norm_edge = np.clip(edge_moment_map / p99 * 255.0, 0, 255).astype(np.uint8)

    effective_max_dia = min(max_diameter_px, int(min(H, W) * 0.85))

    # Contour extraction (same as detect_craters Stage 2)
    thresh_val, _ = cv2.threshold(norm_edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eff_thresh = max(int(thresh_val * 0.70), 30)
    _, binary = cv2.threshold(norm_edge, eff_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for cnt in contours:
        if len(cnt) < 6:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw >= W - 8 or bh >= H - 8:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < math.pi * min_diameter_px * 0.7 or perim > math.pi * effective_max_dia * 2.0:
            continue

        pts_2d = cnt.reshape(-1, 2).astype(np.float64)
        result = fit_ellipse_fitzgibbon(pts_2d)
        if result is None:
            continue

        cx, cy = result['center_px']
        d_major = result['semi_major_px'] * 2.0
        d_minor = result['semi_minor_px'] * 2.0
        ar = result['aspect_ratio']
        dia = (d_major + d_minor) / 2.0

        if not (min_diameter_px <= dia <= effective_max_dia):
            continue
        if ar < min_aspect_ratio:
            continue
        if not (0 <= cx < W and 0 <= cy < H):
            continue

        cnt_area = cv2.contourArea(cnt)
        ell_area = math.pi * (d_major / 2.0) * (d_minor / 2.0)
        area_ratio = min(cnt_area, ell_area) / max(cnt_area, ell_area, 1e-4)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, 1)
        mean_edge = float(np.mean(norm_edge[mask > 0])) / 255.0 if mask.any() else 0.0
        conf = float(np.clip(0.45 * area_ratio + 0.35 * mean_edge + 0.20 * ar, 0.05, 0.99))

        if conf >= confidence_thresh:
            candidates.append({
                'center_px': (round(float(cx), 2), round(float(cy), 2)),
                'radius_px': round(float(dia / 2.0), 2),
                'diameter_px': round(float(dia), 2),
                'diameter_m': round(float(dia * gsd_m_per_px), 2),
                'aspect_ratio': round(float(ar), 3),
                'confidence': round(float(conf), 3),
                'method': 'fitzgibbon_direct_ls',
            })

    # NMS deduplicate
    candidates.sort(key=lambda c: c['confidence'], reverse=True)
    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        cx1, cy1 = cand['center_px']
        r1 = cand['radius_px']
        is_dup = False
        for ex in selected:
            cx2, cy2 = ex['center_px']
            r2 = ex['radius_px']
            if (math.hypot(cx1 - cx2, cy1 - cy2) < 0.40 * max(r1, r2)
                    and abs(r1 - r2) < 0.50 * max(r1, r2)):
                is_dup = True
                break
        if not is_dup:
            selected.append(cand)

    logger.info(
        f"Fitzgibbon crater detection: {len(selected)} craters (from {len(candidates)} candidates)."
    )
    return selected


# ---------------------------------------------------------------------------
# Checkpoint 5 — Crater Size-Frequency Distribution (SFD)
# ---------------------------------------------------------------------------

def compute_sfd(
    craters: List[Dict[str, Any]],
    n_bins: int = 12,
    min_diameter_m: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute Crater Size-Frequency Distribution (SFD) for a set of detected craters.

    The SFD power law N(D) = c * D^(-b) where b ≈ 2.0–3.2 is the standard
    planetary geology tool for relative lunar surface age dating (older surfaces
    have more accumulated small craters, steeper power-law slope).

    Parameters
    ----------
    craters        : list of crater dicts from detect_craters() (must have 'diameter_m')
    n_bins         : number of logarithmic diameter bins
    min_diameter_m : minimum diameter to include in analysis

    Returns
    -------
    dict with:
        'diameters_m'       : list of crater diameters in meters
        'bin_edges_m'       : log-spaced bin edges in meters
        'bin_counts'        : cumulative count N(>D) per bin (cumulative SFD)
        'bin_centers_m'     : geometric mean diameter of each bin
        'power_law_slope'   : fitted slope b from log-log regression (negative)
        'power_law_intercept': fitted log(c) intercept
        'n_craters'         : total number of craters used
        'sfd_r2'            : R² goodness of fit for the power law
    """
    if not craters:
        return {
            'diameters_m': [],
            'bin_edges_m': [],
            'bin_counts': [],
            'bin_centers_m': [],
            'power_law_slope': None,
            'power_law_intercept': None,
            'n_craters': 0,
            'sfd_r2': None,
        }

    diameters = np.array(
        [c['diameter_m'] for c in craters if c.get('diameter_m', 0) > min_diameter_m],
        dtype=np.float64
    )
    diameters = diameters[diameters > 0]

    if len(diameters) < 2:
        return {
            'diameters_m': diameters.tolist(),
            'bin_edges_m': [],
            'bin_counts': [],
            'bin_centers_m': [],
            'power_law_slope': None,
            'power_law_intercept': None,
            'n_craters': int(len(diameters)),
            'sfd_r2': None,
        }

    d_min = max(diameters.min(), 1.0)
    d_max = diameters.max() * 1.01

    bin_edges = np.logspace(np.log10(d_min), np.log10(d_max), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    # Cumulative SFD: N(>D) = number of craters with diameter > D_i
    cumulative_counts = np.array(
        [int(np.sum(diameters >= d)) for d in bin_edges[:-1]],
        dtype=np.float64
    )

    # Power law fit on log-log scale: log(N) = b * log(D) + log(c)
    # Only fit on bins with at least 1 crater
    valid = cumulative_counts > 0
    slope = intercept = r2 = None
    if np.sum(valid) >= 3:
        log_d = np.log10(bin_centers[valid])
        log_n = np.log10(cumulative_counts[valid])
        # Least-squares linear regression
        A = np.column_stack([log_d, np.ones_like(log_d)])
        try:
            result = np.linalg.lstsq(A, log_n, rcond=None)
            coeffs = result[0]
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
            # Compute R²
            log_n_pred = slope * log_d + intercept
            ss_res = float(np.sum((log_n - log_n_pred) ** 2))
            ss_tot = float(np.sum((log_n - log_n.mean()) ** 2))
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        except np.linalg.LinAlgError:
            pass

    return {
        'diameters_m': diameters.tolist(),
        'bin_edges_m': bin_edges.tolist(),
        'bin_counts': cumulative_counts.tolist(),
        'bin_centers_m': bin_centers.tolist(),
        'power_law_slope': round(slope, 4) if slope is not None else None,
        'power_law_intercept': round(intercept, 4) if intercept is not None else None,
        'n_craters': int(len(diameters)),
        'sfd_r2': round(r2, 4) if r2 is not None else None,
    }


def crater_density_per_km2(craters: List[Dict[str, Any]], area_km2: float) -> float:
    """Compute crater density per km² from a list of detected craters."""
    if area_km2 <= 0:
        return 0.0
    return round(len(craters) / area_km2, 4)


def bucket_diameter_histogram(craters: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Bucket craters into standard size classes by diameter_m.
    Buckets: '<1km', '1-3km', '3-10km', '>10km'.
    """
    buckets: Dict[str, int] = {'<1km': 0, '1-3km': 0, '3-10km': 0, '>10km': 0}
    for c in craters:
        d_m = c.get('diameter_m', 0)
        if d_m < 1000:
            buckets['<1km'] += 1
        elif d_m < 3000:
            buckets['1-3km'] += 1
        elif d_m < 10000:
            buckets['3-10km'] += 1
        else:
            buckets['>10km'] += 1
    return buckets
