"""
core/overlap_check.py
======================
Stage 0 — Footprint Overlap Pre-Check (fast-fail guard).

Runs before any expensive computation (FFT, Phase Congruency, LoFTR/SIFT).
Rejects non-overlapping or implausible image pairs in milliseconds.

Specification:
  - If geospatial bounds exist (CRS + transform on both), compute bounding-box
    intersection over union (IoU) directly — O(1), exact.
  - Else (no georeferencing): downsample both images to a common small size
    (64x64) and compute normalized cross-correlation (NCC) peak location +
    magnitude as a coarse overlap proxy.
  - Fails fast with OverlapTooLowError ("OVERLAP_TOO_LOW") before Stage 2.
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
from scipy import signal

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

logger = logging.getLogger(__name__)


class OverlapTooLowError(ValueError):
    """Raised when estimated overlap between images is below required threshold."""
    def __init__(self, message: str = "Estimated footprint overlap is below required threshold"):
        super().__init__(message)
        self.error_code = "OVERLAP_TOO_LOW"


def _extract_geospatial_bounds(
    meta: Any,
    image_shape: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract (min_x, min_y, max_x, max_y) in projected/geospatial units if CRS
    and a non-trivial affine transform are present. Returns None otherwise.
    """
    if meta is None:
        return None

    crs = getattr(meta, "crs", None)
    if not crs or str(crs).strip().lower() in ("none", "", "null"):
        return None

    transform = getattr(meta, "transform", None)
    if transform is None:
        return None

    # Check for identity transform (indicating absence of genuine georeferencing)
    if hasattr(transform, "is_identity") and transform.is_identity:
        return None

    h, w = image_shape[:2]
    try:
        # Affine multiplication: (x, y) = transform * (px, py)
        # or transform.c + transform.a * px + transform.b * py
        if hasattr(transform, "__mul__"):
            corners = [
                transform * (0, 0),
                transform * (w, 0),
                transform * (w, h),
                transform * (0, h),
            ]
        elif hasattr(transform, "a") and hasattr(transform, "c"):
            corners = [
                (transform.c, transform.f),
                (transform.c + transform.a * w, transform.f + transform.d * w),
                (transform.c + transform.a * w + transform.b * h, transform.f + transform.d * w + transform.e * h),
                (transform.c + transform.b * h, transform.f + transform.e * h),
            ]
        else:
            return None

        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        return min(xs), min(ys), max(xs), max(ys)
    except Exception as e:
        logger.debug(f"Failed to extract geospatial bounds: {e}")
        return None


def _bounding_box_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """
    Compute Intersection-over-Union between two 2D axis-aligned bounding boxes
    specified as (min_x, min_y, max_x, max_y) in O(1).
    """
    min_x_a, min_y_a, max_x_a, max_y_a = box_a
    min_x_b, min_y_b, max_x_b, max_y_b = box_b

    inter_min_x = max(min_x_a, min_x_b)
    inter_min_y = max(min_y_a, min_y_b)
    inter_max_x = min(max_x_a, max_x_b)
    inter_max_y = min(max_y_a, max_y_b)

    inter_w = max(0.0, inter_max_x - inter_min_x)
    inter_h = max(0.0, inter_max_y - inter_min_y)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (max_x_a - min_x_a) * (max_y_a - min_y_a))
    area_b = max(0.0, (max_x_b - min_x_b) * (max_y_b - min_y_b))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def _ncc_overlap_proxy(
    img_a: np.ndarray,
    img_b: np.ndarray,
    target_dim: int = 64,
    meta_a: Any = None,
    meta_b: Any = None,
) -> float:
    """
    Downsample both images to target_dim x target_dim and compute normalized
    cross-correlation peak location and magnitude as a coarse overlap proxy.
    Accounts for scale disparity between images before resizing to target_dim.
    """
    # 1. Determine scale disparity ratio
    scale_ratio = 1.0
    gsd_a = getattr(meta_a, "gsd", None) if meta_a else None
    gsd_b = getattr(meta_b, "gsd", None) if meta_b else None
    if gsd_a is not None and gsd_b is not None:
        try:
            ga, gb = float(gsd_a), float(gsd_b)
            if ga > 0 and gb > 0 and abs(ga - gb) > 1e-4:
                scale_ratio = max(ga, gb) / min(ga, gb)
        except (ValueError, TypeError):
            pass

    # If GSD is not differentiating, compute from image dimensions (pixel_dimension_ratio logic)
    dim_a = max(img_a.shape[:2]) if img_a.ndim >= 2 else 1
    dim_b = max(img_b.shape[:2]) if img_b.ndim >= 2 else 1
    pixel_dim_ratio = float(max(dim_a, dim_b) / max(min(dim_a, dim_b), 1))

    if scale_ratio <= 1.05 and pixel_dim_ratio > 1.05:
        scale_ratio = pixel_dim_ratio

    # Rescale the larger image down to match the smaller image's effective scale before 64x64 resize
    a_proc = img_a
    b_proc = img_b
    if scale_ratio > 1.05:
        if dim_a > dim_b:
            scale_factor = float(dim_b) / float(dim_a)
            new_w = max(16, int(round(img_a.shape[1] * scale_factor)))
            new_h = max(16, int(round(img_a.shape[0] * scale_factor)))
            if _HAS_CV2:
                a_proc = cv2.resize(img_a.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif dim_b > dim_a:
            scale_factor = float(dim_a) / float(dim_b)
            new_w = max(16, int(round(img_b.shape[1] * scale_factor)))
            new_h = max(16, int(round(img_b.shape[0] * scale_factor)))
            if _HAS_CV2:
                b_proc = cv2.resize(img_b.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Downsample
    if _HAS_CV2:
        a_small = cv2.resize(
            a_proc.astype(np.float32),
            (target_dim, target_dim),
            interpolation=cv2.INTER_AREA,
        )
        b_small = cv2.resize(
            b_proc.astype(np.float32),
            (target_dim, target_dim),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sy_a = max(1, a_proc.shape[0] // target_dim)
        sx_a = max(1, a_proc.shape[1] // target_dim)
        a_small = a_proc[::sy_a, ::sx_a][:target_dim, :target_dim].astype(np.float32)
        sy_b = max(1, b_proc.shape[0] // target_dim)
        sx_b = max(1, b_proc.shape[1] // target_dim)
        b_small = b_proc[::sy_b, ::sx_b][:target_dim, :target_dim].astype(np.float32)

    # Normalize to zero mean and unit variance
    mean_a = np.mean(a_small)
    std_a = np.std(a_small)
    if std_a < 1e-6:
        return 0.0
    a_norm = (a_small - mean_a) / std_a

    mean_b = np.mean(b_small)
    std_b = np.std(b_small)
    if std_b < 1e-6:
        return 0.0
    b_norm = (b_small - mean_b) / std_b

    # Full 2D correlation via FFT
    # Cross-correlation between a and b is equivalent to conv2d(a, rot180(b))
    corr = signal.fftconvolve(
        a_norm,
        b_norm[::-1, ::-1],
        mode="full",
    ) / float(target_dim * target_dim)

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_val = float(corr[peak_idx])

    # Offset relative to zero-shift center
    dy = peak_idx[0] - (target_dim - 1)
    dx = peak_idx[1] - (target_dim - 1)

    # Geometric overlap fraction of two target_dim x target_dim frames offset by (dx, dy)
    overlap_w = max(0, target_dim - abs(dx))
    overlap_h = max(0, target_dim - abs(dy))
    geom_overlap = (overlap_w * overlap_h) / float(target_dim * target_dim)

    # Estimated overlap score:
    # If peak_val is noise floor (< 0.06), confidence is 0.0.
    # Between 0.06 and 0.15, smoothly scale confidence to 1.0.
    conf = float(np.clip((peak_val - 0.06) / (0.15 - 0.06), 0.0, 1.0))
    overlap_fraction = geom_overlap * conf
    return float(np.clip(overlap_fraction, 0.0, 1.0))


def estimate_overlap(
    img_a: np.ndarray,
    img_b: np.ndarray,
    meta_a: Any = None,
    meta_b: Any = None,
) -> float:
    """
    Returns estimated overlap fraction [0.0 - 1.0].

    Method:
      If geospatial bounds exist (CRS + transform on both), compute
      bounding-box intersection over union directly — O(1), exact.
      Else (no georeferencing): downsample both images to a common small size
      (e.g. 64x64) and compute normalized cross-correlation peak location +
      magnitude as a coarse overlap proxy.
    """
    bounds_a = _extract_geospatial_bounds(meta_a, img_a.shape)
    bounds_b = _extract_geospatial_bounds(meta_b, img_b.shape)

    # Compare CRSs if both bounds are present
    if bounds_a is not None and bounds_b is not None:
        crs_a = str(getattr(meta_a, "crs", "")).strip().upper()
        crs_b = str(getattr(meta_b, "crs", "")).strip().upper()
        if crs_a == crs_b or not crs_a or not crs_b:
            iou = _bounding_box_iou(bounds_a, bounds_b)
            logger.info(f"Stage 0: Geospatial O(1) bounding-box IoU = {iou:.4f}")
            return float(iou)

    # Fallback to downsampled NCC peak proxy
    proxy = _ncc_overlap_proxy(img_a, img_b, target_dim=64, meta_a=meta_a, meta_b=meta_b)
    logger.info(f"Stage 0: Downsampled NCC overlap proxy = {proxy:.4f}")
    return float(proxy)


def overlap_gate(overlap_fraction: float, min_required: float = 0.15) -> None:
    """
    Raises OverlapTooLowError if overlap_fraction < min_required.
    """
    if overlap_fraction < min_required:
        raise OverlapTooLowError(
            f"OVERLAP_TOO_LOW: Estimated footprint overlap {overlap_fraction:.4f} is below required minimum {min_required:.4f}"
        )
