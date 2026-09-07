"""
core/warp_and_eval.py
=====================
Image Warping, GeoTIFF Export, and Quantitative Validation Metrics.

Implements:
  - Non-rigid TPS and projective homography backwards image warping
  - Duck-typing transform abstraction supporting ThinPlateSplineTransform
  - Root-Mean-Square Error (RMSE) computation
  - Inlier Ratio calculation
  - Spatial Distribution Index (SDI) via normalized Shannon entropy with log2(0) guard
  - 2D residual displacement error map visualization
  - PDS/GIS-compliant GeoTIFF export via rasterio

References:
  Bookstein, F. (1989). Principal Warps: Thin-Plate Splines.
  Shannon, C. E. (1948). A Mathematical Theory of Communication.
"""

import os
import math
import logging
from typing import Tuple, Optional, Union, Any, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Try importing rasterio
try:
    import rasterio
    from rasterio.transform import from_origin
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    logger.warning("rasterio not installed; export_geotiff will fall back to numpy/PIL.")

# Try importing OpenCV
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.warning("OpenCV not installed; warp_image will use scipy/numpy fallback.")


def warp_image(
    source_img: np.ndarray,
    transform: Any,
    ref_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Warp source_img into the coordinate frame of ref_shape.
    Duck-types on hasattr(transform, "apply") to support both ThinPlateSplineTransform
    and 3x3 homography matrix.

    Parameters
    ----------
    source_img : (H_s, W_s) source image to be warped
    transform  : ThinPlateSplineTransform (has .apply) or (3, 3) homography ndarray
    ref_shape  : target (H_r, W_r) shape

    Returns
    -------
    warped : (H_r, W_r) float64 registered image
    """
    H_r, W_r = ref_shape[:2]
    H_s, W_s = source_img.shape[:2]

    # Duck-type 1: Transform is a ThinPlateSplineTransform (or object with .apply)
    if hasattr(transform, "apply"):
        # Backward warping: for each grid coordinate (x, y) in ref, query source position
        # Create coordinate grid in reference space
        y_coords, x_coords = np.mgrid[0:H_r, 0:W_r]
        ref_points = np.column_stack([x_coords.ravel(), y_coords.ravel()]).astype(np.float64)

        # Chunked evaluation to keep memory footprint bounded
        chunk_size = 65536
        src_x = np.zeros(len(ref_points), dtype=np.float32)
        src_y = np.zeros(len(ref_points), dtype=np.float32)

        for i in range(0, len(ref_points), chunk_size):
            chunk = ref_points[i:i + chunk_size]
            warped_chunk = transform.apply(chunk)
            src_x[i:i + chunk_size] = warped_chunk[:, 0].astype(np.float32)
            src_y[i:i + chunk_size] = warped_chunk[:, 1].astype(np.float32)

        map_x = src_x.reshape((H_r, W_r))
        map_y = src_y.reshape((H_r, W_r))

        if _HAS_CV2:
            warped = cv2.remap(
                source_img.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return warped.astype(np.float64)
        else:
            # SciPy fallback
            from scipy.ndimage import map_coordinates
            coords = np.array([map_y, map_x])
            warped = map_coordinates(source_img, coords, order=1, mode='constant', cval=0.0)
            return warped.astype(np.float64)

    # Duck-type 2: Transform is a 3x3 Homography matrix
    H_matrix = np.asarray(transform, dtype=np.float64)
    if H_matrix.shape == (3, 3):
        if _HAS_CV2:
            # In OpenCV, warpPerspective dst(x, y) = src(M * [x, y, 1]) uses M directly with WARP_INVERSE_MAP
            warped = cv2.warpPerspective(
                source_img.astype(np.float32),
                H_matrix,
                (W_r, H_r),
                flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return warped.astype(np.float64)
        else:
            # Fallback identity/slice
            warped = np.zeros((H_r, W_r), dtype=np.float64)
            h_copy = min(H_r, H_s)
            w_copy = min(W_r, W_s)
            warped[:h_copy, :w_copy] = source_img[:h_copy, :w_copy]
            return warped

    logger.warning("Unrecognized transform format; returning source image resized to ref_shape.")
    if _HAS_CV2:
        return cv2.resize(source_img, (W_r, H_r), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    return source_img.copy()


def compute_rmse(*args, **kwargs) -> float:
    """
    Compute Root-Mean-Square Error (RMSE).
    Supports two calling conventions:
      1) compute_rmse(reprojected_pts, reference_pts) -> reprojection RMSE in pixels
      2) compute_rmse(refined_matches, registered, img_a, transform=...) -> [Orchestrator format]
         Evaluates genuine reprojection residual error ||T(pts_a) - pts_b||_2 across verified inliers.
    """
    transform = kwargs.get('transform', None)

    if len(args) == 2:
        reprojected_pts, reference_pts = args
        pts_rep = np.asarray(reprojected_pts, dtype=np.float64)
        pts_ref = np.asarray(reference_pts, dtype=np.float64)
        if len(pts_rep) == 0 or len(pts_ref) == 0:
            return 0.0
        diff = pts_rep - pts_ref
        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        return round(rmse, 4)

    if len(args) == 3:
        refined_matches, registered, img_a = args
        if isinstance(refined_matches, np.ndarray) and refined_matches.ndim == 2 and refined_matches.shape[1] >= 4 and len(refined_matches) > 0:
            pts_a = refined_matches[:, :2].astype(np.float64)
            pts_b = refined_matches[:, 2:4].astype(np.float64)

            # If a transform (homography matrix or TPS) is provided, project pts_a to frame B
            if transform is not None:
                if hasattr(transform, 'apply'):
                    pts_b_proj = transform.apply(pts_a)
                elif isinstance(transform, np.ndarray) and transform.shape == (3, 3):
                    ones = np.ones((len(pts_a), 1), dtype=np.float64)
                    pts_h = np.hstack([pts_a, ones])
                    proj = (transform @ pts_h.T).T
                    w = proj[:, 2:3] + 1e-10
                    pts_b_proj = proj[:, :2] / w
                else:
                    pts_b_proj = pts_a
                diff = pts_b_proj - pts_b
            elif len(pts_a) >= 4 and _HAS_CV2:
                # Estimate least-squares homography to measure reprojection consistency
                H, _ = cv2.findHomography(pts_a, pts_b, 0)
                if H is not None:
                    ones = np.ones((len(pts_a), 1), dtype=np.float64)
                    pts_h = np.hstack([pts_a, ones])
                    proj = (H @ pts_h.T).T
                    w = proj[:, 2:3] + 1e-10
                    pts_b_proj = proj[:, :2] / w
                    diff = pts_b_proj - pts_b
                else:
                    diff = pts_a - pts_b
            else:
                diff = pts_a - pts_b

            rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            return round(rmse, 4)
        else:
            diff = registered - img_a
            rmse = float(np.sqrt(np.mean(diff**2)))
            return round(rmse, 4)

    raise ValueError(f"compute_rmse expects 2 or 3 positional arguments; received {len(args)}")


def compute_inlier_ratio(*args, **kwargs) -> float:
    """
    Calculate ratio of geometrically verified inliers.
    Supports:
      1) compute_inlier_ratio(inlier_mask) -> float
      2) compute_inlier_ratio(n_inliers, n_candidates) -> float
    """
    if len(args) == 1:
        mask = np.asarray(args[0], dtype=bool)
        if len(mask) == 0:
            return 0.0
        return round(float(np.mean(mask)), 4)

    if len(args) == 2:
        n_inliers, n_candidates = args
        if n_candidates <= 0:
            return 0.0
        return round(float(n_inliers / n_candidates), 4)

    raise ValueError(f"compute_inlier_ratio expects 1 or 2 positional arguments; received {len(args)}")


def compute_sdi(
    match_points: np.ndarray,
    image_shape: Tuple[int, int],
    grid: int = 8,
) -> float:
    """
    Compute Spatial Distribution Index (SDI) across image canvas using normalized Shannon entropy:
      H = - sum_{i=1}^{M} p_i * log2(p_i)
      SDI = H / log2(M)

    Guards against log2(0) by treating empty bins (p_i = 0) as 0 contribution.

    Parameters
    ----------
    match_points : (N, 2) or (N, >=4) match array
    image_shape  : (H, W) canvas shape
    grid         : grid subdivision (default 8x8 = 64 cells)

    Returns
    -------
    sdi : float in range [0.0, 1.0] (1.0 = perfectly uniform spatial distribution)
    """
    pts = np.asarray(match_points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0

    if pts.ndim == 2 and pts.shape[1] >= 2:
        xs = pts[:, 0]
        ys = pts[:, 1]
    else:
        return 0.0

    H, W = image_shape[:2]
    M = grid * grid
    H_max = math.log2(M) if M > 1 else 1.0

    # Bin points into grid
    dx = max(W / grid, 1e-5)
    dy = max(H / grid, 1e-5)

    bin_x = np.clip((xs / dx).astype(int), 0, grid - 1)
    bin_y = np.clip((ys / dy).astype(int), 0, grid - 1)
    bin_indices = bin_y * grid + bin_x

    counts = np.bincount(bin_indices, minlength=M)
    total = np.sum(counts)

    if total == 0:
        return 0.0

    # Probabilities p_i
    p = counts / total

    # Shannon entropy with strict log2(0) guard: only sum over p_i > 0
    non_zero = p > 0
    entropy = -np.sum(p[non_zero] * np.log2(p[non_zero]))

    sdi = float(entropy / H_max)
    return round(float(np.clip(sdi, 0.0, 1.0)), 4)


def generate_residual_error_map(
    reprojected_pts: np.ndarray,
    reference_pts: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Generate a 2D spatial residual error magnitude heatmap across the image canvas.

    Parameters
    ----------
    reprojected_pts : (N, 2) array of reprojected points
    reference_pts   : (N, 2) array of reference ground truth / matched points
    image_shape     : (H, W) canvas shape

    Returns
    -------
    residual_map : (H, W) float64 interpolated residual error map
    """
    H, W = image_shape[:2]
    pts_rep = np.asarray(reprojected_pts, dtype=np.float64)
    pts_ref = np.asarray(reference_pts, dtype=np.float64)

    residuals = np.linalg.norm(pts_rep - pts_ref, axis=1)

    if len(residuals) < 4:
        # Uniform low residual canvas if few points
        return np.zeros((H, W), dtype=np.float64)

    try:
        from scipy.interpolate import Rbf
        rbf = Rbf(pts_ref[:, 0], pts_ref[:, 1], residuals, function='linear', smooth=0.5)
        grid_y, grid_x = np.mgrid[0:H:16, 0:W:16]
        dense_res = rbf(grid_x, grid_y)
        if _HAS_CV2:
            full_map = cv2.resize(dense_res, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            full_map = np.repeat(np.repeat(dense_res, 16, axis=0), 16, axis=1)[:H, :W]
        return np.clip(full_map, 0.0, None).astype(np.float64)
    except Exception as e:
        logger.warning(f"Residual error map RBF failed ({e}); returning zero canvas.")
        return np.zeros((H, W), dtype=np.float64)


def export_geotiff(
    warped_img: np.ndarray,
    *args,
    **kwargs,
) -> None:
    """
    Export registered warped raster to GeoTIFF format with spatial referencing.

    Supports both:
      1) export_geotiff(warped_img, path, meta) [Orchestrator format]
      2) export_geotiff(warped_img, meta, path) [Prompt format]
    """
    path = None
    meta = None

    if len(args) == 2:
        a1, a2 = args
        if isinstance(a1, (str, os.PathLike)):
            path = a1
            meta = a2
        else:
            meta = a1
            path = a2
    elif 'out_path' in kwargs:
        path = kwargs['out_path']
        meta = kwargs.get('ref_meta', None)
    elif 'path' in kwargs:
        path = kwargs['path']
        meta = kwargs.get('meta', None)

    if path is None:
        raise ValueError("export_geotiff requires a destination file path.")

    out_path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    H, W = warped_img.shape[:2]

    # Attempt GeoTIFF export via rasterio
    if _HAS_RASTERIO and out_path.endswith(('.tif', '.tiff')):
        try:
            crs = meta.get('crs', 'EPSG:4326') if hasattr(meta, 'get') else getattr(meta, 'crs', 'EPSG:4326')
            transform = meta.get('transform', None) if hasattr(meta, 'get') else getattr(meta, 'transform', None)
            if transform is None:
                gsd = meta.get('gsd', 0.5) if hasattr(meta, 'get') else getattr(meta, 'gsd', 0.5)
                transform = from_origin(0.0, float(H * gsd), float(gsd), float(gsd))

            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=H,
                width=W,
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(warped_img.astype(np.float32), 1)

            logger.info(f"Successfully exported GeoTIFF to: {out_path}")
            return
        except Exception as e:
            logger.warning(f"rasterio export failed ({e}); falling back to npy.")

    # Fallback: save as .npy
    npy_path = out_path.replace('.tif', '.npy').replace('.tiff', '.npy')
    np.save(npy_path, warped_img)
    logger.info(f"Saved warped image as numpy array to: {npy_path}")


def build_metrics_report(
    rmse: float,
    inlier_ratio: float,
    sdi: float,
    n_inliers: Optional[int] = None,
    n_total: Optional[int] = None,
    elapsed_s: Optional[float] = None,
    transform: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format quantitative registration metrics into a standardized report dictionary.
    """
    return {
        'rmse_px': round(float(rmse), 4),
        'inlier_ratio': round(float(inlier_ratio), 4),
        'sdi': round(float(sdi), 4),
        'n_inliers': int(n_inliers) if n_inliers is not None else 0,
        'n_total': int(n_total) if n_total is not None else 0,
        'elapsed_s': round(float(elapsed_s), 2) if elapsed_s is not None else 0.0,
        'transform': str(transform) if transform is not None else "unknown",
    }
