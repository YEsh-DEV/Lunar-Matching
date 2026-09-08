"""
core/summary_builder.py
=======================
Unified Chatbot-Ready Output Schema Builder for LUNA-MATCH.

Assembles a clean, standardized, chatbot/VLM/RAG-friendly JSON payload
for any given registration job, combining:
  1. Standardized geodetic registration metrics (RMSE, inlier ratio, SDI, etc.)
  2. Plain-language quality classification & diagnostic reasoning
  3. Context-aware input metadata (GSDs, scale disparity, solar correction)
  4. Relative paths to all generated visual artifacts (registered GeoTIFF,
     checkerboard mosaic, tiepoint correspondence map, residual error map)

Schema Version: 1.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

# Optional imports for on-demand artifact generation
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

from core.ingest_preprocess import read_raster

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"


# ---------------------------------------------------------------------------
# Confidence & Quality Evaluation Thresholds
# ---------------------------------------------------------------------------

def classify_registration_quality(
    metrics: Dict[str, Any],
    input_metadata: Dict[str, Any],
) -> Tuple[str, str, str, list]:
    """
    Evaluate registration metrics and assign a defensible, plain-language
    confidence label, letter grade, reasoning narrative, and diagnostic warnings.

    Parameters
    ----------
    metrics : dictionary containing rmse_px, inlier_ratio, sdi, n_inliers, n_total, etc.
    input_metadata : dictionary containing scale_disparity_ratio, pixel_dimension_ratio, solar_correction_applied, etc.

    Returns
    -------
    confidence_label : str (e.g. "high confidence", "moderate confidence", "low confidence ...")
    grade            : str ("A", "B", "C", "D", "F")
    reasoning        : str explaining why this classification was assigned
    warnings         : list of string warning messages
    """
    status = metrics.get("status", "UNKNOWN")
    is_fallback = metrics.get("is_synthetic_fallback", False)
    warnings = []

    # Check solar correction
    if not input_metadata.get("solar_correction_applied", False):
        warnings.append("Solar ephemeris angles missing in metadata; Lommel-Seeliger illumination normalization was bypassed.")

    scale_ratio = float(input_metadata.get("scale_disparity_ratio", 1.0))
    pixel_ratio = float(input_metadata.get("pixel_dimension_ratio", 1.0))
    # Effective scale disparity accounts for both physical GSD and pixel dimension disparity
    effective_scale = max(scale_ratio, pixel_ratio)

    if effective_scale > 3.0:
        warnings.append(f"High scale disparity ({effective_scale:.2f}x) between input images.")

    if status != "DONE":
        reasoning = f"Registration did not complete successfully (current state: {status})."
        err = metrics.get("error")
        if err:
            reasoning += f" Error details: {err}"
        return "failed", "F", reasoning, warnings

    if is_fallback:
        warnings.append("Synthetic fallback matches were active during this job.")
        return (
            "unreliable — synthetic fallback active",
            "D",
            "Pipeline fell back to synthetic matching coordinates; metrics do not reflect genuine surface correspondence.",
            warnings,
        )

    raw_rmse = metrics.get("rmse_px")
    rmse = float(raw_rmse) if raw_rmse is not None else 999.0

    raw_ratio = metrics.get("inlier_ratio")
    inlier_ratio = float(raw_ratio) if raw_ratio is not None else 0.0

    raw_inliers = metrics.get("n_inliers")
    n_inliers = int(raw_inliers) if raw_inliers is not None else 0

    raw_sdi = metrics.get("sdi")
    sdi = float(raw_sdi) if raw_sdi is not None else 0.0

    transform = str(metrics.get("transform_type") or metrics.get("transform") or "homography")

    if n_inliers < 6:
        warnings.append(f"Marginal inlier count ({n_inliers} verified inliers); geometric model has minimal degrees-of-freedom redundancy.")

    if sdi < 0.20:
        warnings.append(f"Low spatial dispersion index (SDI={sdi:.4f}); keypoints are clustered in a localized subregion rather than distributed across the scene.")

    # 1. High Confidence
    # Requires sub-pixel RMSE (< 1.0 px), robust overdetermined inliers (>= 8 with >= 20% ratio or >= 12),
    # good spatial distribution (SDI >= 0.20), and effective scale disparity <= 3.5x.
    if (
        rmse <= 1.0
        and (
            (n_inliers >= 8 and inlier_ratio >= 0.20)
            or (n_inliers >= 12 and inlier_ratio >= 0.15)
            or (inlier_ratio >= 0.70 and n_inliers >= 6)
        )
        and sdi >= 0.20
        and effective_scale <= 3.5
    ):
        confidence_label = "high confidence"
        grade = "A"
        reasoning = (
            f"Sub-pixel reprojection accuracy (RMSE {rmse:.4f} px) with strong inlier verification "
            f"({n_inliers} inliers, {inlier_ratio*100:.1f}%) and solid spatial distribution (SDI {sdi:.4f}) "
            f"under a verified {transform} transformation."
        )

    # 2. Moderate Confidence
    # Acceptable planetary geodetic accuracy (RMSE <= 2.0 px), meets minimum 4-DOF homography constraint,
    # and effective scale ratio is manageable (<= 3.5x).
    elif rmse <= 2.0 and n_inliers >= 4 and inlier_ratio >= 0.10 and effective_scale <= 3.5:
        confidence_label = "moderate confidence"
        grade = "B"
        reasons = []
        if effective_scale > 2.5:
            reasons.append(f"moderate scale disparity ({effective_scale:.2f}x)")
        if n_inliers < 8:
            reasons.append(f"sparse inlier count ({n_inliers} inliers)")
        if sdi < 0.25:
            reasons.append(f"moderate spatial clustering (SDI {sdi:.4f})")
        reason_str = ", ".join(reasons) if reasons else "acceptable geodetic alignment"
        reasoning = (
            f"Registration succeeded with RMSE {rmse:.4f} px ({inlier_ratio*100:.1f}% inliers). "
            f"Confidence is moderate due to {reason_str}, but geometric solution is mathematically sound."
        )

    # 3. Low Confidence
    else:
        grade = "C"
        if effective_scale > 3.5 and n_inliers < 10:
            confidence_label = "low confidence — scale disparity may exceed matcher capability"
            reasoning = (
                f"Extreme scale difference ({effective_scale:.2f}x) between input images exceeds standard "
                f"feature scale invariance limits, resulting in only {n_inliers} verified inliers."
            )
        elif rmse > 2.0:
            confidence_label = "low confidence — high reprojection residual error"
            reasoning = (
                f"Reprojection RMSE ({rmse:.4f} px) exceeds acceptable planetary geodetic threshold (2.0 px), "
                f"suggesting non-planar terrain relief distortion or residual misalignments."
            )
        else:
            confidence_label = "low confidence — sparse surface inliers"
            reasoning = (
                f"Inlier count ({n_inliers}) or inlier ratio ({inlier_ratio*100:.1f}%) is below optimal "
                f"thresholds for robust geodetic alignment."
            )

    return confidence_label, grade, reasoning, warnings


def _parse_shape(shape_val: Any) -> Optional[Tuple[int, int]]:
    """Parse image shape from tuple, list, or string format e.g. '(975, 1600)'."""
    if isinstance(shape_val, (tuple, list)) and len(shape_val) >= 2:
        return int(shape_val[0]), int(shape_val[1])
    if isinstance(shape_val, str):
        cleaned = shape_val.strip("()[] ")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
    return None


def _to_relative_path(path: Union[str, Path], root: Path) -> str:
    """Convert absolute path to a clean relative path string relative to root."""
    try:
        p = Path(path)
        if p.is_absolute():
            return str(p.relative_to(root))
        return str(p)
    except Exception:
        return str(path)


def _ensure_visual_artifacts(job_dir: Path, root: Path) -> Dict[str, Optional[str]]:
    """
    Verify and locate visual artifacts in the job directory.
    Returns relative paths for registered GeoTIFF, checkerboard, tiepoints, and residual map.
    Renders missing lightweight previews if source files are available.
    """
    out_dir = job_dir / "output"
    inter_dir = job_dir / "intermediate"
    artifacts: Dict[str, Optional[str]] = {
        "registered_geotiff": None,
        "checkerboard_png": None,
        "tiepoints_png": None,
        "residual_map_png": None,
    }

    # 1. Registered GeoTIFF / Raster
    reg_tif = out_dir / "registered.tif"
    reg_npy = out_dir / "registered.npy"
    if reg_tif.exists():
        artifacts["registered_geotiff"] = _to_relative_path(reg_tif, root)
    elif reg_npy.exists():
        artifacts["registered_geotiff"] = _to_relative_path(reg_npy, root)

    # 2. Residual Map
    res_png = out_dir / "residual_map.png"
    if res_png.exists():
        artifacts["residual_map_png"] = _to_relative_path(res_png, root)
    else:
        # Check if we can generate it on the fly
        matches_ver = inter_dir / "matches_verified.npy"
        transform_file = inter_dir / "transform_params.json"
        if matches_ver.exists() and transform_file.exists() and _HAS_CV2:
            try:
                from core.warp_and_eval import generate_residual_error_map
                pts_ver = np.load(str(matches_ver))
                with open(transform_file, "r") as tf:
                    tdata = json.load(tf)
                H_mat = np.array(tdata.get("homography", np.eye(3)))
                if len(pts_ver) > 0 and pts_ver.shape[1] >= 4:
                    pts_a = pts_ver[:, :2]
                    pts_b = pts_ver[:, 2:4]
                    ones = np.ones((len(pts_a), 1), dtype=np.float64)
                    pts_h = np.hstack([pts_a, ones])
                    proj = (H_mat @ pts_h.T).T
                    pts_rep = proj[:, :2] / (proj[:, 2:3] + 1e-10)
                    
                    # Canvas shape
                    meta_path = job_dir / "input" / "metadata.json"
                    canvas_shape = (512, 512)
                    if meta_path.exists():
                        with open(meta_path, "r") as mf:
                            m = json.load(mf)
                            shape_str = m.get("image_a", {}).get("shape", "")
                            if "(" in shape_str:
                                parts = shape_str.strip("()").split(",")
                                canvas_shape = (int(parts[0]), int(parts[1]))
                    
                    res_map = generate_residual_error_map(pts_rep, pts_b, canvas_shape)
                    res_u8 = cv2.normalize(res_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    res_color = cv2.applyColorMap(res_u8, cv2.COLORMAP_JET)
                    cv2.imwrite(str(res_png), res_color)
                    artifacts["residual_map_png"] = _to_relative_path(res_png, root)
            except Exception as e:
                logger.debug(f"Could not generate residual map for {job_dir.name}: {e}")

    # 3. Checkerboard Overlay
    cb_png = out_dir / "preview_checkerboard.png"
    if not cb_png.exists():
        cb_png = out_dir / "checkerboard_overlay.png"
    if cb_png.exists():
        artifacts["checkerboard_png"] = _to_relative_path(cb_png, root)
    else:
        # Check if we can generate it
        paths_file = job_dir / "input" / "input_paths.json"
        if paths_file.exists() and (reg_tif.exists() or reg_npy.exists()) and _HAS_CV2:
            try:
                with open(paths_file, "r") as pf:
                    ip = json.load(pf)
                src_a = ip.get("img_a_path")
                if src_a and os.path.exists(src_a):
                    raw_a, _ = read_raster(src_a)
                    if reg_tif.exists() and _HAS_RASTERIO:
                        with rasterio.open(str(reg_tif)) as s:
                            reg_img = s.read(1).astype(np.float64)
                    else:
                        reg_img = np.load(str(reg_npy))
                    
                    # Create checkerboard
                    H, W = raw_a.shape[:2]
                    if reg_img.shape[:2] != (H, W):
                        reg_img = cv2.resize(reg_img, (W, H))
                    cb = np.zeros((H, W), dtype=np.float64)
                    tile_size = 36
                    for y in range(0, H, tile_size):
                        for x in range(0, W, tile_size):
                            y_end = min(H, y + tile_size)
                            x_end = min(W, x + tile_size)
                            if ((y // tile_size) + (x // tile_size)) % 2 == 0:
                                cb[y:y_end, x:x_end] = raw_a[y:y_end, x:x_end]
                            else:
                                cb[y:y_end, x:x_end] = reg_img[y:y_end, x:x_end]
                    cb_u8 = (np.clip(cb, 0, 1) * 255.0).astype(np.uint8)
                    cb_target = out_dir / "preview_checkerboard.png"
                    cv2.imwrite(str(cb_target), cb_u8)
                    artifacts["checkerboard_png"] = _to_relative_path(cb_target, root)
            except Exception as e:
                logger.debug(f"Could not generate checkerboard for {job_dir.name}: {e}")

    # 4. Tiepoints Correspondence Map
    tp_png = out_dir / "preview_tiepoints.png"
    if not tp_png.exists():
        tp_png = out_dir / "tie_points_matches.png"
    if tp_png.exists():
        artifacts["tiepoints_png"] = _to_relative_path(tp_png, root)
    else:
        # Check if we can generate it
        paths_file = job_dir / "input" / "input_paths.json"
        matches_file = inter_dir / "matches_verified.npy"
        if not matches_file.exists():
            matches_file = inter_dir / "matches_raw.npy"
        if paths_file.exists() and matches_file.exists() and _HAS_CV2:
            try:
                with open(paths_file, "r") as pf:
                    ip = json.load(pf)
                src_a = ip.get("img_a_path")
                src_b = ip.get("img_b_path")
                if src_a and src_b and os.path.exists(src_a) and os.path.exists(src_b):
                    raw_a, _ = read_raster(src_a)
                    raw_b, _ = read_raster(src_b)
                    matches = np.load(str(matches_file))
                    
                    H_a, W_a = raw_a.shape[:2]
                    H_b, W_b = raw_b.shape[:2]
                    canvas_h = max(H_a, H_b)
                    canvas_w = W_a + W_b
                    
                    def to_u8(img):
                        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
                        return (norm * 255.0).astype(np.uint8)
                    
                    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                    canvas[:H_a, :W_a] = cv2.cvtColor(to_u8(raw_a), cv2.COLOR_GRAY2BGR)
                    canvas[:H_b, W_a:canvas_w] = cv2.cvtColor(to_u8(raw_b), cv2.COLOR_GRAY2BGR)
                    
                    draw_matches = matches[::max(1, len(matches) // 60)][:60]
                    for m in draw_matches:
                        pt1 = (int(round(m[0])), int(round(m[1])))
                        pt2 = (int(round(m[2])) + W_a, int(round(m[3])))
                        conf = float(m[4]) if len(m) > 4 else 0.8
                        color = (0, 240, 100) if conf > 0.6 else (255, 200, 0)
                        cv2.circle(canvas, pt1, 3, color, -1)
                        cv2.circle(canvas, pt2, 3, color, -1)
                        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
                    
                    tp_target = out_dir / "preview_tiepoints.png"
                    cv2.imwrite(str(tp_target), canvas)
                    artifacts["tiepoints_png"] = _to_relative_path(tp_target, root)
            except Exception as e:
                logger.debug(f"Could not generate tiepoints for {job_dir.name}: {e}")

    return artifacts


# ---------------------------------------------------------------------------
# Main Unified Summary Builder
# ---------------------------------------------------------------------------

def build_chatbot_summary(
    job_id: str,
    base_jobs_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Build a unified chatbot-ready summary for a registration job.

    Parameters
    ----------
    job_id        : unique job identifier
    base_jobs_dir : optional override for jobs directory root (defaults to data/jobs)

    Returns
    -------
    summary : dict conforming to Chatbot Output Schema v1.0

    Raises
    ------
    FileNotFoundError : if job directory does not exist
    """
    root = Path.cwd()
    if base_jobs_dir is not None:
        jobs_root = Path(base_jobs_dir)
    else:
        jobs_root = root / "data" / "jobs"

    job_dir = jobs_root / job_id
    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    # 1. Load status.json
    status_file = job_dir / "status.json"
    status_info = {}
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                status_info = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {status_file}: {e}")

    status = status_info.get("status", "UNKNOWN")

    # 2. Load metrics.json
    metrics_file = job_dir / "output" / "metrics.json"
    metrics_raw: Dict[str, Any] = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                metrics_raw = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {metrics_file}: {e}")

    # Standardize metrics fields
    metrics = {
        "status": metrics_raw.get("status", status),
        "rmse_px": metrics_raw.get("rmse_px", None),
        "inlier_ratio": metrics_raw.get("inlier_ratio", None),
        "sdi": metrics_raw.get("sdi", None),
        "transform_type": metrics_raw.get("transform", None),
        "n_inliers": metrics_raw.get("n_inliers", 0),
        "n_total": metrics_raw.get("n_total", 0),
        "elapsed_s": metrics_raw.get("elapsed_s", status_info.get("elapsed_s", 0.0)),
        "is_synthetic_fallback": metrics_raw.get("is_synthetic_fallback", False),
    }
    if metrics_raw.get("error"):
        metrics["error"] = metrics_raw.get("error")
    elif status_info.get("error"):
        metrics["error"] = status_info.get("error")

    # 3. Load input metadata
    meta_file = job_dir / "input" / "metadata.json"
    paths_file = job_dir / "input" / "input_paths.json"
    
    img_a_path = None
    img_b_path = None
    gsd_a = 1.0
    gsd_b = 1.0
    shape_a = None
    shape_b = None
    solar_applied = False

    if paths_file.exists():
        try:
            with open(paths_file, "r") as pf:
                ip = json.load(pf)
                img_a_path = _to_relative_path(ip.get("img_a_path", ""), root)
                img_b_path = _to_relative_path(ip.get("img_b_path", ""), root)
        except Exception:
            pass

    if meta_file.exists():
        try:
            with open(meta_file, "r") as mf:
                mdata = json.load(mf)
            if not img_a_path and mdata.get("img_a_path"):
                img_a_path = _to_relative_path(mdata["img_a_path"], root)
            if not img_b_path and mdata.get("img_b_path"):
                img_b_path = _to_relative_path(mdata["img_b_path"], root)

            ma = mdata.get("image_a", {})
            mb = mdata.get("image_b", {})

            try:
                gsd_a = float(ma.get("gsd", 1.0))
            except Exception:
                gsd_a = 1.0
            try:
                gsd_b = float(mb.get("gsd", 1.0))
            except Exception:
                gsd_b = 1.0

            shape_a = _parse_shape(ma.get("shape"))
            shape_b = _parse_shape(mb.get("shape"))

            # Check solar angles
            angles_a = [ma.get("incidence_angle"), ma.get("emission_angle"), ma.get("phase_angle")]
            angles_b = [mb.get("incidence_angle"), mb.get("emission_angle"), mb.get("phase_angle")]
            has_angles_a = all(a is not None and str(a).lower() != "none" for a in angles_a)
            has_angles_b = all(b is not None and str(b).lower() != "none" for b in angles_b)
            solar_applied = has_angles_a and has_angles_b
        except Exception as e:
            logger.warning(f"Failed to parse metadata from {meta_file}: {e}")

    # Fallback to reading shapes from disk if not found in metadata
    if shape_a is None and img_a_path:
        p_a = root / img_a_path if not Path(img_a_path).is_absolute() else Path(img_a_path)
        if p_a.exists():
            try:
                if _HAS_RASTERIO:
                    with rasterio.open(str(p_a)) as s:
                        shape_a = (s.height, s.width)
                elif _HAS_CV2:
                    im_tmp = cv2.imread(str(p_a))
                    if im_tmp is not None:
                        shape_a = im_tmp.shape[:2]
            except Exception:
                pass

    if shape_b is None and img_b_path:
        p_b = root / img_b_path if not Path(img_b_path).is_absolute() else Path(img_b_path)
        if p_b.exists():
            try:
                if _HAS_RASTERIO:
                    with rasterio.open(str(p_b)) as s:
                        shape_b = (s.height, s.width)
                elif _HAS_CV2:
                    im_tmp = cv2.imread(str(p_b))
                    if im_tmp is not None:
                        shape_b = im_tmp.shape[:2]
            except Exception:
                pass

    dim_a = max(shape_a) if shape_a else 1
    dim_b = max(shape_b) if shape_b else 1
    pixel_dim_ratio = float(max(dim_a, dim_b) / max(min(dim_a, dim_b), 1))

    scale_disparity = float(max(gsd_a, gsd_b) / max(min(gsd_a, gsd_b), 1e-6))

    input_metadata = {
        "img_a_path": img_a_path,
        "img_b_path": img_b_path,
        "gsd_a_m_per_px": round(gsd_a, 4),
        "gsd_b_m_per_px": round(gsd_b, 4),
        "scale_disparity_ratio": round(scale_disparity, 2),
        "pixel_dimension_ratio": round(pixel_dim_ratio, 2),
        "solar_correction_applied": solar_applied,
    }

    # 4. Classify Quality & Confidence
    confidence_label, grade, reasoning, warnings = classify_registration_quality(
        metrics=metrics,
        input_metadata=input_metadata,
    )

    quality_assessment = {
        "confidence_label": confidence_label,
        "grade": grade,
        "reasoning": reasoning,
        "warnings": warnings,
    }

    # 5. Visual Artifacts
    artifacts = _ensure_visual_artifacts(job_dir, root)

    # Clean numeric metrics for client (remove internal keys)
    client_metrics = {
        "rmse_px": metrics["rmse_px"],
        "inlier_ratio": metrics["inlier_ratio"],
        "sdi": metrics["sdi"],
        "transform_type": metrics["transform_type"],
        "n_inliers": metrics["n_inliers"],
        "n_total": metrics["n_total"],
        "elapsed_s": metrics["elapsed_s"],
    }

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "status": status,
        "quality_assessment": quality_assessment,
        "metrics": client_metrics,
        "input_metadata": input_metadata,
        "artifacts": artifacts,
    }

    return summary_payload
