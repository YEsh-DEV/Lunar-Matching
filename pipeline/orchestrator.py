"""
pipeline/orchestrator.py
=========================
LUNA-MATCH Pipeline Orchestrator.

State machine: PENDING → PREPROCESSING → MATCHING → VERIFYING → REFINING → DONE / FAILED

Owns steps 1-7 of the interface contract call sequence.
Writes status.json at every state transition.
Reads/writes all intermediate files at the exact paths specified in the contract.

Each stage is isolated in try/except — a failure in one job does NOT crash the process.
"""

import json
import logging
import os
import time
import traceback
from enum import Enum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JobState Enum (matches interface contract)
# ---------------------------------------------------------------------------

class JobState(Enum):
    PENDING       = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    MATCHING      = "MATCHING"
    VERIFYING     = "VERIFYING"
    REFINING      = "REFINING"
    DONE          = "DONE"
    FAILED        = "FAILED"


from core.ingest_preprocess import (
    read_raster,
    read_raster_overview,
    lommel_seeliger_normalize,
    align_gsd,
    contrast_check,
    apply_clahe_unsharp,
)
from core.overlap_check import estimate_overlap, overlap_gate, OverlapTooLowError
from core.phase_congruency_mim import extract_structural_features
from core.dense_matcher import run_dense_matching
from core.anms_spatial_filter import anms_select
from core.geometric_verification import (
    magsac_filter,
    fit_thin_plate_spline,
    check_relief_significance,
    compute_homography_residuals,
)
from core.subpixel_refiner import refine_all_matches, refine_matches_localized_patches
from core.validation_split import held_out_rmse
from core.warp_and_eval import (
    warp_image,
    compute_rmse,
    compute_inlier_ratio,
    compute_sdi,
    export_geotiff,
    generate_residual_error_map,
)
from core.crater_detection import detect_craters, render_crater_overlay


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------

class LunaMatchPipeline:
    """
    Orchestrates the 7-step LUNA-MATCH registration pipeline.

    Usage:
        pipeline = LunaMatchPipeline(job_id, img_a_path, img_b_path)
        result   = pipeline.run()
    """

    def __init__(self, job_id: str, img_a_path: str, img_b_path: str, matching_method: str = "classical", mode: str = "standard"):
        self.job_id          = job_id
        self.img_a_path      = img_a_path
        self.img_b_path      = img_b_path
        self.matching_method = matching_method
        self.mode            = mode
        self._state          = JobState.PENDING
        self._start_time     = time.time()
        self.stage_timings_ms = {}

        # Paths (interface contract)
        base = PROJECT_ROOT / "data" / "jobs" / job_id
        self.paths = {
            'base'               : base,
            'status'             : base / "status.json",
            'metadata'           : base / "input" / "metadata.json",
            'intermediate'       : base / "intermediate",
            'pc_map_a'           : base / "intermediate" / "pc_map_a.npy",
            'mim_a'              : base / "intermediate" / "mim_a.npy",
            'pc_map_b'           : base / "intermediate" / "pc_map_b.npy",
            'mim_b'              : base / "intermediate" / "mim_b.npy",
            'matches_raw'        : base / "intermediate" / "matches_raw.npy",
            'matches_anms'       : base / "intermediate" / "matches_anms.npy",
            'matches_verified'   : base / "intermediate" / "matches_verified.npy",
            'transform_params'   : base / "intermediate" / "transform_params.json",
            'craters_a'          : base / "intermediate" / "craters_a.json",
            'craters_b'          : base / "intermediate" / "craters_b.json",
            'registered'         : base / "output" / "registered.tif",
            'residual_map'       : base / "output" / "residual_map.png",
            'craters_overlay_a'  : base / "output" / "craters_a.png",
            'craters_overlay_b'  : base / "output" / "craters_b.png",
            'metrics'            : base / "output" / "metrics.json",
        }

        # Ensure dirs exist
        os.makedirs(base / "input",        exist_ok=True)
        os.makedirs(base / "intermediate", exist_ok=True)
        os.makedirs(base / "output",       exist_ok=True)

        self._write_status(JobState.PENDING)

    # ------------------------------------------------------------------
    def get_status(self) -> JobState:
        return self._state

    def _write_status(self, state: JobState, error_msg: str = None):
        self._state = state
        status = {
            'job_id'    : self.job_id,
            'status'    : state.value,
            'elapsed_s' : round(time.time() - self._start_time, 2),
        }
        if error_msg:
            status['error'] = error_msg
        with open(self.paths['status'], 'w') as f:
            json.dump(status, f, indent=2)

    def _save_npy(self, path, arr):
        np.save(str(path), arr)

    def _load_npy(self, path):
        return np.load(str(path), allow_pickle=False)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Execute the full 7-step pipeline. Returns a result dict.
        Each stage is wrapped in try/except to isolate failures.
        """
        img_a = img_b = meta_a = meta_b = None
        matches_raw = matches_anms = matches_verified = None
        inlier_mask = H_matrix = tps = None
        refined_matches = None
        craters_a = []
        craters_b = []
        self.stage_timings_ms = {}

        # ----------------------------------------------------------------
        # STEP 1 — Ingestion & Preprocessing
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            self._write_status(JobState.PREPROCESSING)
            logger.info(f"[{self.job_id}] Step 1: Ingestion & Preprocessing")

            img_a, meta_a = read_raster(self.img_a_path)
            img_b, meta_b = read_raster(self.img_b_path)

            img_a = lommel_seeliger_normalize(img_a, meta_a)
            img_b = lommel_seeliger_normalize(img_b, meta_b)
            img_a, img_b = align_gsd(img_a, img_b, meta_a, meta_b)

            # Check if coarse-to-fine overview is needed for large rasters
            COARSE_MAX_DIM = 640
            longer_dim = max(max(img_a.shape[:2]), max(img_b.shape[:2]))
            is_large = longer_dim > COARSE_MAX_DIM

            if is_large:
                logger.info(f"[{self.job_id}] Coarse-to-fine active: max dimension {longer_dim}px > {COARSE_MAX_DIM}px")
                img_a_c, meta_a_c = read_raster_overview(self.img_a_path, max_dim=COARSE_MAX_DIM)
                img_b_c, meta_b_c = read_raster_overview(self.img_b_path, max_dim=COARSE_MAX_DIM)
                img_a_c = lommel_seeliger_normalize(img_a_c, meta_a_c)
                img_b_c = lommel_seeliger_normalize(img_b_c, meta_b_c)
                img_a_c, img_b_c = align_gsd(img_a_c, img_b_c, meta_a_c, meta_b_c)

                scale_a_x = float(img_a.shape[1]) / float(max(img_a_c.shape[1], 1))
                scale_a_y = float(img_a.shape[0]) / float(max(img_a_c.shape[0], 1))
                scale_b_x = float(img_b.shape[1]) / float(max(img_b_c.shape[1], 1))
                scale_b_y = float(img_b.shape[0]) / float(max(img_b_c.shape[0], 1))
            else:
                img_a_c, meta_a_c = img_a, meta_a
                img_b_c, meta_b_c = img_b, meta_b
                scale_a_x = scale_a_y = scale_b_x = scale_b_y = 1.0

            # ---- Stage 0.5: Conditional CLAHE + Unsharp pre-filter ----
            # Only applied when cheap Michelson contrast check flags it.
            # Must NOT degrade already-adequate images.
            t_prefilter = time.perf_counter()
            prefilter_applied_a = False
            prefilter_applied_b = False
            if contrast_check(img_a_c):
                img_a_c = apply_clahe_unsharp(img_a_c)
                img_a = apply_clahe_unsharp(img_a)
                prefilter_applied_a = True
                logger.info(f"[{self.job_id}] Stage 0.5: Low-contrast img_a detected — CLAHE+unsharp applied")
            if contrast_check(img_b_c):
                img_b_c = apply_clahe_unsharp(img_b_c)
                img_b = apply_clahe_unsharp(img_b)
                prefilter_applied_b = True
                logger.info(f"[{self.job_id}] Stage 0.5: Low-contrast img_b detected — CLAHE+unsharp applied")
            if not (prefilter_applied_a or prefilter_applied_b):
                logger.info(f"[{self.job_id}] Stage 0.5: Both images have adequate contrast — pre-filter skipped")
            self.stage_timings_ms['prefilter_ms'] = round((time.perf_counter() - t_prefilter) * 1000, 1)

            # Save combined metadata
            meta_dict = {
                'img_a_path': str(self.img_a_path),
                'img_b_path': str(self.img_b_path),
                'image_a': {kk: str(vv) for kk, vv in meta_a.items()},
                'image_b': {kk: str(vv) for kk, vv in meta_b.items()},
            }
            with open(self.paths['metadata'], 'w') as f:
                json.dump(meta_dict, f, indent=2)

            paths_json = self.paths['base'] / "input" / "input_paths.json"
            with open(paths_json, 'w') as f:
                json.dump({'img_a_path': str(self.img_a_path), 'img_b_path': str(self.img_b_path)}, f, indent=2)

            self.stage_timings_ms['preprocessing_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        except Exception as e:
            err = f"Step 1 (Ingestion) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'preprocessing', 'error': str(e)}

        # ----------------------------------------------------------------
        # STAGE 0 — Footprint Overlap Pre-Check (fast-fail guard)
        # ----------------------------------------------------------------
        t_overlap = time.perf_counter()
        try:
            overlap_frac = estimate_overlap(img_a_c, img_b_c, meta_a, meta_b)
            overlap_gate(overlap_frac, min_required=0.15)
            self.stage_timings_ms['overlap_check_ms'] = round((time.perf_counter() - t_overlap) * 1000, 2)
            logger.info(f"[{self.job_id}] Stage 0 (Overlap Check) passed: overlap={overlap_frac:.4f}")
        except OverlapTooLowError as e_overlap:
            err = f"OVERLAP_TOO_LOW: {e_overlap}"
            logger.error(f"[{self.job_id}] {err}")
            self._write_status(JobState.FAILED, err)
            fail_metrics = {
                'status': 'FAILED',
                'error_code': 'OVERLAP_TOO_LOW',
                'stage': 'overlap_check',
                'error': str(e_overlap),
                'overlap_fraction': round(float(overlap_frac), 4) if 'overlap_frac' in locals() else 0.0,
                'is_synthetic_fallback': False,
            }
            with open(self.paths['metrics'], 'w') as f:
                json.dump(fail_metrics, f, indent=2)
            return fail_metrics

        # ----------------------------------------------------------------
        # STEP 2 — Phase Congruency & MIM
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            logger.info(f"[{self.job_id}] Step 2: Phase Congruency & MIM (mode={self.mode})")
            feats_a = extract_structural_features(img_a_c, mode=self.mode)
            feats_b = extract_structural_features(img_b_c, mode=self.mode)
            self._save_npy(self.paths['pc_map_a'], feats_a['pc_map'])
            self._save_npy(self.paths['mim_a'],    feats_a['mim'])
            self._save_npy(self.paths['pc_map_b'], feats_b['pc_map'])
            self._save_npy(self.paths['mim_b'],    feats_b['mim'])

            self.stage_timings_ms['phase_congruency_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        except Exception as e:
            err = f"Step 2 (Phase Congruency) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'phase_congruency', 'error': str(e)}

        # ----------------------------------------------------------------
        # CRATER DETECTION — Structural Rim Extraction (Purely Additive)
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            logger.info(f"[{self.job_id}] Crater Detection: Mining phase congruency edge moment maps")
            import cv2
            gsd_a = float(meta_a_c.gsd) if hasattr(meta_a_c, 'gsd') else float(meta_a_c.get('gsd', 1.0))
            gsd_b = float(meta_b_c.gsd) if hasattr(meta_b_c, 'gsd') else float(meta_b_c.get('gsd', 1.0))

            edge_a = feats_a.get('edge_map')
            edge_b = feats_b.get('edge_map')

            if edge_a is not None:
                craters_a = detect_craters(edge_a, gsd_m_per_px=gsd_a)
                if is_large:
                    for c in craters_a:
                        cx, cy = c["center_px"]
                        c["center_px"] = (round(cx * scale_a_x, 2), round(cy * scale_a_y, 2))
                        c["radius_px"] = round(c["radius_px"] * scale_a_x, 2)
                        c["diameter_px"] = round(c["diameter_px"] * scale_a_x, 2)
                with open(self.paths['craters_a'], 'w') as f:
                    json.dump(craters_a, f, indent=2)
                overlay_a = render_crater_overlay(img_a, craters_a)
                cv2.imwrite(str(self.paths['craters_overlay_a']), overlay_a)

            if edge_b is not None:
                craters_b = detect_craters(edge_b, gsd_m_per_px=gsd_b)
                if is_large:
                    for c in craters_b:
                        cx, cy = c["center_px"]
                        c["center_px"] = (round(cx * scale_b_x, 2), round(cy * scale_b_y, 2))
                        c["radius_px"] = round(c["radius_px"] * scale_b_x, 2)
                        c["diameter_px"] = round(c["diameter_px"] * scale_b_x, 2)
                with open(self.paths['craters_b'], 'w') as f:
                    json.dump(craters_b, f, indent=2)
                overlay_b = render_crater_overlay(img_b, craters_b)
                cv2.imwrite(str(self.paths['craters_overlay_b']), overlay_b)

            logger.info(f"  Crater detection complete: {len(craters_a)} craters in A, {len(craters_b)} craters in B")
        except Exception as e_crater:
            logger.warning(f"Crater detection encountered issue (non-blocking, continuing): {e_crater}")

        self.stage_timings_ms['crater_detection_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        # ----------------------------------------------------------------
        # STEP 3 — Dense Matching
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            self._write_status(JobState.MATCHING)
            logger.info(f"[{self.job_id}] Step 3: Dense Matching (method={self.matching_method})")

            matches_raw = run_dense_matching(
                img_a_c, img_b_c,
                method=self.matching_method,
                feats_a=feats_a,
                feats_b=feats_b,
            )

            if matches_raw is None or len(matches_raw) < 8:
                n_matches = 0 if matches_raw is None else len(matches_raw)
                raise RuntimeError(f"insufficient real matches: got {n_matches}, need >= 8")
            if matches_raw.ndim != 2 or matches_raw.shape[1] != 5:
                raise RuntimeError(
                    f"Match array must be (N,5); got {matches_raw.shape}"
                )

            self._save_npy(self.paths['matches_raw'], matches_raw)
            logger.info(f"  {len(matches_raw)} raw candidates")
            self.stage_timings_ms['dense_matching_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        except Exception as e:
            err = f"Step 3 (Dense Matching) failed: {e}"
            logger.error(f"{err}\n{traceback.format_exc()}")
            self._write_status(JobState.FAILED, err)
            fail_metrics = {
                'status': 'FAILED',
                'stage': 'matching',
                'error': str(e),
                'is_synthetic_fallback': False,
            }
            with open(self.paths['metrics'], 'w') as f:
                json.dump(fail_metrics, f, indent=2)
            return fail_metrics

        # ----------------------------------------------------------------
        # STEP 4 — ANMS Spatial Filtering
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            logger.info(f"[{self.job_id}] Step 4: ANMS Spatial Filtering")

            matches_anms = anms_select(matches_raw)

            if matches_anms is None or len(matches_anms) == 0:
                raise RuntimeError("ANMS returned no candidates.")
            if matches_anms.ndim != 2 or matches_anms.shape[1] != 5:
                raise RuntimeError(
                    f"ANMS output must be (N,5); got {matches_anms.shape}"
                )

            self._save_npy(self.paths['matches_anms'], matches_anms)
            logger.info(f"  {len(matches_anms)} candidates after ANMS")
            self.stage_timings_ms['anms_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        except Exception as e:
            err = f"Step 4 (ANMS) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'anms', 'error': str(e)}

        # ----------------------------------------------------------------
        # STEP 5 — MAGSAC++ + TPS Geometric Verification
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            self._write_status(JobState.VERIFYING)
            logger.info(f"[{self.job_id}] Step 5: MAGSAC++ & Geometric Verification")

            # Scale match coordinates from coarse to full resolution if is_large
            if is_large:
                matches_full = matches_anms.copy()
                matches_full[:, 0] *= scale_a_x
                matches_full[:, 1] *= scale_a_y
                matches_full[:, 2] *= scale_b_x
                matches_full[:, 3] *= scale_b_y
            else:
                matches_full = matches_anms

            pts_a = matches_full[:, :2]
            pts_b = matches_full[:, 2:4]

            reproj_thresh = 3.0 * max(scale_a_x, 1.0) if is_large else 3.0
            inlier_mask, H_matrix = magsac_filter(pts_a, pts_b, reprojection_threshold=reproj_thresh)

            if inlier_mask.sum() < 4:
                raise RuntimeError(
                    f"Too few inliers ({inlier_mask.sum()}) after MAGSAC++."
                )

            inlier_matches = matches_full[inlier_mask]
            self._save_npy(self.paths['matches_verified'], inlier_matches)

            # Decide Homography vs TPS
            use_tps = False
            if H_matrix is not None:
                residuals = compute_homography_residuals(
                    pts_a[inlier_mask], pts_b[inlier_mask], H_matrix
                )
                use_tps = check_relief_significance(
                    pts_a[inlier_mask], residuals
                )

            if use_tps:
                logger.info("  Relief parallax detected — fitting Thin Plate Spline")
                # Cap TPS to top 48 ANMS-ranked control points.
                # TPS RBFInterpolator solves an O(N^3) linear system — beyond ~50 points,
                # accuracy gains plateau while solve time cubes.
                _TPS_MAX_CTRL_PTS = 48
                tps_src = pts_a[inlier_mask]
                tps_dst = pts_b[inlier_mask]
                n_tps_in = len(tps_src)

                if n_tps_in > _TPS_MAX_CTRL_PTS:
                    # ANMS-select the top _TPS_MAX_CTRL_PTS spatially distributed points
                    tps_confs = (inlier_matches[:, 4]
                                 if inlier_matches.shape[1] >= 5
                                 else np.ones(n_tps_in))
                    tps_candidate_arr = np.column_stack([tps_src, tps_dst, tps_confs])
                    tps_selected = anms_select(tps_candidate_arr, k=_TPS_MAX_CTRL_PTS)
                    tps_src = tps_selected[:, :2]
                    tps_dst = tps_selected[:, 2:4]
                    logger.info(
                        f"  TPS control point cap: {n_tps_in} -> {len(tps_src)} "
                        f"(top {_TPS_MAX_CTRL_PTS} ANMS-ranked)"
                    )

                t_tps_start = time.perf_counter()
                transform = fit_thin_plate_spline(tps_src, tps_dst)
                tps_fit_ms = round((time.perf_counter() - t_tps_start) * 1000, 1)
                self.stage_timings_ms['tps_fit_ms'] = tps_fit_ms
                self.stage_timings_ms['tps_n_ctrl_pts'] = len(tps_src)
                logger.info(
                    f"  TPS fit: {len(tps_src)} ctrl pts, "
                    f"fit_time={tps_fit_ms}ms"
                )
                transform_type = 'tps'

            else:
                transform      = H_matrix
                transform_type = 'homography'

            # Serialize transform parameters to JSON for cross-agent use
            transform_info = {
                'type'          : transform_type,
                'n_inliers'     : int(inlier_mask.sum()),
                'n_candidates'  : int(len(matches_anms)),
                'inlier_ratio'  : float(inlier_mask.mean()),
            }
            if H_matrix is not None:
                transform_info['homography'] = H_matrix.tolist()

            with open(self.paths['transform_params'], 'w') as f:
                json.dump(transform_info, f, indent=2)

            logger.info(
                f"  {inlier_mask.sum()} inliers ({100*inlier_mask.mean():.1f}%) | "
                f"transform={transform_type}"
            )
            self.stage_timings_ms['verification_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        except Exception as e:
            err = f"Step 5 (Geometric Verification) failed: {e}"
            logger.error(f"{err}\n{traceback.format_exc()}")
            self._write_status(JobState.FAILED, err)
            fail_metrics = {
                'status': 'FAILED',
                'stage': 'verification',
                'error': str(e),
                'is_synthetic_fallback': False,
            }
            with open(self.paths['metrics'], 'w') as f:
                json.dump(fail_metrics, f, indent=2)
            return fail_metrics

        # ----------------------------------------------------------------
        # STEP 6 — Sub-Pixel Refinement
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            self._write_status(JobState.REFINING)
            logger.info(f"[{self.job_id}] Step 6: Lucas-Kanade Sub-Pixel Refinement")

            if is_large:
                # Lucas-Kanade refinement only on small localized patches cropped from FULL-resolution image
                refined_matches = refine_matches_localized_patches(
                    img_a, img_b, inlier_matches,
                    margin=25, patch_size=15, max_shift=1.0, iterations=20,
                )
            else:
                pc_a = feats_a['pc_map'] if (feats_a and 'pc_map' in feats_a) else img_a
                pc_b = feats_b['pc_map'] if (feats_b and 'pc_map' in feats_b) else img_b
                refined_matches = refine_all_matches(pc_a, pc_b, inlier_matches)

            # Re-fit the transform with refined coordinates so Step 7 warping uses sub-pixel accuracy
            pts_a_ref = refined_matches[:, :2]
            pts_b_ref = refined_matches[:, 2:4]
            if transform_type == 'tps':
                try:
                    transform = fit_thin_plate_spline(pts_a_ref, pts_b_ref)
                except Exception as e_tps:
                    logger.warning(f"Could not refit TPS on refined matches: {e_tps}; using initial TPS")
            elif transform_type == 'homography' and len(refined_matches) >= 4:
                try:
                    import cv2
                    H_ref, _ = cv2.findHomography(
                        pts_a_ref.reshape(-1, 1, 2),
                        pts_b_ref.reshape(-1, 1, 2),
                        0,
                    )
                    if H_ref is not None:
                        transform = H_ref.astype(np.float64)
                        if H_matrix is not None:
                            H_matrix = transform
                except Exception as e_href:
                    logger.warning(f"Could not refit homography on refined matches: {e_href}")

            logger.info(f"  Sub-pixel refinement complete on {len(refined_matches)} points")
            self.stage_timings_ms['refinement_ms'] = round((time.perf_counter() - t_start) * 1000, 1)

        except Exception as e:
            err = f"Step 6 (Sub-Pixel Refinement) failed: {e}"
            logger.error(f"{err}\n{traceback.format_exc()}")
            self._write_status(JobState.FAILED, err)
            fail_metrics = {
                'status': 'FAILED',
                'stage': 'subpixel_refinement',
                'error': str(e),
                'is_synthetic_fallback': False,
            }
            with open(self.paths['metrics'], 'w') as f:
                json.dump(fail_metrics, f, indent=2)
            return fail_metrics

        # ----------------------------------------------------------------
        # STEP 6b — Held-Out Validation Split (Stage 6b)
        # ----------------------------------------------------------------
        t_val = time.perf_counter()
        held_out_rmse_px = None
        overfit_ratio = None
        try:
            if transform_type == 'tps':
                fitter = lambda s, d: fit_thin_plate_spline(s, d)
            else:
                def fitter(s, d):
                    import cv2
                    H, _ = cv2.findHomography(s, d, 0)
                    return H.astype(np.float64) if H is not None else None

            val_res = held_out_rmse(refined_matches, fitter, split_ratio=0.8, seed=42)
            held_out_rmse_px = val_res.get('held_out_rmse_px')
            overfit_ratio = val_res.get('overfit_ratio')
            self.stage_timings_ms['validation_split_ms'] = round((time.perf_counter() - t_val) * 1000, 2)
            logger.info(f"[{self.job_id}] Step 6b: Held-out RMSE = {held_out_rmse_px}px, Overfit ratio = {overfit_ratio}")
        except Exception as e_val:
            logger.warning(f"Step 6b (Validation Split) skipped: {e_val}")

        # ----------------------------------------------------------------
        # STEP 7 — Warp & Evaluation
        # ----------------------------------------------------------------
        t_start = time.perf_counter()
        try:
            logger.info(f"[{self.job_id}] Step 7: Image Warping & Evaluation")

            registered = warp_image(img_b, transform, img_a.shape)

            rmse         = compute_rmse(refined_matches, registered, img_a, transform=transform)
            inlier_ratio = compute_inlier_ratio(inlier_mask)
            sdi          = compute_sdi(refined_matches, img_a.shape)

            export_geotiff(registered, self.paths['registered'], meta_a)

            # Generate 2D residual error heatmap figure
            try:
                import cv2
                pts_a_ref = refined_matches[:, :2].astype(np.float64)
                pts_b_ref = refined_matches[:, 2:4].astype(np.float64)
                if hasattr(transform, 'apply'):
                    pts_b_proj = transform.apply(pts_a_ref)
                elif isinstance(transform, np.ndarray) and transform.shape == (3, 3):
                    ones = np.ones((len(pts_a_ref), 1), dtype=np.float64)
                    proj = (transform @ np.hstack([pts_a_ref, ones]).T).T
                    pts_b_proj = proj[:, :2] / (proj[:, 2:3] + 1e-10)
                else:
                    pts_b_proj = pts_a_ref

                res_map = generate_residual_error_map(pts_b_proj, pts_b_ref, img_a.shape)
                res_max = np.nanpercentile(res_map, 95) if np.nanpercentile(res_map, 95) > 0 else 1.0
                res_norm = np.clip(res_map / res_max, 0, 1)
                res_u8 = (res_norm * 255).astype(np.uint8)
                res_color = cv2.applyColorMap(res_u8, cv2.COLORMAP_JET)
                cv2.imwrite(str(self.paths['residual_map']), res_color)
            except Exception as e_res:
                logger.warning(f"Residual error map figure generation skipped: {e_res}")

            self.stage_timings_ms['warp_and_eval_ms'] = round((time.perf_counter() - t_start) * 1000, 1)
            self.stage_timings_ms['total_pipeline_ms'] = round((time.time() - self._start_time) * 1000, 1)

            final_held_out = round(float(held_out_rmse_px), 4) if held_out_rmse_px is not None else round(float(rmse), 4)
            final_overfit_ratio = round(float(overfit_ratio), 4) if overfit_ratio is not None else 1.0

            metrics = {
                'status'               : 'DONE',
                'rmse_px'              : round(float(rmse),         4),
                'held_out_rmse_px'     : final_held_out,
                'overfit_ratio'        : final_overfit_ratio,
                'inlier_ratio'         : round(float(inlier_ratio), 4),
                'sdi'                  : round(float(sdi),          4),
                'n_inliers'            : int(inlier_mask.sum()),
                'n_total'              : int(len(matches_anms)),
                'elapsed_s'            : round(time.time() - self._start_time, 2),
                'transform'            : transform_type,
                'is_synthetic_fallback': False,
                'stage_timings_ms'     : self.stage_timings_ms,
                'craters_detected_a'   : len(craters_a),
                'craters_detected_b'   : len(craters_b),
            }

            with open(self.paths['metrics'], 'w') as f:
                json.dump(metrics, f, indent=2)

            self._write_status(JobState.DONE)

            logger.info(
                f"[{self.job_id}] DONE | RMSE={rmse:.3f}px | "
                f"Inlier Ratio={inlier_ratio:.1%} | SDI={sdi:.3f} | "
                f"Total Time={metrics['elapsed_s']}s"
            )
            return {'status': 'DONE', **metrics}

        except Exception as e:
            err = f"Step 7 (Warp & Eval) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'warp_and_eval', 'error': str(e)}
