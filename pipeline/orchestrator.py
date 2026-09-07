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


from core.ingest_preprocess import read_raster, lommel_seeliger_normalize, align_gsd
from core.phase_congruency_mim import extract_structural_features
from core.dense_matcher import run_dense_matching
from core.anms_spatial_filter import anms_select
from core.geometric_verification import (
    magsac_filter,
    fit_thin_plate_spline,
    check_relief_significance,
    compute_homography_residuals,
)
from core.subpixel_refiner import refine_all_matches
from core.warp_and_eval import (
    warp_image,
    compute_rmse,
    compute_inlier_ratio,
    compute_sdi,
    export_geotiff,
)


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

    def __init__(self, job_id: str, img_a_path: str, img_b_path: str, matching_method: str = "classical"):
        self.job_id          = job_id
        self.img_a_path      = img_a_path
        self.img_b_path      = img_b_path
        self.matching_method = matching_method
        self._state          = JobState.PENDING
        self._start_time     = time.time()

        # Paths (interface contract)
        base = Path("data") / "jobs" / job_id
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
            'registered'         : base / "output" / "registered.tif",
            'residual_map'       : base / "output" / "residual_map.png",
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

        # ----------------------------------------------------------------
        # STEP 1 — Ingestion & Preprocessing
        # ----------------------------------------------------------------
        try:
            self._write_status(JobState.PREPROCESSING)
            logger.info(f"[{self.job_id}] Step 1: Ingestion & Preprocessing")

            img_a, meta_a = read_raster(self.img_a_path)
            img_b, meta_b = read_raster(self.img_b_path)

            img_a = lommel_seeliger_normalize(img_a, meta_a)
            img_b = lommel_seeliger_normalize(img_b, meta_b)
            img_a, img_b = align_gsd(img_a, img_b, meta_a, meta_b)

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

        except Exception as e:
            err = f"Step 1 (Ingestion) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'preprocessing', 'error': str(e)}

        # ----------------------------------------------------------------
        # STEP 2 — Phase Congruency & MIM
        # ----------------------------------------------------------------
        try:
            logger.info(f"[{self.job_id}] Step 2: Phase Congruency & MIM")
            feats_a = extract_structural_features(img_a)
            feats_b = extract_structural_features(img_b)
            self._save_npy(self.paths['pc_map_a'], feats_a['pc_map'])
            self._save_npy(self.paths['mim_a'],    feats_a['mim'])
            self._save_npy(self.paths['pc_map_b'], feats_b['pc_map'])
            self._save_npy(self.paths['mim_b'],    feats_b['mim'])

        except Exception as e:
            err = f"Step 2 (Phase Congruency) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'phase_congruency', 'error': str(e)}

        # ----------------------------------------------------------------
        # STEP 3 — Dense Matching
        # ----------------------------------------------------------------
        try:
            self._write_status(JobState.MATCHING)
            logger.info(f"[{self.job_id}] Step 3: Dense Matching (method={self.matching_method})")

            matches_raw = run_dense_matching(img_a, img_b, method=self.matching_method)

            if matches_raw is None or len(matches_raw) < 8:
                n_matches = 0 if matches_raw is None else len(matches_raw)
                raise RuntimeError(f"insufficient real matches: got {n_matches}, need >= 8")
            if matches_raw.ndim != 2 or matches_raw.shape[1] != 5:
                raise RuntimeError(
                    f"Match array must be (N,5); got {matches_raw.shape}"
                )

            self._save_npy(self.paths['matches_raw'], matches_raw)
            logger.info(f"  {len(matches_raw)} raw candidates")

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

        except Exception as e:
            err = f"Step 4 (ANMS) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'anms', 'error': str(e)}

        # ----------------------------------------------------------------
        # STEP 5 — MAGSAC++ + TPS Geometric Verification
        # ----------------------------------------------------------------
        try:
            self._write_status(JobState.VERIFYING)
            logger.info(f"[{self.job_id}] Step 5: MAGSAC++ & Geometric Verification")

            pts_a = matches_anms[:, :2]
            pts_b = matches_anms[:, 2:4]

            inlier_mask, H_matrix = magsac_filter(pts_a, pts_b)

            if inlier_mask.sum() < 4:
                raise RuntimeError(
                    f"Too few inliers ({inlier_mask.sum()}) after MAGSAC++."
                )

            inlier_matches = matches_anms[inlier_mask]
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
                transform = fit_thin_plate_spline(
                    pts_a[inlier_mask], pts_b[inlier_mask]
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
        try:
            self._write_status(JobState.REFINING)
            logger.info(f"[{self.job_id}] Step 6: Lucas-Kanade Sub-Pixel Refinement")

            refined_matches = refine_all_matches(img_a, img_b, inlier_matches)
            logger.info(f"  Sub-pixel refinement complete on {len(refined_matches)} points")

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
        # STEP 7 — Warp & Evaluation
        # ----------------------------------------------------------------
        try:
            logger.info(f"[{self.job_id}] Step 7: Image Warping & Evaluation")

            registered = warp_image(img_b, transform, img_a.shape)

            rmse         = compute_rmse(refined_matches, registered, img_a, transform=transform)
            inlier_ratio = compute_inlier_ratio(inlier_mask)
            sdi          = compute_sdi(refined_matches, img_a.shape)

            export_geotiff(registered, self.paths['registered'], meta_b)


            metrics = {
                'status'               : 'DONE',
                'rmse_px'              : round(float(rmse),         4),
                'inlier_ratio'         : round(float(inlier_ratio), 4),
                'sdi'                  : round(float(sdi),          4),
                'n_inliers'            : int(inlier_mask.sum()),
                'n_total'              : int(len(matches_anms)),
                'elapsed_s'            : round(time.time() - self._start_time, 2),
                'transform'            : transform_type,
                'is_synthetic_fallback': False,
            }

            with open(self.paths['metrics'], 'w') as f:
                json.dump(metrics, f, indent=2)

            self._write_status(JobState.DONE)

            logger.info(
                f"[{self.job_id}] DONE | RMSE={rmse:.3f}px | "
                f"Inlier Ratio={inlier_ratio:.1%} | SDI={sdi:.3f}"
            )
            return {'status': 'DONE', **metrics}

        except Exception as e:
            err = f"Step 7 (Warp & Eval) failed: {traceback.format_exc()}"
            logger.error(err)
            self._write_status(JobState.FAILED, err)
            return {'status': 'FAILED', 'stage': 'warp_and_eval', 'error': str(e)}
