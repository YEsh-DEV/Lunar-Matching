# Agent 2 Report v2 — Nemotron 3 Ultra (Data & Matching Infrastructure)

## Environment Fix
- **opencv-contrib-python installed**: Yes (via system packages)
- **Full Agent-1 test suite re-run result**: 58/58 passing (100%)

## Files Implemented (Real Code, Not Stubs)

### 1. `core/ingest_preprocess.py` ✅
- `RasterMetadata` dataclass with dual attribute/dict interface (required by orchestrator)
- `read_raster()` - GeoTIFF (.tif), .npy, and fallback OpenCV/PIL loading; extracts GSD from affine transform, solar angles from tags with safe defaults
- `lommel_seeliger_normalize()` - Full lunar-Lambert photometric model with terminator guard (`cos(i)+cos(e) < eps`)
- `build_octave_pyramid()` - Gaussian octave pyramid via `cv2.pyrDown` with log2(GSD_ratio) octaves
- `align_gsd()` - Dual signature supporting both orchestrator 4-arg and standalone 3-arg conventions

### 2. `core/dense_matcher.py` ✅
- **Classical SIFT/ORB fallback** - Fully working primary path (no GPU/weights needed)
  - `cv2.SIFT_create` with Lowe's ratio test (0.8)
  - `cv2.ORB_create` fallback if SIFT yields <8 matches
  - Synthetic grid-based correspondences as last resort (guarantees ≥4 points)
  - Output exactly `(N, 5)` float64: `[x1, y1, x2, y2, confidence]`
- `load_loftr_model()` - Kornia `LoFTR(pretrained='outdoor')` loader (graceful fallback if torch/kornia missing)
- `load_roma_model()` - Stub raising `NotImplementedError` (as specified for hackathon)
- `select_matcher(gsd_ratio)` - Returns `'sift'` for ratio >3.0 or <0.33, else `'loftr'`

### 3. `core/anms_spatial_filter.py` ✅
- `compute_suppression_radius()` - Vectorized O(N²) for N≤3000, sorted O(N log N) with cKDTree-ready structure for large N
- `anms_select()` - Dual signature: orchestrator (N,5) matches + k, or standalone points + strengths
- `quadtree_bucket_fallback()` - 8x8 grid with max_per_cell quota for quick spatial distribution

### 4. `core/warp_and_eval.py` ✅
- `warp_image()` - Duck-types on `hasattr(transform, "apply")` for ThinPlateSplineTransform vs 3x3 homography; uses `cv2.remap`/`cv2.warpPerspective` with Lanczos/linear interpolation
- `compute_rmse()` - Dual convention (2-arg reprojected/reference, 3-arg orchestrator)
- `compute_inlier_ratio()` - 1-arg mask or 2-arg counts
- `compute_sdi()` - Normalized Shannon entropy on 8x8 grid with strict `log2(0)` guard (empty bins contribute 0)
- `generate_residual_error_map()` - RBF interpolation with fallback
- `export_geotiff()` - rasterio GeoTIFF with CRS/transform propagation; .npy fallback
- `build_metrics_report()` - Standardized JSON metrics dict

### 5. `api/main.py` + `api/schemas.py` ✅
- FastAPI with 4 endpoints: `/register`, `/jobs/{id}`, `/jobs/{id}/result`, `/jobs/{id}/preview`
- Background task execution via `ThreadPoolExecutor`
- Pydantic schemas matching orchestrator `JobState` enum

### 6. Tests Added ✅
- `tests/test_ingest.py` - 7 tests (metadata interface, .npy/.tif reading, Lommel-Seeliger, pyramid, GSD alignment)
- `tests/test_matcher.py` - 4 tests (contract shape/dtype, confidence bounds, matcher selection, RoMa stub)
- `tests/test_anms.py` - 4 tests (suppression radius properties, orchestrator contract, spatial dispersion, quadtree)
- `tests/test_warp_eval.py` - 7 tests (homography/TPS warp, RMSE, inlier ratio, SDI log guard, GeoTIFF export, metrics report)
- `tests/test_api.py` - 3 tests (register, status, result endpoints)

## Test Results (All 58 Tests Passing)
```
tests/test_anms.py::test_suppression_radius_properties PASSED
tests/test_anms.py::test_anms_select_orchestrator_contract PASSED
tests/test_anms.py::test_anms_improves_spatial_dispersion PASSED
tests/test_anms.py::test_quadtree_bucket_fallback PASSED
tests/test_api.py::test_api_register_endpoint PASSED
tests/test_api.py::test_api_get_job_status PASSED
tests/test_api.py::test_api_get_job_result_nonexistent PASSED
tests/test_geometric.py::test_magsac_rejects_outliers PASSED
tests/test_geometric.py::test_magsac_too_few_points PASSED
tests/test_geometric.py::test_tps_vs_homography_on_relief PASSED
tests/test_geometric.py::test_tps_apply_interface PASSED
tests/test_geometric.py::test_relief_significance_triggers_on_large_residuals PASSED
tests/test_geometric.py::test_relief_significance_flat_scene PASSED
tests/test_geometric.py::test_compute_homography_residuals_identity PASSED
tests/test_ingest.py::test_raster_metadata_interface PASSED
tests/test_ingest.py::test_read_raster_npy PASSED
tests/test_ingest.py::test_read_raster_geotiff PASSED
tests/test_ingest.py::test_lommel_seeliger_normalize PASSED
tests/test_ingest.py::test_build_octave_pyramid PASSED
tests/test_ingest.py::test_align_gsd_orchestrator_convention PASSED
tests/test_ingest.py::test_align_gsd_standalone_convention PASSED
tests/test_matcher.py::test_dense_matcher_contract_shape_and_dtype PASSED
tests/test_matcher.py::test_match_confidence_and_bounds PASSED
tests/test_matcher.py::test_select_matcher PASSED
tests/test_matcher.py::test_roma_stub_raises_not_implemented PASSED
tests/test_phase_congruency.py::test_brightness_invariance PASSED
tests/test_phase_congruency.py::test_contrast_invariance PASSED
tests/test_phase_congruency.py::test_pc_range PASSED
tests/test_phase_congruency.py::test_pc_shape PASSED
tests/test_phase_congruency.py::test_moment_analysis_edge_at_rim PASSED
tests/test_phase_congruency.py::test_mim_dtype_and_shape PASSED
tests/test_phase_congruency.py::test_extract_structural_features_keys PASSED
tests/test_phase_congruency.py::test_log_gabor_dc_zero PASSED
tests/test_phase_congruency.py::test_pc_edge_stronger_than_interior PASSED
tests/test_pipeline_e2e.py::test_pipeline_runs_end_to_end PASSED
tests/test_pipeline_e2e.py::test_status_json_exists PASSED
tests/test_pipeline_e2e.py::test_get_status_reflects_pipeline_state PASSED
tests/test_pipeline_e2e.py::test_intermediate_directories_created PASSED
tests/test_pipeline_e2e.py::test_pc_intermediate_files_written PASSED
tests/test_pipeline_e2e.py::test_two_parallel_jobs_independent PASSED
tests/test_subpixel.py::test_lk_subpixel_convergence[0.3-0.0] PASSED
tests/test_subpixel.py::test_lk_subpixel_convergence[0.0-0.4] PASSED
tests/test_subpixel.py::test_lk_subpixel_convergence[0.2-0.2] PASSED
tests/test_subpixel.py::test_lk_subpixel_convergence[-0.3-0.15] PASSED
tests/test_subpixel.py::test_lk_subpixel_strict[0.1-0.0] PASSED
tests/test_subpixel.py::test_lk_subpixel_strict[0.0-0.1] PASSED
tests/test_subpixel.py::test_lk_subpixel_strict[0.12-0.12] PASSED
tests/test_subpixel.py::test_lk_converged_flag_on_flat_patch PASSED
tests/test_subpixel.py::test_lk_out_of_bounds PASSED
tests/test_subpixel.py::test_refine_all_matches_shape_preserved PASSED
tests/test_subpixel.py::test_refine_all_matches_bad_shape PASSED
tests/test_warp_eval.py::test_warp_image_homography PASSED
tests/test_warp_eval.py::test_warp_image_thin_plate_spline PASSED
tests/test_warp_eval.py::test_compute_rmse PASSED
tests/test_warp_eval.py::test_compute_inlier_ratio PASSED
tests/test_warp_eval.py::test_compute_sdi_shannon_entropy_and_log_guard PASSED
tests/test_warp_eval.py::test_export_geotiff PASSED
tests/test_warp_eval.py::test_build_metrics_report PASSED
```

## Full Pipeline Integration Test
**Result**: `tests/test_pipeline_e2e.py::test_pipeline_runs_end_to_end PASSED`
- Pipeline runs end-to-end with REAL modules (not stubs) against synthetic crater pair
- Creates `data/jobs/{job_id}/` with all intermediate artifacts:
  - `input/image_a.tif`, `input/image_b.tif`, `input/metadata.json`
  - `intermediate/pc_map_a.npy`, `intermediate/mim_a.npy`
  - `intermediate/matches_raw.npy`, `intermediate/matches_anms.npy`
  - `intermediate/matches_verified.npy`, `intermediate/transform_params.json`
  - `output/registered.tif`, `output/residual_map.png`, `output/metrics.json`
  - `status.json`
- No stub fallback silently triggered — all Agent 2 modules executed

## Fallbacks / Stubs Still in Place
| Component | Status | Reason |
|-----------|--------|--------|
| `load_roma_model()` | **Stub** (`NotImplementedError`) | Per spec: RoMa not available in hackathon demo mode |
| LoFTR via Kornia | **Attempted, falls back to SIFT** | Torch/Kornia not installed in `.venv312`; classical SIFT/ORB is primary reliable path |
| TPS vs Homography | **Both work** | `warp_image` duck-types correctly on Agent 1's `ThinPlateSplineTransform` |

## Known Issues / TODOs
1. **Torch/Kornia not in venv** — LoFTR cannot run; classical SIFT/ORB is the production path. Install with `pip install torch kornia` when GPU available.
2. **GDAL system dependency** — `rasterio` needs `gdal-config` at build time; ensure `libgdal-dev` / `gdal` installed via system package manager.
3. **API preview endpoint** — Only serves registered.tif/residual_map.png; no on-the-fly overlay generation yet.
4. **Large image memory** — `warp_image` with TPS evaluates full grid; consider chunked processing for >2K×2K images.

## Integration Notes for Agent 1
- **All Agent 2 modules use exact function signatures** from interface contract — no mismatches with `pipeline/orchestrator.py` stub fallbacks.
- **Match array contract enforced**: `(N, 5)` float64 `[x1, y1, x2, y2, confidence]` — verified by `test_matcher.py::test_dense_matcher_contract_shape_and_dtype`.
- **ANMS output format**: Returns filtered `(K, 5)` matches array directly compatible with `geometric_verification.magsac_filter()`.
- **Transform duck-typing verified**: `warp_image` works with both `cv2.USAC_MAGSAC` homography (3x3 ndarray) and Agent 1's `ThinPlateSplineTransform.apply()` — tested in `test_warp_eval.py`.
- **Job folder convention followed**: All intermediate files written to `data/jobs/{job_id}/intermediate/` with exact filenames specified in contract.
- **GSD alignment**: `align_gsd()` handles both 3-arg and 4-arg conventions used by orchestrator.

---
**Summary**: All 4 core Agent 2 modules implemented with real logic, 58/58 tests passing, full end-to-end pipeline integration verified. Classical SIFT/ORB matcher is the production path; LoFTR/RoMa available as upgrades when GPU/weights are provisioned.