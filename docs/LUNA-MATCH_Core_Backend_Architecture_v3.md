# LUNA-MATCH — Core Backend System Architecture (v3, Pure Engine)
### Deterministic planetary image registration & analysis backend. No AI/LLM content in this document — see companion document for the agent layer.

---

## 0. System Contract

```
INPUT:  Two lunar rasters (GeoTIFF / PDS / plain image) — a "source" (moving) and a "reference" (fixed)
OUTPUT: Registered raster + full metrics/diagnostics JSON + on-demand visual/graph artifacts + exportable control points
```

Nine-stage pipeline (upgraded from the original 5-stage MVP; stages 0 and 4b are new hardening additions):

| # | Stage | File |
|---|---|---|
| 0 | Footprint Overlap Pre-Check (fast-fail guard) | `core/overlap_check.py` |
| 1 | Ingestion & Photometric Normalization | `core/ingest_preprocess.py` |
| 2 | Illumination-Invariant Structural Representation (Phase Congruency + MIM) | `core/phase_congruency_mim.py` |
| 2b | Deterministic Crater Detection | `core/crater_detection.py` |
| 3 | Dense Matching | `core/dense_matcher.py` |
| 4 | Spatial Uniformity Filtering (ANMS) | `core/anms_spatial_filter.py` |
| 5 | Robust Geometric Verification (+ condition-number gating) | `core/geometric_verification.py` |
| 6 | Sub-Pixel Refinement | `core/subpixel_refiner.py` |
| 6b | Held-Out Validation Split | `core/validation_split.py` |
| 7 | Warping, Full Metrics (RMSE/MAE/SSIM/NCC/SDI), Export | `core/warp_and_eval.py` |

Literature basis (unchanged from v1, still authoritative):
Kovesi (1999) — Phase Congruency · Li, Hu, Ai (2018) — RIFT · Nefian et al. (NASA Ames, 2014) — Lommel-Seeliger · Brown & Szeliski (2005) — ANMS · Barath et al. (2020) — MAGSAC++ · Lucas & Kanade (1981) · Bookstein (1989) — Thin Plate Spline.

---

## 1. Repository Structure

```
luna-match/
  api/
    main.py                     # FastAPI app, all routes
    schemas.py                  # Pydantic models, single source of truth for wire format
    errors.py                   # Structured error taxonomy (Section 4.4)
  core/
    overlap_check.py            # Stage 0
    ingest_preprocess.py        # Stage 1
    phase_congruency_mim.py     # Stage 2
    crater_detection.py         # Stage 2b
    dense_matcher.py            # Stage 3
    anms_spatial_filter.py      # Stage 4
    geometric_verification.py   # Stage 5
    subpixel_refiner.py         # Stage 6
    validation_split.py         # Stage 6b
    warp_and_eval.py            # Stage 7
    summary_builder.py          # Assembles final /summary payload (pure aggregation, zero computation)
    graph_renderer.py           # Deterministic chart rendering (matplotlib, no LLM)
  pipeline/
    orchestrator.py             # State machine, stage sequencing, telemetry
  data/
    jobs/{job_id}/...           # Section 5
  tests/
    test_overlap_check.py
    test_ingest.py
    test_phase_congruency.py
    test_crater_detection.py
    test_matcher.py
    test_anms.py
    test_geometric.py
    test_subpixel.py
    test_validation_split.py
    test_warp_eval.py
    test_summary_builder.py
    test_graph_renderer.py
    test_pipeline_e2e.py
    test_api.py
  Dockerfile
  render.yaml
  requirements.txt
```

---

## 2. Stage Specifications

### Stage 0 — Footprint Overlap Pre-Check
**Runs before any expensive computation.** Rejects non-overlapping or implausible pairs in milliseconds.

```python
# core/overlap_check.py
def estimate_overlap(img_a: np.ndarray, img_b: np.ndarray,
                      meta_a: RasterMetadata, meta_b: RasterMetadata) -> float:
    """
    Returns estimated overlap fraction [0.0-1.0].
    Method: if geospatial bounds exist (CRS+transform on both), compute
    bounding-box intersection over union directly — O(1), exact.
    Else (no georeferencing): downsample both images to a common small size
    (e.g. 64x64) and compute normalized cross-correlation peak location +
    magnitude as a coarse overlap proxy.
    """

def overlap_gate(overlap_fraction: float, min_required: float = 0.15) -> None:
    """Raises OverlapTooLowError if overlap_fraction < min_required."""
```
Contract: orchestrator calls this immediately after Stage 1 ingestion, before Stage 2. On failure, job transitions directly to `FAILED` with `error_code = "OVERLAP_TOO_LOW"` — stages 2 through 7 never execute.

### Stage 1 — Ingestion & Photometric Normalization
```python
# core/ingest_preprocess.py
class RasterMetadata(BaseModel):
    gsd_m_per_px: float
    incidence_angle_deg: float | None
    emission_angle_deg: float | None
    phase_angle_deg: float | None
    crs: str | None
    transform: object | None
    width: int
    height: int
    dtype: str

def read_raster(path: str) -> tuple[np.ndarray, RasterMetadata]: ...
def read_raster_windowed(path: str, window: tuple, overview_level: int | None) -> tuple[np.ndarray, RasterMetadata]: ...
def read_raster_overview(path: str, max_dim: int = 2048) -> tuple[np.ndarray, RasterMetadata]: ...
def lommel_seeliger_normalize(image: np.ndarray, meta: RasterMetadata) -> np.ndarray:
    """R(i,e) = cos(i) / (cos(i) + cos(e)). Skipped (not fabricated) when solar angles absent — flagged in output."""
def build_octave_pyramid(image: np.ndarray, source_gsd: float, target_gsd: float) -> list[np.ndarray]: ...
def align_gsd(source_img, source_meta, ref_meta) -> np.ndarray: ...
```
Selection rule (memory safety): if `width * height * 8 bytes` exceeds a configured ceiling (default 500MB), Stage 1 automatically uses `read_raster_overview()` for coarse-path stages and `read_raster_windowed()` for final full-resolution refinement crops — never a full in-memory load of an oversized raster.

### Stage 2 — Phase Congruency & MIM
```python
# core/phase_congruency_mim.py
def log_gabor_filter_bank(shape: tuple, n_scales: int = 3, n_orientations: int = 6) -> list:
    """Cached per (shape, n_scales, n_orientations) key — process-scoped, persists across requests."""
def compute_even_odd_responses(image, filter_bank) -> tuple: ...
def phase_congruency(image, n_scales=3, n_orientations=6) -> np.ndarray: ...
def moment_analysis(pc_per_orientation) -> tuple[np.ndarray, np.ndarray]:
    """Returns (M_edges, m_corners)."""
def compute_mim(amplitude_per_orientation) -> np.ndarray: ...
def mim_descriptor(mim, keypoints, grid_size=6) -> np.ndarray:
    """Boundary keypoints use reflection padding — never emit zero-vector descriptors."""
def extract_structural_features(image) -> dict:
    """Single entry point: returns {pc_map, mim, edge_moment_map, corner_moment_map}. Computed exactly once per image per job."""
```
FFT execution must use `scipy.fft.fft2/ifft2(..., workers=-1)`, never `numpy.fft`. This is a hard requirement, not a suggestion — it is the difference between 1.8s and 4.2s on a standard image.

### Stage 2b — Crater Detection (non-blocking)
```python
# core/crater_detection.py
def detect_craters(edge_moment_map: np.ndarray, gsd_m_per_px: float,
                    min_diameter_px: int = 8, max_diameter_px: int = 500,
                    min_aspect_ratio: float = 0.60, confidence_thresh: float = 0.35) -> list[dict]:
    """
    1. PAPR guard: papr = max(M)/mean(M); if papr < 4.0 -> return [] (noise/flat rejection).
    2. Percentile-normalize M to 99th percentile.
    3. cv2.HoughCircles for circular rim candidates.
    4. Contour + cv2.fitEllipse for oblique/degraded rims, filtered by aspect_ratio.
    5. Non-maximum suppression across both candidate sets.
    6. Physical scaling: diameter_m = 2 * radius_px * gsd_m_per_px.
    Returns list of {center_px, radius_px, diameter_m, aspect_ratio, confidence}.
    """
def render_crater_overlay(image: np.ndarray, craters: list[dict]) -> np.ndarray: ...
def crater_density_per_km2(craters: list[dict], area_km2: float) -> float: ...
def bucket_diameter_histogram(craters: list[dict]) -> dict:
    """Buckets: '<1km','1-3km','3-10km','>10km'."""
```
Contract: wrapped in try/except at the orchestrator call site; a crater-detection failure **never** fails the parent registration job — it degrades to `craters: null` with a warning in the summary.

### Stage 3 — Dense Matching
```python
# core/dense_matcher.py
def select_matcher(gsd_ratio: float) -> str:
    """gsd_ratio > 3.0 or < 0.33 -> 'classical'; else -> 'structural' (if enabled) or 'classical' default."""
def run_dense_matching(img_a, img_b, gsd_ratio: float = 1.0, method: str = "classical",
                        feats_a: dict | None = None, feats_b: dict | None = None) -> np.ndarray:
    """
    Returns (N,5) float64 [x1,y1,x2,y2,confidence]. Contract is immutable —
    every downstream stage depends on this exact shape/dtype.
    When method='structural', MUST reuse feats_a/feats_b from Stage 2 —
    recomputing structural features here is a defect, not an option.
    """
def run_structural_matching(img_a, img_b, feats_a: dict, feats_b: dict) -> np.ndarray: ...
```

### Stage 4 — ANMS Spatial Filtering
```python
# core/anms_spatial_filter.py
def compute_suppression_radius(points: np.ndarray, strengths: np.ndarray, c_robust: float = 0.9) -> np.ndarray: ...
def anms_select(matches: np.ndarray, k: int | None = None) -> np.ndarray:
    """k defaults to min(2000, len(matches)//2) — adaptive, never a fixed constant that outstrips available matches."""
def quadtree_bucket_fallback(points: np.ndarray, grid: int = 8) -> np.ndarray: ...
```

### Stage 5 — Geometric Verification
```python
# core/geometric_verification.py
def magsac_filter(pts_a: np.ndarray, pts_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cv2.findHomography(..., method=cv2.USAC_MAGSAC). Returns (inlier_mask, H)."""
def condition_number(H: np.ndarray) -> float:
    """kappa = sigma_max/sigma_min via np.linalg.svd(H). """
def select_geometric_model(H: np.ndarray, inlier_pts_a, inlier_pts_b,
                            kappa_threshold: float = 1e4) -> dict:
    """
    If condition_number(H) > kappa_threshold -> refit as affine
    (cv2.estimateAffinePartial2D); if still degenerate -> similarity
    (scale+rotation+translation only, 4 DOF). Returns
    {transform, transform_type, condition_number, ill_conditioned: bool}.
    """
def fit_thin_plate_spline(src_pts, dst_pts) -> object: ...
def check_relief_significance(inlier_pts, homography_residuals, threshold_px: float = 1.5,
                               fraction_threshold: float = 0.15) -> bool: ...
def compute_homography_residuals(H, pts_a, pts_b) -> np.ndarray: ...
```
Decision order (must be implemented exactly in this sequence): MAGSAC++ homography → condition-number check → (if ill-conditioned) affine/similarity fallback → (if well-conditioned) relief-significance check → (if significant) TPS refit.

### Stage 6 — Sub-Pixel Refinement
```python
# core/subpixel_refiner.py
def lucas_kanade_refine(struct_map_f: np.ndarray, struct_map_g: np.ndarray, point: tuple,
                         patch_size: int = 15, iterations: int = 8,
                         convergence_threshold: float = 0.03, max_shift: float = 1.0,
                         min_eigenvalue: float = 1e-6) -> dict:
    """
    Operates on Stage-2 structural maps, not raw pixel intensity — this is
    what makes refinement illumination-invariant.
    Returns {x, y, converged: bool, exit_reason: 'converged_threshold'|
    'iteration_cap'|'exceeded_max_shift'|'low_eigenvalue'|'boundary'}.
    A point failing to converge is NOT an error — it means the original
    coordinate was already accurate; original coordinate is retained.
    """
def refine_all_matches(struct_f, struct_g, matches: np.ndarray, **kwargs) -> np.ndarray:
    """Returns refined (N,5) array + a parallel diagnostics dict with exit-reason counts."""
```

### Stage 6b — Held-Out Validation Split
```python
# core/validation_split.py
def held_out_rmse(matches: np.ndarray, transform_fitter: Callable, split_ratio: float = 0.8,
                   seed: int = 42) -> dict:
    """
    Shuffle verified inliers deterministically (fixed seed for reproducibility),
    fit transform_fitter on the first split_ratio fraction, compute RMSE
    against the remaining (1-split_ratio) fraction never used for fitting.
    Returns {held_out_rmse_px, fit_rmse_px, overfit_ratio: held_out/fit}.
    overfit_ratio significantly > 1.5 is a documented red flag surfaced in
    quality_assessment.warnings.
    """
```

### Stage 7 — Warping, Metrics, Export
```python
# core/warp_and_eval.py
def warp_image(source_img, transform, ref_shape: tuple) -> np.ndarray:
    """Duck-types on hasattr(transform,'apply') for TPS vs 3x3 ndarray homography."""
def compute_rmse(reprojected_pts, reference_pts) -> float: ...
def compute_mae(reprojected_pts, reference_pts) -> float: ...
def compute_inlier_ratio(n_inliers: int, n_candidates: int) -> float: ...
def compute_sdi(match_points, image_shape, grid: int = 8) -> float: ...
def compute_ssim(registered: np.ndarray, reference: np.ndarray, overlap_mask: np.ndarray | None) -> float:
    """skimage.metrics.structural_similarity over the valid overlap region only."""
def compute_ncc(registered: np.ndarray, reference: np.ndarray, overlap_mask: np.ndarray | None) -> float: ...
def generate_residual_error_map(reprojected_pts, reference_pts, image_shape) -> np.ndarray: ...
def export_geotiff(warped_img, ref_meta: RasterMetadata, out_path: str) -> None:
    """CRITICAL: export uses the REFERENCE image's (meta_b's) grid, since the
    source is warped INTO the reference frame. Verify this direction explicitly
    with a unit test — this exact mistake (using the wrong image's georeference)
    has occurred before in this codebase and must have a regression test."""
def export_control_points_csv(matches: np.ndarray, out_path: str) -> None:
    """Columns: x1,y1,x2,y2,confidence. One row per verified, refined match."""
def decompose_transform_readable(H: np.ndarray) -> dict:
    """Returns {rotation_deg, scale, tx, ty} decomposed from a similarity/affine
    approximation of H, for human-readable display — never shown as a raw 3x3
    matrix to an end consumer without this decomposition alongside it."""
```

---

## 3. Orchestrator — State Machine

```python
# pipeline/orchestrator.py
class JobState(str, Enum):
    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    STRUCTURAL_ANALYSIS = "STRUCTURAL_ANALYSIS"
    MATCHING = "MATCHING"
    VERIFYING = "VERIFYING"
    REFINING = "REFINING"
    DONE = "DONE"
    FAILED = "FAILED"

class LunaMatchPipeline:
    def __init__(self, job_id: str, img_a_path: str, img_b_path: str,
                 matching_method: str = "classical", mode: Literal["fast","standard"] = "standard"):
        ...
    def run(self) -> dict:
        """
        Sequence: overlap_gate -> ingest+normalize+align -> extract_structural_features
        (both images) -> detect_craters (non-blocking) -> run_dense_matching ->
        anms_select -> magsac_filter -> select_geometric_model -> [TPS if relief] ->
        refine_all_matches -> held_out_rmse -> warp_image -> full metrics -> export.
        Writes status.json after EVERY state transition (atomic write: write to
        .tmp then os.replace, never a partial JSON on disk).
        A stage exception sets state=FAILED with {failed_stage, error_code,
        error_message} and stops — never silently continues with garbage data.
        """
    def get_status(self) -> dict: ...
```
Concurrency: multiple jobs run independently via `ThreadPoolExecutor`; each job's `data/jobs/{job_id}/` directory is fully isolated — no shared mutable state between concurrent jobs except the process-scoped, read-only filter-bank cache (Stage 2), which is safe because it is keyed by immutable shape and never written to mid-computation.

---

## 4. API Layer — Full Specification

### 4.1 Endpoint Table

| Endpoint | Method | Purpose | Success | Failure |
|---|---|---|---|---|
| `/health` | GET | Liveness/readiness | 200 | — |
| `/register` | POST | Create + start a job | 202 | 422 (bad input) |
| `/jobs/{job_id}` | GET | Poll state | 200 | 404 |
| `/jobs/{job_id}/result` | GET | Raw numeric metrics (legacy-simple) | 200 | 404, 409 (not DONE) |
| `/jobs/{job_id}/summary` | GET | Full structured payload (Section 4.3) | 200 | 404, 409 |
| `/jobs/{job_id}/preview` | GET | Rendered raster/PNG artifact | 200 (image) | 404, 400 (bad `kind`) |
| `/jobs/{job_id}/graphs` | GET | Rendered chart PNG | 200 (image) | 404, 400 |
| `/jobs/{job_id}/export` | GET | Control-point CSV | 200 (text/csv) | 404, 409 |

### 4.2 Request/Response Schemas

```python
# api/schemas.py

class RegisterRequest(BaseModel):
    img_a_path: str
    img_b_path: str
    job_id: str | None = None          # server generates uuid4 if omitted
    matching_method: Literal["classical","structural"] = "classical"
    mode: Literal["fast","standard"] = "standard"

class RegisterResponse(BaseModel):
    job_id: str
    status: Literal["PENDING"]

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    stage: str | None
    error_code: str | None
    error_message: str | None
    elapsed_s: float

class QualityAssessment(BaseModel):
    confidence_label: str
    grade: Literal["A","B","C","D","F"]
    reasoning: str
    warnings: list[str]

class Metrics(BaseModel):
    rmse_px: float
    mae_px: float
    held_out_rmse_px: float
    inlier_ratio: float = Field(ge=0, le=1)
    sdi: float = Field(ge=0, le=1)
    ssim: float = Field(ge=0, le=1)
    ncc: float = Field(ge=-1, le=1)
    condition_number: float
    ill_conditioned: bool
    transform_type: Literal["homography","affine","similarity","thin_plate_spline"]
    transform_params_readable: dict
    n_inliers: int
    n_total: int
    elapsed_s: float
    stage_timings_ms: dict[str, float]

class InputMetadata(BaseModel):
    img_a_path: str
    img_b_path: str
    gsd_a_m_per_px: float
    gsd_b_m_per_px: float
    scale_disparity_ratio: float
    pixel_dimension_ratio: float
    footprint_overlap_pct: float
    area_covered_km2_a: float
    area_covered_km2_b: float
    solar_correction_applied: bool

class CraterSummary(BaseModel):
    detected_in_source: int
    detected_in_reference: int
    diameter_min_m: float | None
    diameter_max_m: float | None
    density_per_km2: float | None
    diameter_histogram: dict[str, int]
    craters_source: list[dict]

class Artifacts(BaseModel):
    registered_geotiff: str
    checkerboard_png: str
    tiepoints_png: str
    residual_map_png: str
    craters_png: str | None
    graph_residual_scatter: str
    graph_residual_histogram: str
    graph_crater_histogram: str | None
    control_points_csv: str

class SummaryResponse(BaseModel):
    schema_version: Literal["1.2"]
    job_id: str
    status: Literal["DONE"]
    quality_assessment: QualityAssessment
    metrics: Metrics
    input_metadata: InputMetadata
    craters: CraterSummary | None
    artifacts: Artifacts
```

### 4.3 `preview` / `graphs` query contract
```
GET /jobs/{job_id}/preview?kind={registered|checkerboard|tiepoints|residual|craters}
GET /jobs/{job_id}/graphs?kind={residual_scatter|residual_histogram|crater_histogram|confidence_gauge}
```
Unknown `kind` → `400 {"error_code":"INVALID_KIND","valid_values":[...]}`. Both endpoints cache the rendered PNG to `output/` on first request and serve from disk thereafter — never re-render on every call.

### 4.4 Error Taxonomy
```python
# api/errors.py
class ErrorCode(str, Enum):
    JOB_NOT_FOUND = "JOB_NOT_FOUND"                # 404
    JOB_NOT_DONE = "JOB_NOT_DONE"                   # 409
    OVERLAP_TOO_LOW = "OVERLAP_TOO_LOW"             # job FAILED, not HTTP error
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"    # job FAILED
    ILL_CONDITIONED_UNRECOVERABLE = "ILL_CONDITIONED_UNRECOVERABLE"  # job FAILED
    INVALID_INPUT_PATH = "INVALID_INPUT_PATH"       # 422
    INVALID_KIND = "INVALID_KIND"                   # 400
    INTERNAL_ERROR = "INTERNAL_ERROR"               # 500, logged with full traceback server-side
```
Every non-2xx response body: `{"error_code": str, "message": str, "job_id": str | null}`. No endpoint ever returns a bare stack trace to the client.

---

## 5. Storage Schema

```
data/jobs/{job_id}/
├── status.json
├── input/
│   ├── metadata.json
│   └── input_paths.json
├── intermediate/
│   ├── pc_map_a.npy, pc_map_b.npy
│   ├── mim_a.npy, mim_b.npy
│   ├── craters_a.json, craters_b.json
│   ├── matches_raw.npy, matches_anms.npy, matches_verified.npy, matches_refined.npy
│   └── transform_params.json         # includes condition_number, transform_type, held_out_rmse
└── output/
    ├── registered.tif
    ├── residual_map.png
    ├── preview_checkerboard.png, preview_tiepoints.png
    ├── craters_a.png, craters_b.png
    ├── graph_residual_scatter.png, graph_residual_histogram.png, graph_crater_histogram.png
    ├── control_points.csv
    └── metrics.json
```
All paths anchored to `PROJECT_ROOT` (resolved once at process start), never to `cwd()`.

---

## 6. Performance Tiers

| Tier | Trigger | Target | Mechanism |
|---|---|---|---|
| Fast | `mode=fast`, image ≤512px | <1s | Downscaled matching path, 2-scale filter bank |
| Standard | default, ~500-600px imagery | <2.5s | Full pipeline, multi-threaded FFT, process-scoped filter cache |
| Large | oversized raster detected at Stage 1 | <10s | `read_raster_overview()` coarse pass + `read_raster_windowed()` refinement crop only |

---

## 7. Testing Plan (one file per module, minimum)
Each `core/*.py` file has a matching `tests/test_*.py` covering: contract shape/dtype, boundary conditions (empty input, single point, degenerate geometry), and — for Stage 0, 2b, 5, 6b specifically — a dedicated test proving the new hardening actually fires (overlap gate rejects a synthetic non-overlapping pair; condition-number gate falls back on a synthetic near-degenerate point set; held-out RMSE differs meaningfully from fit RMSE on a synthetic overfit scenario). `tests/test_pipeline_e2e.py` runs the full sequence on the verified ground-truth pair and asserts against known tolerances, not just "does not crash."

---

## 8. Non-Goals (explicit boundary)
- No PDS3/PDS4/SPICE ingestion unless real mission data in that format is supplied.
- No GPU-trained deep learning matcher/detector anywhere in this repo.
- No conversational or session state of any kind — that is entirely the concern of the companion AI Agent architecture document. This backend's only obligation to any consumer is the versioned `/summary` JSON contract in Section 4.2.
