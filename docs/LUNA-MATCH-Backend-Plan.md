# LUNA-MATCH — End-to-End Backend Development Plan
### SIH26166 — Multi-modal, Sun-angle and Scale-invariant Lunar Image Correspondence
### Team Vadapav

---

## 0. System Overview

```
Input: Two lunar rasters (e.g. Chandrayaan-2 OHRC + LRO NAC reference, or TMC-2 + IIRS)
Output: Registered GeoTIFF + residual error vector map + metrics report (RMSE, Inlier Ratio, SDI)
```

Five-stage pipeline:

1. **Ingestion & Preprocessing** — read rasters, extract metadata, photometric normalization, octave resolution pyramid
2. **Illumination-Invariant Representation** — Phase Congruency + Maximum Index Map (MIM)
3. **Dense Matching + Spatial Uniformity** — LoFTR/RoMa + Adaptive Non-Maximal Suppression (ANMS)
4. **Robust Geometric Verification** — MAGSAC++ + Thin Plate Spline (TPS) non-rigid model
5. **Sub-pixel Refinement, Warping & Evaluation** — Lucas-Kanade + warp + metrics dashboard

Backing literature (all math sourced from these):
- Kovesi, *Image Features from Phase Congruency* (Videre, 1999)
- Li, Hu, Ai, *RIFT: Radiation-invariant Feature Transform* (IEEE TGRS, 2018)
- Nefian et al., *Photometric Lunar Surface Reconstruction* (NASA Ames, ICIP 2014)
- LROC NAC Processing Guide
- Brown & Szeliski, *Multi-Image Matching using Multi-Scale Oriented Patches* (CVPR 2005)
- Sun et al., *LoFTR* (CVPR 2021)
- Edstedt et al., *RoMa* (CVPR 2024)
- Barath et al., *MAGSAC++* (CVPR 2020)
- Lucas & Kanade, *An Iterative Image Registration Technique* (1981)

---

## 1. Repository Structure

```
luna-match/
  api/
    main.py                    # FastAPI app, routes
    schemas.py                 # Pydantic request/response models
  core/
    ingest_preprocess.py       # Stage 1
    phase_congruency_mim.py    # Stage 2
    dense_matcher.py           # Stage 3 (matching)
    anms_spatial_filter.py     # Stage 3 (uniformity)
    geometric_verification.py  # Stage 4
    subpixel_refiner.py        # Stage 5 (refinement)
    warp_and_eval.py           # Stage 5 (warp + metrics)
  pipeline/
    orchestrator.py            # runs stages 1-5 in order, tracks job state
  models/
    loftr_weights/              # pretrained weights (downloaded, not committed)
  data/
    jobs/{job_id}/input/
    jobs/{job_id}/output/
  tests/
    test_ingest.py
    test_phase_congruency.py
    test_matcher.py
    test_geometric.py
    test_subpixel.py
    test_pipeline_e2e.py
  requirements.txt
  README.md
```

---

## 2. Stage 1 — Ingestion & Preprocessing

**File:** `core/ingest_preprocess.py`

### 2.1 Tasks
- Read GeoTIFF / PDS4 / IMG rasters (`rasterio` / GDAL)
- Extract metadata: GSD (pixel scale), sun incidence angle `i`, emission angle `e`, phase angle `α`
- Handle pushbroom sensor geometry quirks (non-square ground pixels from elliptical orbits, "summed mode" 2x1 binning doubling GSD — per LROC NAC Guide)
- Photometric normalization using the Lunar-Lambertian (Lommel-Seeliger) reflectance model
- Build an octave resolution pyramid to bridge the GSD gap (OHRC ~0.25 m/px → LRO NAC ~0.5–1.5 m/px → TMC-2 ~5 m/px → IIRS ~80 m/px)

### 2.2 Math

**Lunar-Lambertian (Lommel-Seeliger) reflectance model** (Nefian et al., NASA Ames):

```
R(i, e, α) = (e^(-c1·α) + c2) · [ (1 - L(α))·cos(i) + 2·L(α)· cos(i) / (cos(i) + cos(e)) ]
```

- `i` = incidence angle (Sun → surface normal)
- `e` = emission angle (satellite → surface normal)
- `α` = phase angle (Sun → satellite)
- `L(α)` = phase-dependent blending parameter between Lommel-Seeliger and Lambertian terms
- `c1, c2` = empirically fitted constants

Normalize each pixel by dividing raw DN by `R(i, e, α)` computed from per-scene solar geometry metadata, equalizing dynamic range between steep sun-facing crater walls and shallow plains.

**Octave pyramid scaling:**

```
GSD_ratio = GSD_target / GSD_source
num_octaves = ceil(log2(GSD_ratio))
```

Downsample the higher-resolution image by successive factor-of-2 Gaussian-blurred decimations until its effective GSD is close to the reference image's GSD, before attempting initial coarse alignment. This avoids "receptive field starvation" in the downstream transformer matcher.

### 2.3 Code to write

```python
# core/ingest_preprocess.py

class RasterMetadata:
    gsd: float
    incidence_angle: float
    emission_angle: float
    phase_angle: float
    crs: str
    transform: object  # affine transform

def read_raster(path: str) -> tuple[np.ndarray, RasterMetadata]:
    """Read GeoTIFF/PDS4/IMG via rasterio, extract band + metadata."""

def lommel_seeliger_normalize(image: np.ndarray, meta: RasterMetadata,
                                c1: float, c2: float) -> np.ndarray:
    """Apply photometric normalization using R(i, e, alpha)."""

def build_octave_pyramid(image: np.ndarray, source_gsd: float,
                           target_gsd: float) -> list[np.ndarray]:
    """Gaussian-blur + downsample to bridge GSD gap. Returns pyramid levels."""

def align_gsd(source_img: np.ndarray, source_meta: RasterMetadata,
                ref_meta: RasterMetadata) -> np.ndarray:
    """Pick/interpolate the pyramid level closest to ref GSD."""
```

---

## 3. Stage 2 — Illumination-Invariant Structural Representation

**File:** `core/phase_congruency_mim.py`

### 3.1 Tasks
- Compute Phase Congruency (PC) maps using a bank of 2D Log-Gabor filters (illumination/contrast invariant edge & corner detection)
- Compute moment analysis to separate crater rim edges from corner/intersection points
- Compute RIFT's Maximum Index Map (MIM) on top of PC responses to make the representation robust across sensor modalities (panchromatic vs hyperspectral)

### 3.2 Math

**Log-Gabor filter in polar frequency coordinates** (scale `s`, orientation `o`):

```
L(ρ, θ, s, o) = exp( -(ln(ρ/ρs))² / (2σρ²) ) · exp( -(θ - θso)² / (2σθ²) )
```

**Even/odd filter responses:**

```
e_so(x,y) = I(x,y) * L_even(s,o)
o_so(x,y) = I(x,y) * L_odd(s,o)
```

**Amplitude and phase:**

```
A_so(x,y) = sqrt(e_so² + o_so²)
φ_so(x,y) = arctan(o_so / e_so)
```

**Sharpened phase deviation function** (Kovesi's PC2 measure — efficient form, no inverse trig):

```
ΔΦ_so(x) = cos(φ_so(x) - φ̄_o(x)) - |sin(φ_so(x) - φ̄_o(x))|

A_so · ΔΦ_so = (e_so·φ_e + o_so·φ_o) - |e_so·φ_o - o_so·φ_e|

where (φ_e, φ_o) = (F, H) / sqrt(F² + H²),  F = Σ_s e_so,  H = Σ_s o_so
```

**Phase congruency measure:**

```
PC(x) = Σ_s,o [ W_o(x) · ⌊A_so·ΔΦ_so - T⌋ ] / (Σ_s,o A_so(x) + ε)
```
(`W_o` = frequency-spread weighting, `T` = noise threshold, `⌊⌋` = zero-clamp)

**2D moment analysis** — separates edges (rims) from corners (crater intersections):

```
a = Σ_o (PC(θo)·cos θo)²
b = 2·Σ_o (PC(θo)·cos θo)(PC(θo)·sin θo)
c = Σ_o (PC(θo)·sin θo)²

M_ψ (max moment, edges)  = ½( c + a + sqrt(b² + (a-c)²) )
m_ψ (min moment, corners) = ½( c + a - sqrt(b² + (a-c)²) )
```

**RIFT Maximum Index Map (MIM)** — cross-modal invariant representation (Li et al.):

```
A_o(x,y) = Σ_{s=1}^{Ns} A_so(x,y)              # sum amplitude over scales, per orientation
ω_max(x,y) = argmax_{o ∈ {1..No}} A_o(x,y)     # dominant orientation index → MIM pixel value
```

MIM is built with a `6×6` spatial grid around each keypoint with circular Gaussian weighting to construct the final descriptor (this is what gets matched, not raw PC values — PC maps directly are too sparse, giving >95% outliers per RIFT paper).

### 3.3 Code to write

```python
# core/phase_congruency_mim.py

def log_gabor_filter_bank(shape: tuple, n_scales=3, n_orientations=6) -> list:
    """Construct Log-Gabor filters in frequency domain for given scales/orientations."""

def compute_even_odd_responses(image: np.ndarray, filter_bank: list) -> tuple:
    """Convolve image with even/odd Log-Gabor filters -> e_so, o_so arrays."""

def phase_congruency(image: np.ndarray, n_scales=3, n_orientations=6,
                       noise_threshold=None) -> np.ndarray:
    """Full Kovesi PC computation. Returns PC(x,y) map."""

def moment_analysis(pc_per_orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (M_edges, m_corners) moment maps."""

def compute_mim(amplitude_per_orientation: np.ndarray) -> np.ndarray:
    """RIFT Maximum Index Map: argmax orientation channel per pixel."""

def mim_descriptor(mim: np.ndarray, keypoints: np.ndarray, grid_size=6) -> np.ndarray:
    """6x6 grid histogram descriptor around each keypoint, Gaussian-weighted."""
```

---

## 4. Stage 3 — Dense Matching & Spatial Uniformity

**Files:** `core/dense_matcher.py`, `core/anms_spatial_filter.py`

### 4.1 Tasks
- Run a detector-free dense/semi-dense matcher (LoFTR primary, RoMa as fallback for extreme baselines >10x scale ratio) on the MIM/PC representation or raw normalized tiles
- Apply Adaptive Non-Maximal Suppression (ANMS) to enforce uniform spatial spread of match points (no clustering on crater rims, none skipped on maria)

### 4.2 Math

**LoFTR dual-softmax matching score:**

```
S(i,j) = (1/τ) · ⟨F̃_A_tr(i), F̃_B_tr(j)⟩

P_c(i,j) = softmax(S(i,·))_j · softmax(S(·,j))_i
```
Matches selected where `P_c(i,j) > θc` and mutual nearest neighbor (MNN) holds.

Fine-level refinement: crop `5×5` window from `1/2`-res fine feature maps around each coarse match, correlate, compute `(Δx, Δy)` as expectation over the correlation heatmap.

**RoMa (fallback for extreme scale ratios):** Uses frozen DINOv2 (ViT-L/14) features; casts matching as regression-by-classification over discrete anchors; 5 multi-scale refiners at strides `{14, 8, 4, 2, 1}` using Charbonnier/Welsch robust losses to iteratively refine a dense warp field `Ŵ_{A→B}` with pixel-wise certainty `p(x_A)`.

**Adaptive Non-Maximal Suppression (Brown & Szeliski):**

```
r_i = min_j ‖x_i - x_j‖₂   subject to   f(x_i) < c_robust · f(x_j),   x_j ∈ I

c_robust = 0.9
```
- Global maximum point → `r = ∞` (never suppressed)
- Sort all points by `r_i` descending
- Keep top `K` points → maximal spatial separation guaranteed

### 4.3 Code to write

```python
# core/dense_matcher.py

def load_loftr_model(weights_path: str, device='cuda') -> object:
    """Load pretrained LoFTR (PyTorch/Kornia)."""

def load_roma_model(weights_path: str, device='cuda') -> object:
    """Load pretrained RoMa for extreme-baseline fallback."""

def run_dense_matching(img_a: np.ndarray, img_b: np.ndarray, model,
                         confidence_thresh=0.5) -> np.ndarray:
    """Returns Nx5 array: [x1, y1, x2, y2, confidence]."""

def select_matcher(gsd_ratio: float):
    """gsd_ratio > 10 -> RoMa, else -> LoFTR."""


# core/anms_spatial_filter.py

def compute_suppression_radius(points: np.ndarray, strengths: np.ndarray,
                                  c_robust=0.9) -> np.ndarray:
    """Returns r_i for every point per Brown & Szeliski formula."""

def anms_select(points: np.ndarray, strengths: np.ndarray, k: int) -> np.ndarray:
    """Sort by suppression radius desc, return top-k indices."""

def quadtree_bucket_fallback(points: np.ndarray, grid=8) -> np.ndarray:
    """Simple grid-quota alternative if ANMS proves too slow for hackathon timeline."""
```

---

## 5. Stage 4 — Robust Geometric Verification

**File:** `core/geometric_verification.py`

### 5.1 Tasks
- Reject outlier matches using MAGSAC++ (threshold-free robust estimator)
- Fit a non-rigid transformation (Thin Plate Spline) instead of a global homography, to account for lunar relief displacement (craters, central peaks)

### 5.2 Math

**Standard RANSAC limitation:** fixed inlier threshold `τ` — too small rejects true correspondences on steep slopes, too large lets shadow artifacts corrupt the model.

**MAGSAC++ marginalized weight function:**

```
w(r) = ∫₀^σmax g(r|σ) f(σ) dσ
```
`g(r|σ)` = trimmed χ² distribution with `n` degrees of freedom (`n=4` for homography).

**Closed-form σ-consensus++ (via incomplete gamma function Γ):**

```
w(r) = [C(n)·2^((n-1)/2) / σmax] · [ Γ((n-1)/2, r²/(2σmax²)) - Γ((n-1)/2, k²/2) ]
```
Solved via Iteratively Reweighted Least Squares (IRLS). Sampling accelerated with Progressive NAPSAC (samples spatially-proximate point pairs, since neighboring points likely share the same local terrain plane).

Available directly via `cv2.USAC_MAGSAC` in OpenCV — **no need to implement MAGSAC++ from scratch**, just use `cv2.findHomography(..., method=cv2.USAC_MAGSAC)` or `cv2.estimateAffinePartial2D` with `cv2.USAC_MAGSAC`.

**Thin Plate Spline (TPS) — elastic non-rigid model:**

TPS models the warp as a global affine term plus a weighted sum of radial basis functions centered at control points (the verified matches), minimizing bending energy:

```
f(x, y) = a1 + ax·x + ay·y + Σ_i wi · U(‖(x,y) - (xi, yi)‖)

U(r) = r² · ln(r)
```

Solved as a linear system for `{a1, ax, ay, wi}` given control point correspondences, subject to the constraint that the sum and first moments of `wi` vanish (this keeps the spline "as flat as possible" while passing through the constraints, allowing local bending only where relief demands it).

### 5.3 Code to write

```python
# core/geometric_verification.py

def magsac_filter(pts_a: np.ndarray, pts_b: np.ndarray,
                    model='homography') -> tuple[np.ndarray, np.ndarray]:
    """
    cv2.findHomography(pts_a, pts_b, method=cv2.USAC_MAGSAC)
    Returns (inlier_mask, transformation_matrix)
    """

def fit_thin_plate_spline(src_pts: np.ndarray, dst_pts: np.ndarray) -> object:
    """
    Solve TPS linear system (scipy.interpolate.RBFInterpolator with
    'thin_plate_spline' kernel, or custom solve).
    Returns a callable TPS transform object.
    """

def check_relief_significance(inlier_pts: np.ndarray, homography_residuals: np.ndarray,
                                 threshold_px: float) -> bool:
    """
    Decide whether homography residuals are large enough to justify
    switching to full TPS (saves compute when terrain is flat).
    """
```

---

## 6. Stage 5 — Sub-pixel Refinement, Warping & Evaluation

**Files:** `core/subpixel_refiner.py`, `core/warp_and_eval.py`

### 6.1 Tasks
- Refine each verified match to sub-pixel precision using Lucas-Kanade gradient descent on local patches
- Warp the source image into the reference coordinate frame using the fitted TPS/homography
- Compute the SIH evaluation metrics: RMSE, Inlier Ratio, Spatial Distribution Index (Shannon entropy)

### 6.2 Math

**Lucas-Kanade continuous registration** — minimize:

```
E = Σ_{x∈R} [ F(x + h) - G(x) ]²
```
Linearize `F(x+h) ≈ F(x) + hᵀ∇F(x)`:

```
E ≈ Σ [ F(x) + Δx·∂F/∂x + Δy·∂F/∂y - G(x) ]²
```

Setting `∂E/∂h = 0` gives the 2x2 normal equations:

```
[Δx]   [ Σ Ix²      Σ Ix·Iy ]⁻¹  [ Σ Ix·(G-F) ]
[Δy] = [ Σ Ix·Iy    Σ Iy²   ]    [ Σ Iy·(G-F) ]
```

Run 5–10 iterations on `15×15` patches around each verified tie-point. Target: localization error `< 0.3 px`.

**Evaluation metrics (SIH targets):**

```
RMSE = sqrt( (1/N) Σ ‖p_reprojected_i - p_reference_i‖² )        target: < 0.5 px

Inlier Ratio = (# verified inliers) / (# total candidate matches)  target: > 75%

Spatial Distribution Index (SDI) — Shannon entropy over an 8x8 grid of match counts:
    p_k = (matches in cell k) / (total matches)
    SDI = - Σ_k p_k · log2(p_k)          higher = more uniform coverage
```

### 6.3 Code to write

```python
# core/subpixel_refiner.py

def lucas_kanade_refine(img_f: np.ndarray, img_g: np.ndarray,
                          point: tuple, patch_size=15, iterations=8) -> tuple:
    """
    Runs LK gradient descent on a patch_size x patch_size window.
    Returns sub-pixel refined (x, y).
    """

def refine_all_matches(img_a, img_b, matches: np.ndarray) -> np.ndarray:
    """Apply lucas_kanade_refine to every verified match."""


# core/warp_and_eval.py

def warp_image(source_img: np.ndarray, transform, ref_shape: tuple) -> np.ndarray:
    """cv2.remap / Lanczos interpolation warp using TPS or homography."""

def compute_rmse(reprojected_pts: np.ndarray, reference_pts: np.ndarray) -> float:
    ...

def compute_inlier_ratio(n_inliers: int, n_candidates: int) -> float:
    ...

def compute_sdi(match_points: np.ndarray, image_shape: tuple, grid=8) -> float:
    """Shannon entropy of match distribution across an 8x8 grid."""

def generate_residual_error_map(reprojected_pts, reference_pts, image_shape) -> np.ndarray:
    """Vector field / heatmap of per-point residual error, for visualization."""

def export_geotiff(warped_img: np.ndarray, ref_meta, out_path: str):
    """Write registered output with correct CRS/transform via rasterio."""

def build_metrics_report(rmse, inlier_ratio, sdi) -> dict:
    """JSON-serializable metrics dict returned by the API."""
```

---

## 7. Pipeline Orchestrator

**File:** `pipeline/orchestrator.py`

```python
class JobState(Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    MATCHING = "matching"
    VERIFYING = "verifying"
    REFINING = "refining"
    DONE = "done"
    FAILED = "failed"

class LunaMatchPipeline:
    def __init__(self, job_id: str, img_a_path: str, img_b_path: str):
        ...

    def run(self) -> dict:
        """
        1. ingest_preprocess.read_raster + lommel_seeliger_normalize + align_gsd
        2. phase_congruency_mim.phase_congruency + compute_mim
        3. dense_matcher.run_dense_matching -> anms_spatial_filter.anms_select
        4. geometric_verification.magsac_filter -> fit_thin_plate_spline
        5. subpixel_refiner.refine_all_matches
        6. warp_and_eval.warp_image + compute_rmse/inlier_ratio/sdi + export_geotiff
        Updates self.state at each transition. Returns final metrics + output paths.
        """

    def get_status(self) -> JobState:
        ...
```

---

## 8. API Layer

**File:** `api/main.py` (FastAPI)

| Endpoint | Method | Purpose |
|---|---|---|
| `/register` | POST | Accepts two images (multipart) + optional metadata, creates job, kicks off pipeline |
| `/jobs/{job_id}` | GET | Returns current `JobState` |
| `/jobs/{job_id}/result` | GET | Returns registered GeoTIFF path + metrics JSON + residual map |
| `/jobs/{job_id}/preview` | GET | (optional, for demo) returns before/after overlay PNG |

```python
# api/schemas.py
class RegisterRequest(BaseModel):
    sensor_a: str   # "OHRC" | "TMC-2" | "IIRS" | "NAC"
    sensor_b: str
    # images sent as multipart file uploads, not in JSON body

class JobStatusResponse(BaseModel):
    job_id: str
    state: str

class JobResultResponse(BaseModel):
    job_id: str
    rmse: float
    inlier_ratio: float
    sdi: float
    registered_image_path: str
    residual_map_path: str
```

---

## 9. Data & Storage Schema

```
data/jobs/{job_id}/
  input/
    image_a.tif
    image_b.tif
    metadata.json          # sensor names, GSD, sun angles
  intermediate/
    pyramid_a/
    pc_map_a.npy
    mim_a.npy
    matches_raw.npy         # [x1,y1,x2,y2,confidence]
    matches_anms.npy
    matches_verified.npy    # post-MAGSAC++
    tps_params.json
  output/
    registered.tif
    residual_map.png
    metrics.json
  status.json               # current JobState + timestamps
```

No database needed for the hackathon MVP — folder-per-job on disk is sufficient. Add SQLite only if judges specifically probe persistence/concurrency.

---

## 10. Dependencies (`requirements.txt`)

```
rasterio
gdal
numpy
scipy
opencv-python           # cv2.USAC_MAGSAC, cv2.remap
opencv-contrib-python
torch
kornia                  # differentiable CV ops, LoFTR integration
fastapi
uvicorn
pydantic
pillow
matplotlib              # residual map visualization
```

Pretrained weights needed: LoFTR (outdoor/indoor checkpoint, fine-tune later if time allows), RoMa (optional fallback).

---

## 11. Testing Plan

| Test file | Validates |
|---|---|
| `test_ingest.py` | Raster reading, GSD extraction, Lommel-Seeliger normalization output range |
| `test_phase_congruency.py` | PC map invariance under synthetic brightness/contrast shifts (same structure, different lighting → similar PC output) |
| `test_matcher.py` | Match count + confidence distribution on a known synthetic pair (same crater, shifted) |
| `test_geometric.py` | MAGSAC++ correctly rejects synthetic outlier matches; TPS reduces residuals vs. homography on synthetic relief |
| `test_subpixel.py` | LK refinement converges and reduces residual on synthetic sub-pixel shifted patches |
| `test_pipeline_e2e.py` | Full run on one real OHRC/NAC crater pair; checks RMSE < target, inlier ratio > target |

---

## 12. Phased Build Order (priority-ranked)

**Phase 1 — MVP skeleton (get something end-to-end first)**
1. `ingest_preprocess.py`: raster read + basic resolution matching (skip Lommel-Seeliger initially, plain normalization ok)
2. Classical matcher fallback: SIFT/ORB + `cv2.findHomography(RANSAC)` (not MAGSAC++ yet)
3. `warp_and_eval.py`: warp + RMSE + inlier ratio
4. Wire into `orchestrator.py` sequential run, no API yet — just a script

**Phase 2 — Core differentiators**
5. `phase_congruency_mim.py`: full Kovesi PC + RIFT MIM
6. Swap matcher to LoFTR (`dense_matcher.py`)
7. Swap RANSAC → MAGSAC++ (`cv2.USAC_MAGSAC`)

**Phase 3 — Precision & robustness**
8. `anms_spatial_filter.py`: ANMS uniform spatial spread
9. `fit_thin_plate_spline` non-rigid warp
10. `subpixel_refiner.py`: Lucas-Kanade refinement
11. `compute_sdi` + residual error map visualization

**Phase 4 — Productization**
12. `api/main.py`: FastAPI wrapper, 3 endpoints
13. Before/after overlay visualization for the demo
14. `octave pyramid` + GSD bridging refinement for the full OHRC↔IIRS range
15. RoMa fallback for extreme scale ratios (only if time remains)

---

## 13. Module → Literature Cross-Reference

| Module File | Literature Basis |
|---|---|
| `ingest_preprocess.py` | Nefian et al. (NASA Ames, Lommel-Seeliger), LROC NAC Processing Guide |
| `phase_congruency_mim.py` | Kovesi (1999), Li et al. — RIFT (2018) |
| `dense_matcher.py` | Sun et al. — LoFTR (2021), Edstedt et al. — RoMa (2024) |
| `anms_spatial_filter.py` | Brown & Szeliski (CVPR 2005) |
| `geometric_verification.py` | Barath et al. — MAGSAC++ (2020) |
| `subpixel_refiner.py` | Lucas & Kanade (1981) |
| `warp_and_eval.py` | SIH26166 problem statement metrics spec |

---

*End of plan.*
