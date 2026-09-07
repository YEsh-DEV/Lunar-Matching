# LUNA-MATCH: Complete Backend Architecture Document

**Problem Statement:** SIH26166 — Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)

**Team:** Vadapav  
**Target:** Working hackathon prototype with credible technical depth

> **Note:** `LUNA-MATCH-Backend-Plan.md` uses the file/folder naming convention actually implemented in `core/`, `pipeline/`, and `api/`. Refer to that document for the canonical module layout.

---

## 1. Executive Summary

This document translates the research literature and technical critique into a concrete, buildable backend architecture. The pipeline has **4 core modules** plus orchestration/API layers. Each module maps to specific algorithms from the referenced papers, with clear "MVP vs Stretch" boundaries so the team can deliver a working demo in the hackathon window while signaling research-grade depth to judges.

---

## 2. Problem Decomposition (What We Actually Solve)

| Challenge | Physical Cause | Why Standard CV Fails | Our Solution |
|-----------|----------------|----------------------|--------------|
| **Sun-angle / shadows** | No lunar atmosphere → hard-edged shadows shift with solar incidence angle | SIFT/ORB lock onto shadow boundaries, not terrain structure | Phase congruency (Kovesi) + RIFT descriptor (Li et al.) — illumination-invariant structural features |
| **Scale gap (8×–250×)** | OHRC ~0.3 m/px, TMC-2 ~5 m/px, IIRS ~80 m/px | Single scale-space (SIFT) cannot bridge 250× directly | Resolution pyramid with TMC-2 as intermediate anchor; hierarchical matching coarse→fine |
| **Cross-modal (pan vs hyperspectral)** | OHRC/NAC = panchromatic; IIRS = 256+ spectral bands | Raw pixel correlation meaningless across modalities | Band synthesis → pseudo-panchromatic layer; then structural matching on phase congruency |
| **Terrain relief (non-flat)** | Craters, central peaks, rille walls → parallax displacement | Global homography assumes planar scene → systematic error | Non-rigid TPS/RBF warp; [stretch] RPC bundle adjustment |
| **Clustered matches** | Detectors flood crater rims, leave maria empty | Uniform coverage needed for stable warp | Adaptive Non-Maximal Suppression (Brown et al. CVPR 2005) / quadtree grid thinning |
| **Outlier correspondences** | Repetitive patterns, shadow edges, noise | Standard RANSAC sensitive to threshold, slow | MAGSAC++ (Barath et al.) with Progressive NAPSAC sampler — threshold-free, faster, more accurate |

---

## 3. End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLIENT REQUEST                                    │
│                    POST /register  {image_A, image_B, meta}                 │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI APPLICATION (api/main.py)                   │
│  • Request validation & job creation                                        │
│  • Async job queue (in-memory for hackathon; Redis if time allows)          │
│  • Status polling endpoint  GET /jobs/{id}                                  │
│  • Result delivery        GET /jobs/{id}/result                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATOR (pipeline/orchestrator.py)         │
│  State machine: PENDING → PREPROCESS → MATCH → VERIFY → WARP → DONE/FAILED │
│  • Calls modules in strict sequence                                         │
│  • Tracks intermediate artifacts per job_id                                 │
│  • Error handling & partial-result persistence                              │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  MODULE 1        │   │  MODULE 2        │   │  MODULE 3        │
│  Ingestion &     │──▶│  Feature         │──▶│  Verification &  │
│  Pre-scaling     │   │  Extraction &    │   │  Filtering       │
│  (core/ingest.py)│   │  Matching        │   │  (core/verify.py)│
└──────────────────┘   │  (core/match.py) │   └────────┬─────────┘
                       └──────────────────┘            │
                                                       ▼
                                              ┌──────────────────┐
                                              │  MODULE 4        │
                                              │  Sub-pixel &     │
                                              │  Warping         │
                                              │  (core/warp.py)  │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  STORAGE         │
                                              │  data/jobs/{id}/ │
                                              │  input/, output/ │
                                              └──────────────────┘
```

---

## 4. Module Specifications (The Actual Build)

### MODULE 1: Ingestion & Pre-scaling  (`core/ingestion.py`)

**Inputs:** Two GeoTIFF/PDS4/IMG files + metadata (GSD, solar incidence/emission/phase angles, sensor model)

**Outputs:** 
- `ref_pyramid[]`, `src_pyramid[]` — lists of `np.ndarray` at aligned resolutions
- `metadata.json` — normalized GSD, solar angles, sensor info per level

**Algorithm Steps:**

| Step | Operation | Library | Key Parameters |
|------|-----------|---------|----------------|
| 1.1 | Read raster + metadata | `rasterio` / `GDAL` | Preserve CRS, transform, nodata |
| 1.2 | Radiometric calibration | Sensor-specific | OHRC/NAC: radiance→reflectance; IIRS: radiance per band |
| 1.3 | **Photometric normalization** | Custom | Lunar-Lambertian model (Nefian et al. NASA 2014) or Hapke if time; at minimum: `Lommel-Seeliger` cosine correction using solar incidence angle |
| 1.4 | **IIRS band synthesis** | Custom | Weighted sum of bands to create pseudo-panchromatic; weights from literature (e.g., 0.7μm continuum band dominant) |
| 1.5 | **Resolution pyramid construction** | `cv2.resize` / `skimage.transform` | Target GSD = TMC-2 (~5 m/px) as anchor; build pyramid levels at 2× steps: OHRC→TMC-2→IIRS |
| 1.6 | Denoising | `cv2.fastNlMeansDenoising` | Preserve edges; `h=10`, `templateWindow=7`, `searchWindow=21` |
| 1.7 | Save intermediate pyramids | `numpy.savez_compressed` | `data/jobs/{id}/pyramids.npz` |

**MVP Scope:** Steps 1.1, 1.2, 1.5, 1.6, 1.7 — basic pyramid + denoise  
**Stretch:** 1.3 (full photometric model), 1.4 (IIRS band synthesis)

---

### MODULE 2: Feature Extraction & Matching  (`core/matching.py`)

**Inputs:** `ref_pyramid[]`, `src_pyramid[]` from Module 1

**Outputs:** 
- `matches_coarse[]` — (x_ref, y_ref, x_src, y_src, confidence) at pyramid level 0
- `matches_fine[]` — refined to full resolution

**Algorithm Choice (Priority Order):**

| Tier | Method | Paper | Pros | Cons | Hackathon Viability |
|------|--------|-------|------|------|---------------------|
| **A (Primary)** | **LoFTR** (pretrained) | Sun et al. 2021 | Dense matches in low-texture; transformer global context; PyTorch/Kornia ready | Heavy model (~100MB); needs GPU for speed | **High** — `kornia.feature.LoFTR` pretrained on outdoor; works OOTB |
| **B (Fallback)** | **RoMa** (pretrained) | Edstedt et al. 2023 | More robust to extreme scale/illum; DINOv2 backbone | Newer, less battle-tested in Python | **Medium** — if LoFTR fails on lunar |
| **C (Classical)** | **RIFT** (Phase Congruency + MIM) | Li et al. 2018 (RIFT paper) | **Built for multi-modal + NRD**; rotation-invariant; no GPU needed | Implementation effort; no PyPI package | **Medium-High** — implement PC + MIM from paper; best for "we solved multi-modal" story |
| **D (Baseline)** | **SIFT/ORB + ANMS** | Brown et al. 2005 | Trivial OpenCV; ANMS gives uniform spread | Fails on shadows, cross-modal, large scale | **Trivial** — safety net for Demo Day |

**Recommended Hackathon Strategy:**
1. **Week 1:** Get Tier D working end-to-end (SIFT + ANMS + RANSAC + warp)
2. **Week 2:** Swap Tier D → Tier A (LoFTR via Kornia) — this is the "wow" moment
3. **Week 3 (if time):** Implement Tier C (RIFT/PC) for cross-modal IIRS demo

**Module 2 Internal Flow:**

```python
def match_pyramids(ref_pyr, src_pyr, method='loftr'):
    # Coarse level (pyramid level 0 ≈ TMC-2 GSD)
    matches_coarse = coarse_match(ref_pyr[0], src_pyr[0], method)
    
    # Upsample & refine at each finer level
    for level in range(1, len(ref_pyr)):
        matches_fine = refine_level(ref_pyr[level], src_pyr[level], 
                                     upsample(matches_coarse))
        matches_coarse = matches_fine
    
    return matches_fine  # at full resolution
```

**Coarse Matching (LoFTR via Kornia):**
```python
from kornia.feature import LoFTR
matcher = LoFTR(pretrained='outdoor')  # or 'indoor' — test both
# Input: grayscale tensors [1,1,H,W], normalized to [0,1]
# Output: dict with 'keypoints0', 'keypoints1', 'confidence'
```

**Classical Fallback (SIFT + ANMS):**
```python
# Phase congruency pre-filter (Kovesi) for illumination invariance
pc_map = phase_congruency(gray_image)  # Implement from Kovesi 1999
# Detect on PC map instead of raw intensity
kps = cv2.SIFT_create().detect(pc_map)
# ANMS for uniform distribution (Brown et al. 2005)
kps_uniform = anms(kps, num_points=2000)
# Describe on original (or PC) image
descs = sift.compute(gray_image, kps_uniform)
```

---

### MODULE 3: Verification & Filtering  (`core/verification.py`)

**Inputs:** Raw matches from Module 2 (typically 2000–5000 candidates)

**Outputs:** Verified inlier matches + geometric model

**Algorithm Pipeline:**

| Stage | Algorithm | Paper | Parameters |
|-------|-----------|-------|------------|
| 3.1 | **Grid/Quadtree Thinning** | Brown et al. 2005 (ANMS) | 50×50 grid, max 4 pts/cell |
| 3.2 | **MAGSAC++** | Barath et al. 2019 (MAGSAC++) | `max_iterations=10000`, `confidence=0.999`, P-NAPSAC sampler |
| 3.3 | **Model Selection** | — | Try Homography → if residual pattern shows systematic error, upgrade to TPS |

**MAGSAC++ Implementation (via `pyransac` or custom):**
```python
# Use cv2.findHomography with method=cv2.USAC_MAGSAC (OpenCV 4.8+)
# Or implement full MAGSAC++ from paper for threshold-free operation
H, inlier_mask = cv2.findHomography(
    src_pts, ref_pts, 
    method=cv2.USAC_MAGSAC,
    ransacReprojThreshold=3.0,  # fallback; MAGSAC++ marginalizes this
    confidence=0.999,
    maxIters=10000
)
```

**Non-Rigid Upgrade (TPS) — Stretch:**
```python
from scipy.interpolate import Rbf
# Thin-plate spline: radial basis function with r^2 * log(r)
rbf_x = Rbf(src_pts[:,0], src_pts[:,1], ref_pts[:,0], function='thin_plate')
rbf_y = Rbf(src_pts[:,0], src_pts[:,1], ref_pts[:,1], function='thin_plate')
# Warp dense grid
warp_x = rbf_x(grid_x, grid_y)
warp_y = rbf_y(grid_x, grid_y)
```

---

### MODULE 4: Sub-pixel Refinement & Warping  (`core/warping.py`)

**Inputs:** Verified matches + full-resolution images

**Outputs:** 
- Registered GeoTIFF (source warped to reference CRS)
- Residual map (per-pixel error)
- Metrics JSON (RMSE, inlier ratio, sub-pixel accuracy)

**Algorithm Steps:**

| Step | Operation | Algorithm | Reference |
|------|-----------|-----------|-----------|
| 4.1 | **Sub-pixel refinement** | Lucas-Kanade (iterative gradient descent) on 11×11 patches | Lucas & Kanade 1981 |
| 4.2 | **Final warp estimation** | Homography (MVP) / TPS (stretch) on refined matches | — |
| 4.3 | **Image warping** | `cv2.warpPerspective` / `scipy.ndimage.map_coordinates` | — |
| 4.4 | **GeoTIFF write** | `rasterio` with updated transform + CRS | — |
| 4.5 | **Quality metrics** | RMSE on check points; inlier ratio; sub-pixel residual histogram | — |

**Lucas-Kanade Sub-pixel (per match):**
```python
def refine_match(ref_img, src_img, pt_ref, pt_src, window=11, max_iter=10):
    # Iterative gradient-based refinement (Lucas-Kanade)
    # Minimizes SSD between warped patch and reference patch
    # Returns sub-pixel displacement (dx, dy) < 0.5 px target
```

---

## 5. Data Structures & Storage Layout

```
data/
└── jobs/
    └── {job_id}/
        ├── input/
        │   ├── ref.tif          # Reference image (e.g., LRO NAC map-projected)
        │   └── src.tif          # Source image (OHRC/TMC-2/IIRS)
        ├── metadata.json        # Sensor, GSD, solar angles, CRS per image
        ├── pyramids.npz         # Compressed pyramid arrays (Module 1 output)
        ├── matches_coarse.npz   # Raw matches at pyramid level 0
        ├── matches_fine.npz     # Refined matches at full resolution
        ├── inliers.npz          # Post-verification inliers
        ├── output/
        │   ├── registered.tif   # Warped source in reference frame
        │   ├── residual_map.tif # Per-pixel registration error
        │   └── overlay.png      # Visualization (alpha blend)
        └── metrics.json         # RMSE, inlier_ratio, n_matches, time_per_stage
```

---

## 6. API Contract (FastAPI)

```python
# POST /register
{
    "ref_image": "base64_or_multipart",
    "src_image": "base64_or_multipart", 
    "ref_metadata": {
        "sensor": "LRO_NAC",
        "gsd_m": 0.5,
        "solar_incidence_deg": 45.2,
        "solar_azimuth_deg": 120.0,
        "crs": "EPSG:4326",
        "transform": [a,b,c,d,e,f]
    },
    "src_metadata": { ... same fields ... },
    "options": {
        "matcher": "loftr",          # loftr | roma | rift | sift
        "warp_type": "homography",   # homography | tps
        "max_keypoints": 2000
    }
}

# Response: {"job_id": "uuid", "status": "pending"}

# GET /jobs/{job_id}
{"job_id": "...", "status": "warping", "progress": 0.8, "current_stage": "subpixel_refinement"}

# GET /jobs/{job_id}/result
{
    "job_id": "...",
    "status": "done",
    "registered_image_url": "/data/jobs/{id}/output/registered.tif",
    "metrics": {
        "rmse_px": 0.34,
        "inlier_ratio": 0.82,
        "num_matches": 1847,
        "processing_time_sec": 42.3
    }
}
```

---

## 7. Software Stack & Dependencies

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **API** | FastAPI | 0.110+ | Async REST endpoints |
| | Uvicorn | 0.29+ | ASGI server |
| **Computer Vision** | OpenCV | 4.8+ | SIFT, RANSAC (USAC_MAGSAC), warp |
| | Kornia | 0.6+ | LoFTR, differentiable CV ops |
| | PyTorch | 2.2+ | Deep learning backend |
| **Geospatial** | Rasterio | 1.3+ | GeoTIFF I/O, CRS handling |
| | GDAL | 3.8+ | PDS4/IMG support, reprojection |
| | PyProj | 3.6+ | Coordinate transformations |
| **Scientific** | NumPy | 1.26+ | Array operations |
| | SciPy | 1.10+ | Rbf (TPS), optimization |
| | scikit-image | 0.22+ | Pyramid, denoising, phase congruency |
| **Utils** | Pydantic | 2.7+ | Request validation |
| | python-multipart | 0.0.6+ | File upload |

**Install:**
```bash
pip install fastapi uvicorn opencv-python kornia torch torchvision \
            rasterio gdal pyproj numpy scipy scikit-image pydantic python-multipart
```

---

## 8. Folder Structure (Final)

```
luna-match/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, job management
│   └── schemas.py           # Pydantic models for request/response
├── core/
│   ├── __init__.py
│   ├── ingestion.py         # Module 1: read, calibrate, pyramid, photometric norm
│   ├── matching.py          # Module 2: LoFTR/RIFT/SIFT + ANMS
│   ├── verification.py      # Module 3: grid thinning + MAGSAC++ + model selection
│   └── warping.py           # Module 4: Lucas-Kanade + warp + GeoTIFF write
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py      # Sequential pipeline runner + state tracking
├── utils/
│   ├── __init__.py
│   ├── phase_congruency.py  # Kovesi 1999 implementation (log-Gabor)
│   ├── rift_descriptor.py   # MIM construction + rotation invariance
│   ├── photometric.py       # Lunar-Lambertian / Hapke normalization
│   ├── anms.py              # Adaptive Non-Maximal Suppression (Brown 2005)
│   └── magsac.py            # MAGSAC++ wrapper (if not using OpenCV built-in)
├── data/
│   └── jobs/                # Runtime job artifacts (gitignored)
├── tests/
│   ├── test_ingestion.py
│   ├── test_matching.py
│   ├── test_verification.py
│   └── test_warping.py
├── scripts/
│   ├── download_test_data.sh
│   └── benchmark.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 9. Build Order (Critical Path)

| Phase | Goal | Deliverable | Timebox |
|-------|------|-------------|---------|
| **0** | Repo init + CI | `git init`, `pre-commit`, GitHub Actions lint/test | 0.5 day |
| **1** | **End-to-end baseline** | SIFT + ANMS + Homography (OpenCV) + warp + API → registered.tif | **2 days** |
| **2** | Pyramid + photometric | Module 1 complete with GSD bridging + basic cosine correction | 1 day |
| **3** | **LoFTR integration** | Swap matcher → LoFTR (Kornia pretrained outdoor) | 1.5 days |
| **4** | MAGSAC++ + grid thinning | Replace cv2.RANSAC → MAGSAC++ + ANMS grid | 1 day |
| **5** | Sub-pixel refinement | Lucas-Kanade per-match refinement | 0.5 day |
| **6** | Non-rigid warp (TPS) | SciPy Rbf thin-plate spline | 1 day [stretch] |
| **7** | IIRS cross-modal demo | Band synthesis + RIFT/PC matcher on IIRS vs OHRC | 1.5 days [stretch] |
| **8** | Demo polish | Overlay viz, metrics dashboard, 3-min demo script | 1 day |

**Total: ~8–10 days** — feasible for a 2-person team with clear ownership:
- **Person A:** Modules 1, 2 (ingestion + matching)
- **Person B:** Modules 3, 4 (verification + warping) + API/orchestrator

---

## 10. Key Algorithms — Implementation Notes from Papers

### 10.1 Phase Congruency (Kovesi 1999) — Illumination-Invariant Features
- **Why:** Dimensionless, invariant to brightness/contrast/shadow
- **Implementation:** Log-Gabor wavelets at multiple orientations/scales
- **PC(x) = Σ|Wₙ(x)| / Σ|Wₙ(x)| + ε** — numerator: weighted sum of wavelet responses in phase; denominator: local energy + noise compensation
- **Use:** Compute PC map → detect keypoints on PC (not raw intensity) → describe with MIM or SIFT

### 10.2 RIFT Descriptor (Li et al. 2018) — Radiation-Invariant for Multi-Modal
- **Detection:** Corner + edge points on PC map (better repeatability + count)
- **Description:** Maximum Index Map (MIM) from log-Gabor convolution sequence
  - MIM(i,j) = argmaxₖ |Wₖ(i,j)| where k = orientation index
  - Robust to Nonlinear Radiation Distortions (NRD)
- **Rotation Invariance:** Construct multiple MIMs at rotated orientations; match via cyclic shift
- **Relevance:** **Directly addresses OHRC↔IIRS cross-modal matching**

### 10.3 LoFTR (Sun et al. 2021) — Detector-Free Dense Matching
- **Architecture:** CNN backbone (1/8 resolution) → Linear Transformer (self+cross attention) → Coarse matches → Fine refinement via correlation
- **Key:** Global receptive field → matches in low-texture (maria)
- **Kornia API:** `kornia.feature.LoFTR(pretrained='outdoor')` — one-liner integration

### 10.4 RoMa (Edstedt et al. 2023) — Robust Dense Matching
- **Improvement over LoFTR:** Frozen DINOv2 coarse features + specialized ConvNet fine features
- **Match Decoder:** Transformer predicting anchor probabilities (multimodal) not coordinates
- **Loss:** Regression-by-classification (coarse) + Robust regression (fine)
- **Use if:** LoFTR fails on extreme lunar illumination/scale

### 10.5 MAGSAC++ (Barath et al. 2019) — Threshold-Free Robust Estimation
- **Core:** Marginalizes over noise scale σ → no manual inlier threshold
- **Model Quality:** Iteratively Reweighted Least Squares (IRLS) — fast, accurate
- **Sampler:** Progressive NAPSAC (P-NAPSAC) — local→global blending exploits spatial coherence
- **OpenCV 4.8+:** `cv2.USAC_MAGSAC` available; fallback to custom impl for full control

### 10.6 Adaptive Non-Maximal Suppression (Brown et al. 2005) — Uniform Distribution
- **Problem:** SIFT clusters on high-contrast regions (crater rims)
- **Solution:** For each keypoint, compute suppression radius rᵢ = min||xᵢ-xⱼ|| s.t. f(xᵢ) < c·f(xⱼ)
- **Select:** Top N keypoints with largest rᵢ → spatially uniform
- **Parameter:** c_robust = 0.9, N = 2000–5000

### 10.7 Lucas-Kanade Sub-pixel (Lucas & Kanade 1981) — Iterative Refinement
- **Principle:** Minimize SSD via Newton-Raphson on image gradients
- **Coarse-to-fine:** Pyramid extends convergence range
- **Target:** < 0.5 px residual error

### 10.8 Lunar Photometric Model (Nefian et al. NASA 2014) — Shadow/Albedo Separation
- **Reflectance:** Lunar-Lambertian hybrid: R = (e⁻ᶜ¹ᵅ + c₂)(1-L(α))cos(i)/cos(e) + 2L(α)cos(i)/(cos(i)+cos(e))
- **Variables:** i=incidence, e=emission, α=phase angle
- **Shadow handling:** Binary mask S=1 for shadowed pixels (discard from albedo estimation)
- **Use:** Normalize OHRC/TMC-2/IIRS to common albedo before matching

---

## 11. Evaluation & Demo Metrics (What Judges Will Check)

| Metric | Target (MVP) | Target (Stretch) | Measurement |
|--------|--------------|------------------|-------------|
| **Sub-pixel RMSE** | < 0.5 px | < 0.2 px | Check points on known correspondences |
| **Inlier Ratio** | > 60% | > 80% | Post-MAGSAC++ / total candidates |
| **Match Uniformity** | Max 2× density variation across grid | Near-uniform | Quadtree cell count variance |
| **Cross-modal (IIRS→OHRC)** | Any valid matches | > 100 inliers | Visual + quantitative |
| **Sun-angle robustness** | Works at 2+ incidence angles | Works at 5+ angles | Test on multi-temporal pairs |
| **Processing Time** | < 60 sec / pair | < 30 sec / pair | End-to-end on 5000×5000 images |
| **Scale Gap Handling** | OHRC↔TMC-2 (8×) | OHRC↔IIRS (250×) | Pyramid bridging success |

---

## 12. Risk Mitigation & Fallback Plan

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LoFTR fails on lunar imagery | Medium | High | Classical SIFT+ANMS baseline ready (Phase 1) |
| GPU memory OOM on large images | High | Medium | Tile-based processing (512×512 overlap); pyramid matching |
| IIRS band synthesis unknown weights | Medium | High | Use single continuum band (0.7μm) as pseudo-panchromatic |
| MAGSAC++ not in OpenCV version | Low | Medium | Implement IRLS from paper; or use cv2.RANSAC with tuned threshold |
| TPS warp too slow on full-res | Medium | Medium | Warp on downsampled (2×) then refine; or use homography MVP |
| No ground truth for evaluation | High | Medium | Use LRO NAC map-projected as reference; synthetic warp tests |

---

## 13. "Stretch" Items — Honest Framing for Judges

| Item | Status | Honest Framing |
|------|--------|----------------|
| Full Hapke photometric model | Not in MVP | "We use Lommel-Seeliger cosine correction as baseline; Hapke integration is Phase 2 for higher-fidelity albedo normalization" |
| RPC bundle adjustment | Not in MVP | "TPS non-rigid warp handles local relief; RPC BA requires precise orbit data — roadmap item for production" |
| LoFTR fine-tuned on lunar DEMs | Not in MVP | "Pretrained outdoor LoFTR works well; domain adaptation via fine-tuning on LRO NAC pairs is future work" |
| IIRS full hyperspectral synthesis | Partial | "Single-band pseudo-panchromatic for matching; full spectral unmixing for science products is downstream" |
| Real-time / on-board processing | Out of scope | "Ground pipeline prototype; quantization + TensorRT for flight software is separate track" |

---

## 14. Quick Start Commands

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run API server
uvicorn api.main:app --reload --port 8000

# 3. Test with sample data (after download_test_data.sh)
curl -X POST "http://localhost:8000/register" \
  -F "ref_image=@data/test/ref_nac.tif" \
  -F "src_image=@data/test/src_ohrc.tif" \
  -F 'ref_metadata={"sensor":"LRO_NAC","gsd_m":0.5,"solar_incidence_deg":45}' \
  -F 'src_metadata={"sensor":"CH2_OHRC","gsd_m":0.3,"solar_incidence_deg":60}'

# 4. Poll for result
curl "http://localhost:8000/jobs/{job_id}/result"
```

---

## 15. References (Paper → Code Mapping)

| Paper | Algorithm | Our Module | Implementation Status |
|-------|-----------|------------|----------------------|
| Kovesi 1999 | Phase Congruency | `utils/phase_congruency.py` | Implement from equations |
| Li et al. 2018 (RIFT) | PC + MIM descriptor | `utils/rift_descriptor.py` | Implement; core for IIRS matching |
| Brown et al. 2005 | ANMS | `utils/anms.py` | Implement from paper Section 3 |
| Lucas & Kanade 1981 | Sub-pixel refinement | `core/warping.py` | OpenCV `calcOpticalFlowPyrLK` or custom |
| Sun et al. 2021 (LoFTR) | Dense transformer matching | `core/matching.py` | `kornia.feature.LoFTR` |
| Edstedt et al. 2023 (RoMa) | Robust dense matching | `core/matching.py` | Fallback if LoFTR struggles |
| Barath et al. 2019 (MAGSAC++) | Robust estimation | `core/verification.py` | `cv2.USAC_MAGSAC` or custom IRLS |
| Nefian et al. 2014 (NASA) | Lunar-Lambertian photometry | `utils/photometric.py` | Implement Eq. 1–3 |

---

**End of Architecture Document** — This is the single source of truth for backend implementation. Each module is independently testable; the orchestrator wires them together. Build Phase 1 end-to-end first, then upgrade modules incrementally.