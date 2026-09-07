# LunarMatching (LUNA-MATCH)
### Multi-modal, Sun-angle and Scale-invariant Lunar Image Correspondence
**Smart India Hackathon 2026 — Problem Statement: SIH26166**

---

## 🛰️ Project Overview

**LUNA-MATCH** is an end-to-end, high-precision lunar image registration pipeline designed to align high-resolution multi-modal lunar orbital imagery (such as Chandrayaan-2 OHRC/TMC-2 and LRO NAC). The system solves core challenges of extraterrestrial surface correspondence:

- **Harsh & Inverted Illumination**: Deep shadows and stark sun-angle variations that break traditional intensity-based matchers (SIFT/ORB).
- **Extreme Scale & Resolution Disparities**: Substantial Ground Sampling Distance (GSD) differences across instruments (e.g., >4x scale ratios).
- **Non-Rigid Topographical Distortions**: Parallax and elevation variations across craters, ridges, and slopes.
- **Sub-Pixel Precision**: Registration accuracy verified using robust geometric estimation and iterative sub-pixel optical flow refinement.

---

## ⚡ 7-Stage Registration Pipeline

```
Image 1 (Ref) ──┐
                ├──► [1. Ingest & Preprocess] ──► [2. Phase Congruency & MIM]
Image 2 (Sens) ─┘          │                                │
                           ▼                                ▼
                   CLAHE + Normalization          Log-Gabor Wavelets (Illumination Invariance)
                                                            │
                                                            ▼
                                                  [3. Dense Matcher (LoFTR)]
                                                            │
                                                            ▼
                                                  [4. ANMS Spatial Filtering]
                                                            │
                                                            ▼
                                                  [5. Robust Geometric Verification (MAGSAC++)]
                                                            │
                                                            ▼
                                                  [6. Lucas-Kanade Sub-Pixel Refinement]
                                                            │
                                                            ▼
                                                  [7. Warping & Quantitative Evaluation]
                                                            │
                                             ┌──────────────┴──────────────┐
                                             ▼                             ▼
                                  Warped GeoTIFF + Overlays        RMSE, SDI, Inlier Ratio
```

1. **Ingestion & Preprocessing** (`core/ingest_preprocess.py`):
   - Multi-format ingestion (GeoTIFF, PNG, JPEG) with nodata masking.
   - Bilateral filtering and Contrast Limited Adaptive Histogram Equalization (CLAHE).
   - Multi-resolution octave pyramid generation for scale normalization.
2. **Illumination-Invariant Representation** (`core/phase_congruency_mim.py`):
   - Frequency-domain feature extraction using multi-scale, multi-orientation 2D Log-Gabor wavelets.
   - Computes Phase Congruency (structural edge/corner strength independent of lighting intensity) and Maximum Index Maps (MIM).
3. **Dense Feature Matching** (`core/dense_matcher.py`):
   - Dense transformer-based correspondence matching via LoFTR (Detector-Free Local Feature Matching with Transformers).
   - High-confidence candidate extraction across illumination-invariant representations.
4. **Adaptive Non-Maximal Suppression (ANMS)** (`core/anms_spatial_filter.py`):
   - Brown et al. (CVPR 2005) radius suppression.
   - Enforces uniform spatial distribution across the lunar terrain to prevent match clustering inside high-contrast crater rims.
5. **Robust Geometric Verification** (`core/geometric_verification.py`):
   - MAGSAC++ (Marginalizing Sample Consensus) with spatial covariance scoring.
   - Affine and non-rigid Thin Plate Spline (TPS) transformation modeling.
6. **Sub-Pixel Refinement** (`core/subpixel_refiner.py`):
   - Iterative Lucas-Kanade optical flow with forward-backward cross-checking and bilinear interpolation.
7. **Warping, Fusion & Quantitative Evaluation** (`core/warp_and_eval.py`):
   - High-fidelity resampling into the reference coordinate frame.
   - Metrics computation: Root Mean Square Error (RMSE), Inlier Ratio, Spatial Dispersion Index (SDI), and SSIM.
   - Visual quality diagnostics: Checkerboard overlay, side-by-side matches, and tie point residual vectors.

---

## 📊 Live Real Lunar Sample Evaluation

Tested on real lunar orbital imagery exhibiting a **~4.2× scale ratio** (Image 1 GSD: 131.6 m/px vs Image 2 GSD: 555.6 m/px) and non-overlapping perspective differences:

| Metric | Result | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Inlier Ratio** | **83.1%** | > 60% | ✅ Passed |
| **Registration RMSE** | **0.99 px** | < 1.5 px (Sub-pixel) | ✅ Passed |
| **Spatial Dispersion Index (SDI)**| **0.584** | > 0.50 | ✅ Passed |
| **Matched Inlier Pairs** | **1,525 points** | > 200 | ✅ Passed |

*Outputs generated in `demo_output/real_pair/`: checkerboard overlay, tie points, and metrics report.*

---

## 📁 Repository Structure

```
LunarMatching/
├── api/                        # FastAPI service & request schemas
│   ├── main.py
│   └── schemas.py
├── core/                       # Algorithmic registration engine
│   ├── ingest_preprocess.py
│   ├── phase_congruency_mim.py
│   ├── dense_matcher.py
│   ├── anms_spatial_filter.py
│   ├── geometric_verification.py
│   ├── subpixel_refiner.py
│   └── warp_and_eval.py
├── data/
│   ├── samples/                # Sample lunar GeoTIFF test pairs
│   └── jobs/                   # Persistent asynchronous job outputs
├── demo_output/                # Visual verification results
│   └── real_pair/              # Demo registration artifacts
├── docs/                       # Architectural specs & technical reports
│   ├── architecture.md
│   ├── frontend_architecture.md
│   ├── LUNA-MATCH-Backend-Plan.md
│   ├── Agent1_report.md
│   └── Agent2_report.md
├── models/
│   └── loftr_weights/          # Pretrained neural network weights
├── pipeline/
│   └── orchestrator.py         # End-to-end multi-stage pipeline coordinator
├── scripts/
│   └── prepare_real_samples.py # Preprocessing & GeoTIFF calibration script
├── tests/                      # Full pytest test suite (58 tests)
│   ├── test_anms.py
│   ├── test_api.py
│   ├── test_geometric.py
│   ├── test_ingest.py
│   ├── test_matcher.py
│   ├── test_phase_congruency.py
│   ├── test_pipeline_e2e.py
│   ├── test_subpixel.py
│   └── test_warp_eval.py
├── demo.py                     # Standalone CLI execution script
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Python 3.10+ recommended:

```bash
git clone https://github.com/YEsh-DEV/LunarMatching.git
cd LunarMatching

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Full Test Suite

The repository includes 58 comprehensive unit and end-to-end integration tests:

```bash
pytest tests/
```

### 3. Run the Live Demo on Real Lunar Imagery

Execute the full 7-stage registration on real lunar imagery:

```bash
python demo.py
```

The outputs (warped raster, checkerboard overlay, tie-point plots, and `metrics.json`) will be generated in `demo_output/real_pair/`.

### 4. Launch the FastAPI Microservice

To run the asynchronous background job registration API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

---

## 📖 Documentation

Detailed architectural reports, mathematical formulations, and engineering blueprints are available in the [`docs/`](docs/) directory:
- [Pipeline Architecture & Mathematical Formulations](docs/architecture.md)
- [Backend Development Plan](docs/LUNA-MATCH-Backend-Plan.md)
- [Frontend Visual Dashboard Specification](docs/frontend_architecture.md)
- [Technical Progress & Verification Reports](docs/Agent1_report.md)

---

## 📜 License

Developed for the Smart India Hackathon (SIH 2026). All rights reserved.
