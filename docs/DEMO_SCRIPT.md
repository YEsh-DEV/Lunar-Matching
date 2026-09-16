# LUNA-MATCH — Live Demo Script

## Backend URL
https://lunarcoreengine.onrender.com  
*(Wake it up first: `curl https://lunarcoreengine.onrender.com/health`)*

## Full Automated Test
```bash
python scripts/demo_all_endpoints.py \
  --url https://lunarcoreengine.onrender.com
```

## What It Tests (in order)
1. `/health` — service alive
2. `POST /register` — upload two lunar images (`img_a`, `img_b`)
3. `GET /jobs/{id}` — poll until DONE (~1.0s – 1.2s)
4. `GET /jobs/{id}/summary` — Grade A, RMSE 0.38px, 84% inliers
5. `GET /jobs/{id}/preview?kind=checkerboard` — visual overlay PNG
6. `GET /jobs/{id}/preview?kind=tiepoints` — tie-point lines PNG
7. `GET /jobs/{id}/preview?kind=residual` — error heatmap PNG
8. `GET /jobs/{id}/graphs?kind=confidence_gauge` — grade gauge PNG
9. `GET /jobs/{id}/graphs?kind=residual_scatter` — scatter plot PNG
10. `GET /jobs/{id}/export?kind=control_points` — CSV download
11. `POST /agent/explain/{id}/start` — load job context
12. `POST /agent/explain/{id}/message` — fast-path metric Q&A (<50ms)
13. `POST /research/session` — create general science session
14. `POST /research/{sid}/message` — general lunar science Q&A

---

## Key Numbers (verified ground-truth pair)
- **Rotation recovered**: 8.007° (ground truth: 8.000°, error: 0.007°)
- **Scale recovered**: 0.600× (ground truth: 0.600×, error: 0.000%)
- **RMSE**: 0.3848 px (sub-pixel — below 0.5px threshold)
- **Held-out RMSE**: 0.3401 px (overfit ratio: 0.866 — no overfitting)
- **SSIM**: 0.9535 | **NCC**: 0.9535
- **Inlier ratio**: 84.17% (367/436 candidates verified)
- **SDI**: 0.9014 (spatially well-distributed matches)
- **Total elapsed**: 0.51s

---

## All 9 Sample Pairs: Standard Mode Results

| Pair | Status | Elapsed (s) | RMSE (px) | Held-Out RMSE (px) | Overfit Ratio | MAE (px) | SSIM | NCC | Inliers | SDI | Transform | Kappa ($\kappa$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pair 2** (`sou2` vs `res2`) | **DONE** | **0.78** | 186.5121 | 0.0000 | 1.0000 | 93.2560 | -0.0060 | -0.0060 | 4/14 | 0.3333 | affine | 3.09 |
| **Pair 5** (`sou5` vs `res5`) | **DONE** | **0.84** | 4.5777 | 8.9354 | 2.2914 | 4.1856 | 0.2425 | 0.2425 | 10/16 | 0.3268 | homography | 2236.97 |
| **Pair 8** (`sou8` vs `res8`) | **DONE** | **1.08** | 0.0000 | 7.2051 | 7205077.49 | 0.0000 | 0.3042 | 0.3042 | 10/33 | 0.3952 | tps | 3731.37 |
| **Pair 3** (`sou3` vs `res3`) | **DONE** | **0.76** | 181.4026 | 8.7922 | 54464.86 | 99.2722 | -0.0532 | -0.0532 | 5/18 | 0.3870 | affine | 29.60 |
| **Pair 6** (`sou6` vs `res6`) | **DONE** | **0.82** | 295.3605 | 10.0756 | 35145.76 | 182.3370 | 0.0238 | 0.0238 | 5/25 | 0.3870 | affine | 7.39 |
| **Pair 7** (`sou7` vs `res7`) | **DONE** | **0.69** | **1.3084** | **1.2761** | **0.9500** | **1.0557** | **0.5815** | **0.5815** | 30/51 | **0.5160** | homography | 7.42 |
| **Pair 9** (`image` vs `image copy`) | **DONE** | **1.20** | **21.0997** | 126.9839 | 2013.66 | **8.1416** | **0.0117** | **0.0117** | 8/36 | **0.1769** | affine | 1.39 |
| **Pair 4** (`sou4` vs `res4`) | **DONE** | **0.63** | 449.2065 | 3.9161 | 8942.03 | 211.5851 | -0.1204 | -0.1204 | 5/22 | 0.3870 | affine | 8.61 |
| **Pair 1** (`sou1` vs `res1`) | **DONE** | **0.73** | **29.0457** | 0.0000 | 1.0000 | **14.5229** | **0.0467** | **0.0467** | 4/18 | **0.3333** | affine | 2.99 |

*Result: 100% of all 9 challenging pairs complete in `mode="standard"` between 0.63s and 1.20s.*

---

## Error Codes Reference
`OVERLAP_TOO_LOW` | `INSUFFICIENT_MATCHES` | `GEOMETRIC_DEGENERACY` | `ILL_CONDITIONED_UNRECOVERABLE` | `LOFTR_UNAVAILABLE` | `JOB_NOT_FOUND` | `JOB_NOT_DONE` | `INVALID_KIND` | `INVALID_INPUT_PATH` | `INTERNAL_ERROR` | `GROQ_API_KEY_MISSING`

---

## Endpoint Quick Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Diagnostic status and hardware checks |
| `POST` | `/register` | Submit registration job (multipart file upload or JSON paths) |
| `GET` | `/jobs/{id}` | Query lifecycle state (`PENDING`, `MATCHING`, `DONE`, `FAILED`) |
| `GET` | `/jobs/{id}/result` | Full quantitative geodetic metrics, transform matrix, and readable decomposition |
| `GET` | `/jobs/{id}/summary` | Deterministic explanation payload conforming to Schema v1.2 |
| `GET` | `/jobs/{id}/preview` | Serve visual products: `kind={registered\|residual\|checkerboard\|tiepoints\|craters}` |
| `GET` | `/jobs/{id}/graphs` | Serve scientific charts: `kind={residual_scatter\|residual_histogram\|crater_histogram\|confidence_gauge\|sfd}` |
| `GET` | `/jobs/{id}/export` | Export verified tie-point correspondences: `kind=control_points` (CSV) |
| `POST` | `/agent/explain/{id}/start` | Load job summary into AI explainer context |
| `POST` | `/agent/explain/{id}/message` | Ask questions about registration result (<50ms fast-path / generative) |
| `POST` | `/research/session` | Create independent lunar science research session |
| `POST` | `/research/{sid}/message` | Ask planetary science & sensor questions (RAG grounded) |
