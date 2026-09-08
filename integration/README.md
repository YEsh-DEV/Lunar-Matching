# LUNA-MATCH API — Integration Guide for VLM/RAG Developer

This document explains how to integrate with the LUNA-MATCH lunar image registration API. No prior knowledge of this codebase is assumed.

---

## What LUNA-MATCH Does (One Paragraph)

**LUNA-MATCH** is an automated pipeline that takes two lunar orbital images — one from Chandrayaan-2 (OHRC, TMC-2, or IIRS) and one from a reference source like NASA LRO NAC — and precisely aligns them pixel-to-pixel. It handles the core challenges of lunar imagery: extreme lighting differences (hard shadows that move with sun angle), large scale gaps (up to 250× resolution difference), and terrain relief (craters and ridges that shift apparent positions). The output is a **registered GeoTIFF** (the second image warped into the first image's coordinate frame) plus **quantitative metrics**: sub-pixel RMSE (target < 1.5 px), inlier ratio (target > 60%), and spatial distribution index (target > 0.5). It exposes this as a REST API so you can submit jobs, poll status, and retrieve results programmatically.

---

## How to Start the API

```bash
# From the repo root (luna-match/)
cd /path/to/luna-match

# Activate the virtual environment (already set up with all dependencies)
source .venv/bin/activate  # or .venv312/bin/activate

# Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Server runs at:** `http://localhost:8000`  
**Swagger UI (interactive docs):** `http://localhost:8000/docs` — use this to test endpoints visually.

---

## The Three Endpoints You Need

### 1. Submit a Registration Job
**POST `/register`** — Start a new alignment job.

**Request (JSON):**
```json
{
  "img_a_path": "/absolute/path/to/chandrayaan2_image.tif",
  "img_b_path": "/absolute/path/to/reference_image.tif",
  "job_id": "optional-custom-id"  // if omitted, UUID generated
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "job_abc123def",
  "status": "PENDING",
  "elapsed_s": 0.0
}
```

> **Note:** Both images must be accessible at the given absolute paths on the **server filesystem**. The API does not accept file uploads in this version.

---

### 2. Poll Job Status
**GET `/jobs/{job_id}`** — Check progress.

**Response:**
```json
{
  "job_id": "job_abc123def",
  "status": "MATCHING",  // PENDING → PREPROCESSING → MATCHING → VERIFYING → REFINING → DONE / FAILED
  "elapsed_s": 12.4,
  "error": null
}
```

**Polling strategy:** Check every 2–5 seconds until `status` is `DONE` or `FAILED`. Typical jobs take 5–30 seconds depending on image size.

---

### 3. Get Results (Summary + Metrics)
**GET `/jobs/{job_id}/result`** — Retrieve final metrics and output file paths.

**Response (DONE):**
```json
{
  "job_id": "job_abc123def",
  "status": "DONE",
  "rmse_px": 0.99,
  "inlier_ratio": 0.831,
  "sdi": 0.584,
  "n_inliers": 1525,
  "n_total": 1834,
  "elapsed_s": 18.7,
  "transform": "homography",
  "output_files": {
    "registered": "/abs/path/to/data/jobs/job_abc123def/output/registered.tif",
    "residual_map": "/abs/path/to/data/jobs/job_abc123def/output/residual_map.png",
    "metrics": "/abs/path/to/data/jobs/job_abc123def/output/metrics.json"
  }
}
```

**Response (FAILED):**
```json
{
  "job_id": "job_abc123def",
  "status": "FAILED",
  "error": "Step 3 (Dense Matching) failed: insufficient real matches: got 3, need >= 8"
}
```

---

### Bonus: Visual Previews (for debugging/verification)

**GET `/jobs/{job_id}/preview?kind=<type>`** — Returns an image (PNG/TIFF) directly.

| `kind` | Description |
|--------|-------------|
| `registered` | Warped GeoTIFF (default) |
| `residual` | Error heatmap (PNG) |
| `checkerboard` | Alternating tile mosaic showing alignment quality |
| `tiepoints` | Side-by-side match correspondence lines |

Use these to visually verify registration quality in your RAG/VLM pipeline.

---

## Example Real Output (When Available)

Once a job completes successfully, you'll find artifacts in:
```
data/jobs/{job_id}/
├── input/
│   ├── metadata.json          # Combined sensor/GSD/solar metadata
│   └── input_paths.json       # Original paths
├── intermediate/
│   ├── pc_map_a.npy           # Phase congruency map (image A)
│   ├── mim_a.npy              # Maximum index map (image A)
│   ├── matches_raw.npy        # Raw (N,5) correspondences
│   ├── matches_anms.npy       # Spatially filtered matches
│   ├── matches_verified.npy   # MAGSAC++ inliers
│   └── transform_params.json  # Homography/TPS parameters
└── output/
    ├── registered.tif         # Warped reference image (GeoTIFF)
    ├── residual_map.png       # Per-pixel error heatmap
    ├── metrics.json           # Same as /result endpoint
    ├── preview_checkerboard.png
    └── preview_tiepoints.png
```

**For your VLM/RAG:** The key files are `metrics.json` (numbers) and `registered.tif` (aligned image). The `checkerboard` preview is excellent for qualitative "does this look right?" verification.

---

## Quick Test with Existing Sample Data

The repo includes pre-processed sample pairs in `data/samples/`:

```bash
# Test 1: Real lunar pair (~4.2× scale gap)
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"img_a_path": "/abs/path/to/luna-match/data/samples/image_1.tif", "img_b_path": "/abs/path/to/luna-match/data/samples/image_2.tif"}'

# Test 2: Ground-truth verified pair (known 8° rotation, 0.6× scale)
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"img_a_path": "/abs/path/to/luna-match/data/samples/verified_a.tif", "img_b_path": "/abs/path/to/luna-match/data/samples/verified_b.tif"}'
```

Replace `/abs/path/to/luna-match/` with the actual absolute path on your server.

---

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `400 Source image not found` | Path doesn't exist on server | Use absolute path, verify file exists |
| `404 Job ID not found` | Wrong ID or job directory cleaned up | Check `data/jobs/` for existing folders |
| `500 Failed to parse job status` | Corrupted `status.json` | Rare — check job folder permissions |
| Job stuck in `MATCHING` | Large images, slow matcher | Wait longer; typical max 60s |

---

## Contact / Next Steps

- **API version:** 2.0.0
- **Health check:** `GET /health` → returns `{"status": "ok", ...}`
- **CORS:** Enabled for all origins (`*`) — works from browser/frontend
- **Authentication:** None in current version (add API key middleware if needed)

For questions about metric interpretation or pipeline behavior, see `docs/architecture.md` or `docs/Agent1_report.md` in the repo.