# LUNA-MATCH — Local Run Guide

## Prerequisites
- Python 3.12
- `git clone https://github.com/YEsh-DEV/Lunar-Matching.git`
- `cd Lunar-Matching`

## Setup (one time)
```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Environment Variables (required for AI chat features)
```bash
export GROQ_API_KEY=your_key_here
export GROQ_MODEL=qwen/qwen3.8-27b
# Get a free key at: console.groq.com
```

## Run the API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Test everything works (in a second terminal)
```bash
python scripts/demo_all_endpoints.py --url http://localhost:8000
```

## Swagger UI (interactive API docs)
Open: `http://localhost:8000/docs`

## All Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | Diagnostic status and health check |
| `POST` | `/register` | Multipart `img_a` + `img_b` or JSON -> `job_id` |
| `GET` | `/jobs/{id}` | Poll status (`PENDING` -> `DONE`) |
| `GET` | `/jobs/{id}/summary` | Full metrics + quality grade |
| `GET` | `/jobs/{id}/preview?kind=...` | Previews: `checkerboard` \| `tiepoints` \| `residual` \| `craters` |
| `GET` | `/jobs/{id}/graphs?kind=...` | Charts: `residual_scatter` \| `residual_histogram` \| `crater_histogram` \| `confidence_gauge` |
| `GET` | `/jobs/{id}/export?kind=control_points` | CSV tie-point download |
| `POST` | `/chat` | `{job_id, message, session_id}` — text chat |
| `POST` | `/analyze?job_id=` | Metric-based analysis |
| `POST` | `/agent/explain/{id}/start` | Load AI context for a job |
| `POST` | `/agent/explain/{id}/message` | `{query}` — fast or generative answer |
| `POST` | `/research/session` | Start general science session |
| `POST` | `/research/{sid}/message` | `{query}` — general lunar science Q&A |
| `GET` | `/chat/{id}/history` | Retrieve session chat history |

## AI Explanation Flow (what to show the judges)
1. Upload two lunar images via `POST /register`
2. Poll `GET /jobs/{id}` until `status=DONE` (~1.2s)
3. Show `GET /jobs/{id}/summary` — Grade, RMSE, inlier ratio
4. Show `GET /jobs/{id}/preview?kind=checkerboard` — visual overlay
5. `POST /agent/explain/{id}/start` — load context
6. Ask metric questions — answered in <1ms (no LLM):
   - `"what is the rmse"`
   - `"how many inliers"`
   - `"is this reliable"`
7. Ask deeper questions — answered by Groq LLM in ~1-3s:
   - `"why did the pipeline choose homography?"`
   - `"is this suitable for crater depth profiling?"`
8. `POST /research/session` + `POST /research/{sid}/message`:
   - `"what is the difference between OHRC and TMC-2?"`

## Key Numbers (verified ground-truth pair)
- **RMSE**: 0.3848 px (sub-pixel, below 0.5px threshold)
- **Inlier ratio**: 84.17% (367/436 verified matches)
- **SSIM**: 0.9535
- **SDI**: 0.9014 (spatially well-distributed)
- **Rotation error vs ground truth**: 0.007°
- **Scale error vs ground truth**: 0.000%
- **Total elapsed**: 0.51s

## Sample Images for Testing
- `data/samples/verified_a.tif` (source image)
- `data/samples/verified_b.tif` (reference image)
- Known ground truth: `rotation=8.0°`, `scale=0.6x`, `dx=15px`, `dy=-10px`

## Run Benchmarks
```bash
python scripts/benchmark_sampledataset.py --mode standard
python scripts/run_verified_stress_demo.py
```
