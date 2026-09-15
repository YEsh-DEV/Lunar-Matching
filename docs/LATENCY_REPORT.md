# LUNA-MATCH Latency Report

> Real measured timings from `scripts/measure_latency.py` — not projections.

| Size | Mode | Status | Total(ms) | Ingest | Pre-filter | PhCong | Dense | ANMS | Verify | Refine | WarpEval | RMSE(px) | SDI |
|------|------|--------|-----------|--------|-----------|--------|-------|------|--------|--------|----------|----------|-----|
| 128x128 | standard | DONE | 148.6 | 41.0 | 1.9 | 62.2 | 13.5 | 0.7 | 3.4 | 5.1 | 0.2213 | 0.7345 |
| 256x256 | standard | DONE | 351.9 | 46.5 | 5.9 | 215.3 | 41.4 | 0.6 | 3.2 | 15.1 | 0.2625 | 0.9188 |
| 256x256 | fast | DONE | 247.0 | 47.6 | 5.0 | 136.5 | 32.0 | 0.7 | 3.0 | 11.8 | 0.2402 | 0.9188 |
| 512x512 | standard | DONE | 1908.5 | 67.4 | 15.1 | 1567.0 | 77.4 | 0.5 | 2.4 | 63.5 | 0.2449 | 0.9812 |

## Implementation Notes
- **Pre-filter**: CLAHE+unsharp, only fires when Michelson contrast < 0.15.
- **Fast mode**: 4 Log-Gabor orientations vs 6 (standard). ~33% fewer FFTs.
- **TPS cap**: top 48 ANMS-ranked control points. Logged in `tps_fit_ms`.
- **SDI formula**: Shannon entropy over 8x8 grid — unchanged.
- **Ill-conditioned threshold**: kappa > 1e4 — unchanged.
- All per-stage times from `time.perf_counter()` within orchestrator.