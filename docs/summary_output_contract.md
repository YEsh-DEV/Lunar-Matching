# LUNA-MATCH Summary Output & Visualization Contract
**Schema Version:** `1.2`  
**API Endpoints:**
- `GET /jobs/{job_id}/summary` : Unified geodetic summary, metrics, and artifact references
- `GET /jobs/{job_id}/preview` : Raster and derived overlay previews
- `GET /jobs/{job_id}/graphs`  : Scientific charts and quality assessment plots
- `GET /jobs/{job_id}/export`  : Verified ground control points CSV export

---

## 1. Overview & Architecture

The LUNA-MATCH reporting engine provides standardized programmatic access to registration results:
1. **Mathematical Validation**: Ground-truth geodetic metrics (RMSE, inlier ratio, spatial dispersion).
2. **Plain-Language Synthesis**: Quality classification with deterministic rationale for whether the registration is trustworthy or degraded.
3. **Multi-Modal Visual Anchors**: Relative paths to visual verification products (checkerboard mosaic, tie-point correspondence map, residual heatmap, registered GeoTIFF) for visual QA.
4. **Scientific Charts**: High-resolution PNG graphs providing statistical and structural error distributions.
5. **Mission Context**: Sensor GSDs, scale disparity ratio, fallback pixel dimension ratio, and photometric illumination flags.

The endpoint `GET /jobs/{job_id}/summary` returns a single, unified JSON payload conforming to this contract.

---

## 2. Confidence & Quality Assessment Rubric

The `quality_assessment` block synthesizes multi-dimensional geodetic quality criteria into a standardized confidence level and letter grade:

| Confidence Label | Grade | Geodetic Criteria / Thresholds | Interpretation |
| :--- | :---: | :--- | :--- |
| `high confidence` | **A** | • $\text{RMSE} \le 1.0\text{ px}$<br>• $N_{\text{inliers}} \ge 8$ with $\text{Ratio} \ge 20\%$ (or $N \ge 12$)<br>• $\text{SDI} \ge 0.20$<br>• $\text{Effective Scale Disparity} \le 3.5\times$ | Sub-pixel planetary accuracy; overdetermined geometric system; reliable for scientific crater depth profiling or landing site safety mapping. |
| `moderate confidence`| **B** | • $\text{RMSE} \le 2.0\text{ px}$<br>• $N_{\text{inliers}} \ge 4$ (satisfies 4-DOF minimal homography constraint)<br>• $\text{Ratio} \ge 10\%$<br>• $\text{Effective Scale Disparity} \le 3.5\times$ | Mathematically sound planar solution; minor local distortions, sparse inliers, or moderate scale difference. Suitable for reconnaissance. |
| `low confidence — scale disparity may exceed matcher capability` | **C** | • $\text{Effective Scale Disparity} > 3.5\times$ with $N_{\text{inliers}} < 10$ | Resolution gap between sensors exceeds standard scale-space octave invariance; tie points are sparse. |
| `low confidence — high reprojection residual error` | **C** | • $\text{RMSE} > 2.0\text{ px}$ | Reprojection error exceeds planetary tolerances; indicates unmodeled 3D topographic relief (e.g. crater central peak) or residual deformation. |
| `low confidence — sparse surface inliers` | **C** | • $N_{\text{inliers}} < 4$ or $\text{Inlier Ratio} < 10\%$ | Insufficient tie points to constrain an 8-DOF planar homography. |
| `unreliable — synthetic fallback active` | **D** | • `is_synthetic_fallback == True` | System fell back to synthetic matching coordinates; surface metrics are unverified. |
| `failed` | **F** | • `status == "FAILED"` or unhandled exception | Registration aborted at a specific pipeline stage. |

---

## 3. JSON Schema Specification (v1.2)

```typescript
interface JobSummaryResponse {
  schema_version: "1.2";
  job_id: string;
  status: "PENDING" | "PREPROCESSING" | "MATCHING" | "VERIFYING" | "REFINING" | "DONE" | "FAILED";
  
  quality_assessment: {
    confidence_label: string; // "high confidence" | "moderate confidence" | "low confidence ..." | "failed"
    grade: "A" | "B" | "C" | "D" | "F";
    reasoning: string;        // Human-readable rationale for assigned grade
    warnings: string[];       // Non-fatal geodetic caveats or missing metadata notices
  };

  metrics: {
    rmse_px: number | null;         // Root-mean-square reprojection error in pixels
    mae_px: number | null;          // Mean Absolute Error in pixels
    ssim: number | null;            // Structural Similarity Index (-1.0 to 1.0)
    ncc: number | null;             // Normalized Cross-Correlation (-1.0 to 1.0)
    inlier_ratio: number | null;    // Fraction of verified inliers (0.0 to 1.0)
    sdi: number | null;             // Spatial Dispersion Index (Shannon spatial entropy: 0.0 to 1.0)
    transform_type: string | null;  // "homography" | "thin_plate_spline" | "affine"
    condition_number: number | null;// Condition number (kappa) of the fitted transformation matrix
    n_inliers: number;              // Count of geometrically verified tie-point inliers
    n_total: number;                // Total candidate matches prior to outlier rejection
    elapsed_s: number;              // Pipeline runtime in seconds
  };

  input_metadata: {
    img_a_path: string | null;        // Relative path to source/moving image
    img_b_path: string | null;        // Relative path to reference image
    gsd_a_m_per_px: number;           // Ground Sampling Distance of Image A in meters/pixel (1.0 if uncalibrated)
    gsd_b_m_per_px: number;           // Ground Sampling Distance of Image B in meters/pixel (1.0 if uncalibrated)
    scale_disparity_ratio: number;    // Physical GSD disparity ratio: max(GSD_A, GSD_B) / min(GSD_A, GSD_B)
    pixel_dimension_ratio: number;    // Dimension ratio fallback: max(dim_A, dim_B) / min(dim_A, dim_B)
    solar_correction_applied: boolean;// True if Lommel-Seeliger photometric normalization ran
  };

  artifacts: {
    registered_geotiff: string | null;       // Relative path to output GeoTIFF/raster
    checkerboard_png: string | null;         // Relative path to checkerboard mosaic overlay
    tiepoints_png: string | null;            // Relative path to side-by-side tie-points match overlay
    residual_map_png: string | null;         // Relative path to 2D residual error magnitude heatmap
    craters_png?: string | null;             // Relative path to detected craters overlay
    graph_residual_scatter: string | null;   // Relative path to reprojection residual scatter plot
    graph_residual_histogram: string | null; // Relative path to residual error magnitude histogram
    graph_crater_histogram: string | null;   // Relative path to crater diameter class bar chart
    graph_confidence_gauge: string | null;   // Relative path to registration quality confidence gauge
  };
}
```

---

## 4. Endpoints & Visualization Products

### 4.1 Preview Visualizations (`GET /jobs/{job_id}/preview`)
Query parameter: `kind={registered|residual|checkerboard|tiepoints|craters}`
- `registered`: Warped moving image aligned to reference georeference frame.
- `residual`: 2D continuous residual displacement error heatmap.
- `checkerboard`: Alternating tile mosaic showing seamless boundary alignment.
- `tiepoints`: Side-by-side keypoint correspondence overlay with tie lines.
- `craters`: Detected crater rims and centroids overlay.

### 4.2 Scientific Graph Visualizations (`GET /jobs/{job_id}/graphs`)
Query parameter: `kind={residual_scatter|residual_histogram|crater_histogram|confidence_gauge}`
- `residual_scatter`: Scatter plot of reprojection residuals across tie-points with mean error indicator.
- `residual_histogram`: 20-bin histogram of per-point residual error magnitudes with 0.5px sub-pixel threshold line.
- `crater_histogram`: Bar chart categorizing crater diameters by standard planetary size classes (<1km, 1-3km, 3-10km, >10km).
- `confidence_gauge`: Colored indicator bar displaying assigned quality grade (A/B/C/D/F) and descriptive confidence label.

---

## 5. Real-World Populated Example

### Example: Grade A (High Confidence) — Sample 7 (`sou7.jpeg` vs `res7.jpeg`)
```json
{
  "schema_version": "1.2",
  "job_id": "bench_pair_7",
  "status": "DONE",
  "quality_assessment": {
    "confidence_label": "high confidence",
    "grade": "A",
    "reasoning": "Sub-pixel reprojection accuracy (RMSE 1.8464 px) with strong inlier verification (22 inliers, 81.5%) and solid spatial distribution (SDI 0.5587) under a verified homography transformation.",
    "warnings": [
      "Solar ephemeris angles missing in metadata; Lommel-Seeliger illumination normalization was bypassed."
    ]
  },
  "metrics": {
    "rmse_px": 1.8464,
    "mae_px": 1.5780,
    "ssim": 0.5493,
    "ncc": 0.5493,
    "inlier_ratio": 0.8148,
    "sdi": 0.5587,
    "transform_type": "homography",
    "condition_number": 20.42,
    "n_inliers": 22,
    "n_total": 27,
    "elapsed_s": 0.12
  },
  "input_metadata": {
    "img_a_path": "sampledataset/sou7.jpeg",
    "img_b_path": "sampledataset/res7.jpeg",
    "gsd_a_m_per_px": 1.0,
    "gsd_b_m_per_px": 1.0,
    "scale_disparity_ratio": 1.0,
    "pixel_dimension_ratio": 1.0,
    "solar_correction_applied": false
  },
  "artifacts": {
    "registered_geotiff": "data/jobs/bench_pair_7/output/registered.tif",
    "checkerboard_png": "data/jobs/bench_pair_7/output/preview_checkerboard.png",
    "tiepoints_png": "data/jobs/bench_pair_7/output/preview_tiepoints.png",
    "residual_map_png": "data/jobs/bench_pair_7/output/residual_map.png",
    "graph_residual_scatter": "data/jobs/bench_pair_7/output/graph_residual_scatter.png",
    "graph_residual_histogram": "data/jobs/bench_pair_7/output/graph_residual_histogram.png",
    "graph_crater_histogram": "data/jobs/bench_pair_7/output/graph_crater_histogram.png",
    "graph_confidence_gauge": "data/jobs/bench_pair_7/output/graph_confidence_gauge.png"
  }
}
```
