# LUNA-MATCH Unified Chatbot & VLM/RAG Output Contract
**Schema Version:** `1.0`  
**API Endpoint:** `GET /jobs/{job_id}/summary`  
**Purpose:** Standardized interface contract for external Large Language Models (LLMs), Vision-Language Models (VLMs), and Retrieval-Augmented Generation (RAG) agents querying lunar registration job status, quantitative metrics, plain-language diagnostic assessments, and visual verification artifacts.

---

## 1. Overview & Architecture

When an external consumer or agent (e.g., Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet, or a LangChain/LlamaIndex RAG pipeline) queries the LUNA-MATCH backend, it requires more than raw coordinates:
1. **Mathematical Validation**: Ground-truth geodetic metrics (RMSE, inlier ratio, spatial dispersion).
2. **Plain-Language Synthesis**: Quality classification with human-understandable reasoning for whether the registration is trustworthy or degraded.
3. **Multi-Modal Visual Anchors**: Relative paths to visual verification products (checkerboard mosaic, tie-point correspondence map, residual heatmap, registered GeoTIFF) for visual QA.
4. **Mission Context**: Sensor GSDs, scale disparity ratio, and whether photometric illumination corrections were applied.

The endpoint `GET /jobs/{job_id}/summary` returns a single, unified JSON payload conforming to this contract.

---

## 2. Confidence & Quality Assessment Rubric

The `quality_assessment` block synthesizes multi-dimensional geodetic quality criteria into a standardized confidence level and letter grade:

| Confidence Label | Grade | Geodetic Criteria / Thresholds | Interpretation |
| :--- | :---: | :--- | :--- |
| `high confidence` | **A** | • $\text{RMSE} \le 1.0\text{ px}$<br>• $N_{\text{inliers}} \ge 8$ with $\text{Ratio} \ge 20\%$ (or $N \ge 12$)<br>• $\text{SDI} \ge 0.20$<br>• $\text{Scale Disparity} \le 3.5\times$ | Sub-pixel planetary accuracy; overdetermined geometric system; reliable for scientific crater depth profiling or landing site safety mapping. |
| `moderate confidence`| **B** | • $\text{RMSE} \le 2.0\text{ px}$<br>• $N_{\text{inliers}} \ge 4$ (satisfies 4-DOF minimal homography constraint)<br>• $\text{Ratio} \ge 10\%$ | Mathematically sound planar solution; minor local distortions, sparse inliers, or moderate scale difference. Suitable for reconnaissance. |
| `low confidence — scale disparity may exceed matcher capability` | **C** | • $\text{Scale Disparity} > 3.5\times$ with $N_{\text{inliers}} < 10$ | Resolution gap between sensors exceeds standard scale-space octave invariance; tie points are sparse. |
| `low confidence — high reprojection residual error` | **C** | • $\text{RMSE} > 2.0\text{ px}$ | Reprojection error exceeds planetary tolerances; indicates unmodeled 3D topographic relief (e.g. crater central peak) or residual deformation. |
| `low confidence — sparse surface inliers` | **C** | • $N_{\text{inliers}} < 4$ or $\text{Inlier Ratio} < 10\%$ | Insufficient tie points to constrain an 8-DOF planar homography. |
| `unreliable — synthetic fallback active` | **D** | • `is_synthetic_fallback == True` | System fell back to synthetic matching coordinates; surface metrics are unverified. |
| `failed` | **F** | • `status == "FAILED"` or unhandled exception | Registration aborted at a specific pipeline stage. |

---

## 3. JSON Schema Specification

```typescript
interface ChatbotSummaryResponse {
  schema_version: "1.0";
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
    inlier_ratio: number | null;    // Fraction of verified inliers (0.0 to 1.0)
    sdi: number | null;             // Spatial Dispersion Index (Shannon spatial entropy: 0.0 to 1.0)
    transform_type: string | null;  // "homography" | "thin_plate_spline" | "affine"
    n_inliers: number;              // Count of geometrically verified tie-point inliers
    n_total: number;                // Total candidate matches prior to outlier rejection
    elapsed_s: number;              // Pipeline runtime in seconds
  };

  input_metadata: {
    img_a_path: string | null;      // Relative path to source/moving image
    img_b_path: string | null;      // Relative path to reference image
    gsd_a_m_per_px: number;         // Ground Sampling Distance of Image A in meters/pixel
    gsd_b_m_per_px: number;         // Ground Sampling Distance of Image B in meters/pixel
    scale_disparity_ratio: number;  // max(GSD_A, GSD_B) / min(GSD_A, GSD_B)
    solar_correction_applied: boolean; // True if Lommel-Seeliger photometric normalization ran
  };

  artifacts: {
    registered_geotiff: string | null; // Relative path to output GeoTIFF/raster
    checkerboard_png: string | null;   // Relative path to checkerboard mosaic overlay
    tiepoints_png: string | null;      // Relative path to side-by-side tie-points match overlay
    residual_map_png: string | null;   // Relative path to 2D residual error magnitude heatmap
  };
}
```

---

## 4. Real-World Populated Example

The following example is the actual output produced by `GET /jobs/bench_pair_2/summary` running on real ISRO lunar test imagery (`sou2.jpeg` vs `res2.jpeg`):

```json
{
  "schema_version": "1.0",
  "job_id": "bench_pair_2",
  "status": "DONE",
  "quality_assessment": {
    "confidence_label": "moderate confidence",
    "grade": "B",
    "reasoning": "Registration succeeded with RMSE 0.0003 px (25.0% inliers). Confidence is moderate due to sparse inlier count (4 inliers), but geometric solution is mathematically sound.",
    "warnings": [
      "Solar ephemeris angles missing in metadata; Lommel-Seeliger illumination normalization was bypassed.",
      "Marginal inlier count (4 verified inliers); geometric model has minimal degrees-of-freedom redundancy."
    ]
  },
  "metrics": {
    "rmse_px": 0.0003,
    "inlier_ratio": 0.25,
    "sdi": 0.3333,
    "transform_type": "homography",
    "n_inliers": 4,
    "n_total": 16,
    "elapsed_s": 10.76
  },
  "input_metadata": {
    "img_a_path": "sampledataset/sou2.jpeg",
    "img_b_path": "sampledataset/res2.jpeg",
    "gsd_a_m_per_px": 1.0,
    "gsd_b_m_per_px": 1.0,
    "scale_disparity_ratio": 1.0,
    "solar_correction_applied": false
  },
  "artifacts": {
    "registered_geotiff": "data/jobs/bench_pair_2/output/registered.tif",
    "checkerboard_png": "data/jobs/bench_pair_2/output/preview_checkerboard.png",
    "tiepoints_png": "data/jobs/bench_pair_2/output/preview_tiepoints.png",
    "residual_map_png": "data/jobs/bench_pair_2/output/residual_map.png"
  }
}
```

Another example from `bench_pair_9` (`image.png` vs `image copy.png`):

```json
{
  "schema_version": "1.0",
  "job_id": "bench_pair_9",
  "status": "DONE",
  "quality_assessment": {
    "confidence_label": "moderate confidence",
    "grade": "B",
    "reasoning": "Registration succeeded with RMSE 0.3435 px (100.0% inliers). Confidence is moderate due to moderate spatial clustering (SDI 0.1591), but geometric solution is mathematically sound.",
    "warnings": [
      "Solar ephemeris angles missing in metadata; Lommel-Seeliger illumination normalization was bypassed.",
      "Low spatial dispersion index (SDI=0.1591); keypoints are clustered in a localized subregion rather than distributed across the scene."
    ]
  },
  "metrics": {
    "rmse_px": 0.3435,
    "inlier_ratio": 1.0,
    "sdi": 0.1591,
    "transform_type": "homography",
    "n_inliers": 8,
    "n_total": 8,
    "elapsed_s": 0.93
  },
  "input_metadata": {
    "img_a_path": "sampledataset/image.png",
    "img_b_path": "sampledataset/image copy.png",
    "gsd_a_m_per_px": 1.0,
    "gsd_b_m_per_px": 1.0,
    "scale_disparity_ratio": 1.0,
    "solar_correction_applied": false
  },
  "artifacts": {
    "registered_geotiff": "data/jobs/bench_pair_9/output/registered.tif",
    "checkerboard_png": "data/jobs/bench_pair_9/output/preview_checkerboard.png",
    "tiepoints_png": "data/jobs/bench_pair_9/output/preview_tiepoints.png",
    "residual_map_png": "data/jobs/bench_pair_9/output/residual_map.png"
  }
}
```

---

## 5. Integration Guide for LLM/VLM Developers

### How to Query via Python Requests
```python
import requests

response = requests.get("http://localhost:8000/jobs/bench_pair_2/summary")
if response.status_code == 200:
    summary = response.json()
    confidence = summary["quality_assessment"]["confidence_label"]
    rmse = summary["metrics"]["rmse_px"]
    checkerboard_path = summary["artifacts"]["checkerboard_png"]
    print(f"Status: {confidence} | Reprojection Error: {rmse} px")
```

### Feeding into a Multimodal Prompt (e.g. Gemini 1.5 / GPT-4o)
1. Pass the `summary` JSON as system context.
2. Attach `summary["artifacts"]["checkerboard_png"]` or `tiepoints_png` as visual image inputs.
3. Prompt the model:
   > *"Review the registration between Lunar Frame A and Reference Frame B. The mathematical engine evaluated this as '{summary['quality_assessment']['confidence_label']}' with an RMSE of {summary['metrics']['rmse_px']} pixels. Based on the attached checkerboard mosaic, confirm whether the crater rims align seamlessly across tile boundaries."*
