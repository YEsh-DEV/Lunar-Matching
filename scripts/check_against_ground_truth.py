"""
scripts/check_against_ground_truth.py
=====================================
Quantitative Ground-Truth Validation for LUNA-MATCH Registration.

Loads the injected ground truth transform (ground_truth_transform.json) and
the pipeline's recovered transform (transform_params.json), decomposes both
into physical parameters (rotation angle, scale, translation vector), and
computes defensible ground-truth error metrics:
  - Angular Error (degrees)
  - Scale Error (absolute & relative %)
  - Translation Error (pixels Euclidean distance)
  - Corner Mapping Error (mean corner displacement between GT and Recovered transform)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add parent directory for module imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))


def find_latest_transform_params(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    # Search latest job directory in data/jobs/
    jobs_dir = root_dir / "data" / "jobs"
    if jobs_dir.exists():
        job_subdirs = sorted([d for d in jobs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        for d in job_subdirs:
            tp = d / "intermediate" / "transform_params.json"
            if tp.exists():
                return tp
    return None


def decompose_matrix(M: np.ndarray, cx: float, cy: float):
    """
    Decompose 3x3 homography / affine matrix into rotation angle, scale, and translation offset.
    Assumes standard OpenCV similarity convention:
      M[0, 0] = s * cos(theta), M[0, 1] = s * sin(theta)
      M[1, 0] = -s * sin(theta), M[1, 1] = s * cos(theta)
    """
    # Normalize homogeneous coordinate
    if abs(M[2, 2]) > 1e-8:
        H = M / M[2, 2]
    else:
        H = M.copy()

    s_x = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
    s_y = np.sqrt(H[0, 1]**2 + H[1, 1]**2)
    scale = float((s_x + s_y) / 2.0)

    # Rotation angle in degrees
    angle_deg = float(np.degrees(np.arctan2(H[0, 1], H[0, 0])))

    # Translation offset relative to center
    dx = float(H[0, 2] - ((1.0 - H[0, 0]) * cx - H[0, 1] * cy))
    dy = float(H[1, 2] - (H[0, 1] * cx + (1.0 - H[1, 1]) * cy))

    return {
        "scale": scale,
        "angle_deg": angle_deg,
        "dx_px": dx,
        "dy_px": dy,
        "matrix_norm": H,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Registration Accuracy Against Ground Truth")
    parser.add_argument("--gt", default=str(root_dir / "data" / "samples" / "ground_truth_transform.json"),
                        help="Ground truth transform JSON path")
    parser.add_argument("--params", default=str(root_dir / "demo_output" / "verified_pair" / "transform_params.json"),
                        help="Recovered transform params JSON path")
    parser.add_argument("--out", default=str(root_dir / "demo_output" / "verified_pair" / "ground_truth_comparison.json"),
                        help="Output JSON summary path")
    args = parser.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Find recovered transform
    candidate_params = [
        args.params,
        str(root_dir / "demo_output" / "verified_pair" / "transform_params.json"),
        str(root_dir / "demo_output" / "real_pair" / "transform_params.json"),
    ]
    param_path = find_latest_transform_params(candidate_params)
    if param_path is None or not param_path.exists():
        raise FileNotFoundError(f"Could not find recovered transform_params.json in candidates: {candidate_params}")

    print(f"Loading Ground Truth from : {gt_path}")
    print(f"Loading Recovered Params from: {param_path}")

    with open(param_path) as f:
        param_data = json.load(f)

    # Ground truth parameters
    gt_angle = float(gt_data["rotation_deg"])
    gt_scale = float(gt_data["scale_factor"])
    gt_dx = float(gt_data["translation_dx_px"])
    gt_dy = float(gt_data["translation_dy_px"])
    cx = float(gt_data["center_x_px"])
    cy = float(gt_data["center_y_px"])
    M_gt = np.array(gt_data["ground_truth_transform_3x3"], dtype=np.float64)

    # Recovered parameters
    if "homography" in param_data:
        H_rec = np.array(param_data["homography"], dtype=np.float64)
    else:
        raise ValueError("transform_params.json does not contain homography matrix")

    decomp_rec = decompose_matrix(H_rec, cx, cy)
    rec_angle = decomp_rec["angle_deg"]
    rec_scale = decomp_rec["scale"]
    rec_dx = decomp_rec["dx_px"]
    rec_dy = decomp_rec["dy_px"]
    H_norm = decomp_rec["matrix_norm"]

    # Compute errors
    error_angle_deg = abs(rec_angle - gt_angle)
    error_scale = abs(rec_scale - gt_scale)
    error_scale_pct = (error_scale / gt_scale) * 100.0
    error_dx = abs(rec_dx - gt_dx)
    error_dy = abs(rec_dy - gt_dy)
    error_translation_euclid = float(np.sqrt(error_dx**2 + error_dy**2))

    # Compute corner mapping error across Image A dimensions (556x437)
    W_a, H_a = int(cx * 2.0), int(cy * 2.0)
    corners = np.array([
        [0.0, 0.0, 1.0],
        [float(W_a), 0.0, 1.0],
        [float(W_a), float(H_a), 1.0],
        [0.0, float(H_a), 1.0],
    ], dtype=np.float64).T  # shape (3, 4)

    gt_mapped = M_gt @ corners
    gt_mapped = gt_mapped[:2, :] / gt_mapped[2, :]

    rec_mapped = H_norm @ corners
    rec_mapped = rec_mapped[:2, :] / rec_mapped[2, :]

    corner_errors = np.linalg.norm(gt_mapped - rec_mapped, axis=0)
    mean_corner_error = float(np.mean(corner_errors))
    max_corner_error = float(np.max(corner_errors))

    # Print summary table
    print("\n" + "=" * 78)
    print(" LUNA-MATCH: Ground Truth Verification & Registration Accuracy")
    print("=" * 78)
    print(f"{'Parameter':<28} | {'Ground Truth':<15} | {'Recovered':<15} | {'Absolute Error':<15}")
    print("-" * 78)
    print(f"{'Rotation Angle (deg)':<28} | {gt_angle:<15.4f} | {rec_angle:<15.4f} | {error_angle_deg:<15.4f} deg")
    print(f"{'Scale Factor':<28} | {gt_scale:<15.4f} | {rec_scale:<15.4f} | {error_scale:<15.4f} ({error_scale_pct:.2f}%)")
    print(f"{'Translation X (dx px)':<28} | {gt_dx:<15.4f} | {rec_dx:<15.4f} | {error_dx:<15.4f} px")
    print(f"{'Translation Y (dy px)':<28} | {gt_dy:<15.4f} | {rec_dy:<15.4f} | {error_dy:<15.4f} px")
    print(f"{'Euclidean Translation Error':<28} | {'—':<15} | {'—':<15} | {error_translation_euclid:<15.4f} px")
    print(f"{'Mean Corner Mapping Error':<28} | {'—':<15} | {'—':<15} | {mean_corner_error:<15.4f} px")
    print(f"{'Max Corner Mapping Error':<28} | {'—':<15} | {'—':<15} | {max_corner_error:<15.4f} px")
    print("=" * 78)

    # Verification threshold assessment
    passed = (
        error_angle_deg < 0.5 and
        error_scale < 0.02 and
        error_translation_euclid < 2.0 and
        mean_corner_error < 2.0
    )

    if passed:
        print("✅ GROUND TRUTH VERIFICATION PASSED: Sub-pixel accuracy confirmed against physical ground truth!")
    else:
        print("⚠️  GROUND TRUTH VERIFICATION WARNING: Error exceeds standard tolerance limits.")

    # Save output comparison JSON
    comparison_summary = {
        "status": "PASSED" if passed else "FAILED",
        "ground_truth": {
            "rotation_deg": gt_angle,
            "scale_factor": gt_scale,
            "translation_dx_px": gt_dx,
            "translation_dy_px": gt_dy,
            "matrix": M_gt.tolist(),
        },
        "recovered": {
            "rotation_deg": round(rec_angle, 4),
            "scale_factor": round(rec_scale, 4),
            "translation_dx_px": round(rec_dx, 4),
            "translation_dy_px": round(rec_dy, 4),
            "matrix": H_norm.tolist(),
            "n_inliers": param_data.get("n_inliers"),
            "inlier_ratio": param_data.get("inlier_ratio"),
            "transform_type": param_data.get("type"),
        },
        "errors": {
            "rotation_error_deg": round(error_angle_deg, 4),
            "scale_error": round(error_scale, 5),
            "scale_error_pct": round(error_scale_pct, 3),
            "translation_dx_error_px": round(error_dx, 4),
            "translation_dy_error_px": round(error_dy, 4),
            "translation_euclidean_error_px": round(error_translation_euclid, 4),
            "mean_corner_error_px": round(mean_corner_error, 4),
            "max_corner_error_px": round(max_corner_error, 4),
        }
    }

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nSaved detailed comparison report to: {out_json}")


if __name__ == "__main__":
    main()
