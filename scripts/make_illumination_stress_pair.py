"""
scripts/make_illumination_stress_pair.py
=========================================
Build Harsh Illumination-Stress Lunar Test Pair for LUNA-MATCH.

Generates a mathematically calibrated Moving/Sensed Image (Image B) with:
  - Exact Ground Truth Geometry:
      * Rotation: 8.0 degrees
      * Scale: 0.60x (GSD_B = GSD_A / 0.60 = 219.33 m/px)
      * Translation: dx = +15.0 px, dy = -10.0 px
  - Harsh Non-Linear Illumination & Shadow Stress:
      * Strong non-linear gamma curve (gamma = 0.35) simulating low-elevation solar incidence
      * Local contrast inversion in a crater floor / wall subregion simulating shifted cast shadow boundary
      * Linear illumination ramp gradient across the image (changing solar azimuth)
      * Gaussian sensor noise N(0, 1.5)

Exports:
  - data/samples/stress_a.tif
  - data/samples/stress_b.tif
  - data/samples/ground_truth_stress.json
  - data/samples/sanity_check_stress_pair.png
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from scripts.prepare_real_samples import find_source_image, mask_and_fill


def main():
    parser = argparse.ArgumentParser(description="Generate Illumination Stress Test Pair")
    parser.add_argument("--rot", type=float, default=8.0, help="Rotation angle in degrees")
    parser.add_argument("--scale", type=float, default=0.60, help="Scale factor")
    parser.add_argument("--dx", type=float, default=15.0, help="Translation dx in pixels")
    parser.add_argument("--dy", type=float, default=-10.0, help="Translation dy in pixels")
    parser.add_argument("--gamma", type=float, default=0.35, help="Nonlinear tone gamma exponent")
    parser.add_argument("--prefix", type=str, default="stress", help="Output filename prefix")
    args = parser.parse_args()

    workspace_root = root_dir.parent
    p1_candidates = [
        str(root_dir / "image.png"),
        str(workspace_root / "image.png"),
        str(workspace_root / "image_1.png"),
        str(root_dir / "data" / "samples" / "image_1.png"),
    ]

    p1 = find_source_image(p1_candidates)
    print(f"Loading base lunar image from: {p1}")

    raw_1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    if raw_1 is None:
        raise ValueError(f"Failed to read source image from {p1}")

    box_1_bl = (0, 105, 400, 437)
    box_1_tr = (505, 556, 0, 75)
    masked_a = mask_and_fill(raw_1, box_1_bl, box_1_tr)

    H, W = masked_a.shape
    print(f"Base image dimensions: {W}x{H} (WxH)")

    rotation_deg = args.rot
    scale_factor = args.scale
    dx = args.dx
    dy = args.dy
    cx = W / 2.0
    cy = H / 2.0

    # OpenCV affine transform matrix
    M_rot = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale_factor)
    M_gt = M_rot.copy()
    M_gt[0, 2] += dx
    M_gt[1, 2] += dy

    M_3x3 = np.eye(3, dtype=np.float64)
    M_3x3[:2, :] = M_gt

    # Warp image A geometrically
    warped_b = cv2.warpAffine(
        masked_a,
        M_gt,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # -----------------------------------------------------------------------
    # Apply Harsh Illumination / Shadow Stress Transform
    # -----------------------------------------------------------------------
    norm_b = warped_b.astype(np.float32) / 255.0

    # 1. Non-linear power-law gamma compression (gamma = 0.35)
    gamma = args.gamma
    gamma_b = np.power(np.maximum(norm_b, 1e-6), gamma)

    # 2. Local contrast inversion in prominent crater subregion
    # Simulates reversed sun-angle / hard shadow boundary sweeping across crater
    Y, X = np.ogrid[:H, :W]
    crater_cx, crater_cy = 200, 185
    dist = np.sqrt((X - crater_cx)**2 + (Y - crater_cy)**2)
    mask_crater = np.clip(1.0 - (dist - 40.0) / 35.0, 0.0, 1.0)
    inverted_b = 1.0 - gamma_b
    blended_b = gamma_b * (1.0 - mask_crater) + inverted_b * mask_crater

    # 3. Directional illumination ramp gradient across the image
    ramp = 0.55 + 0.65 * (X / float(W))
    stressed_float = blended_b * ramp

    # 4. Sensor noise
    rng = np.random.default_rng(1337)
    noise = rng.normal(0.0, 1.5, (H, W))
    masked_b = np.clip(stressed_float * 255.0 + noise, 0, 255).astype(np.uint8)

    # -----------------------------------------------------------------------
    # Export Outputs
    # -----------------------------------------------------------------------
    out_dir = root_dir / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    gt_info = {
        "dataset": "LUNA-MATCH Illumination-Stress Ground-Truth Pair",
        "base_image": Path(p1).name,
        "rotation_deg": rotation_deg,
        "scale_factor": scale_factor,
        "translation_dx_px": dx,
        "translation_dy_px": dy,
        "center_x_px": cx,
        "center_y_px": cy,
        "gamma": gamma,
        "local_contrast_inversion": {
            "center_x": crater_cx,
            "center_y": crater_cy,
            "radius": 75,
            "description": "Simulated shifted hard shadow boundary with local contrast inversion"
        },
        "gsd_a_m_per_px": 131.6,
        "gsd_b_m_per_px": float(round(131.6 / scale_factor, 4)),
        "ground_truth_transform_3x3": M_3x3.tolist(),
        "affine_2x3": M_gt.tolist(),
    }

    gt_json_path = out_dir / f"ground_truth_{prefix}.json"
    with open(gt_json_path, "w") as f:
        json.dump(gt_info, f, indent=2)
    print(f"Saved ground truth parameters to: {gt_json_path}")

    # Export GeoTIFFs
    gsd_a = 131.6
    tif_path_a = out_dir / f"{prefix}_a.tif"
    norm_a = (masked_a.astype(np.float32) / 255.0)
    transform_a = from_origin(0.0, 0.0, float(gsd_a), float(gsd_a))

    with rasterio.open(
        str(tif_path_a),
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform_a,
    ) as dst:
        dst.write(norm_a, 1)
    print(f"Exported Image A GeoTIFF: {tif_path_a} (GSD={gsd_a} m/px)")

    gsd_b = float(round(gsd_a / scale_factor, 4))
    tif_path_b = out_dir / f"{prefix}_b.tif"
    norm_out_b = (masked_b.astype(np.float32) / 255.0)
    transform_b = from_origin(0.0, 0.0, float(gsd_b), float(gsd_b))

    with rasterio.open(
        str(tif_path_b),
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform_b,
    ) as dst:
        dst.write(norm_out_b, 1)
    print(f"Exported Image B GeoTIFF: {tif_path_b} (GSD={gsd_b} m/px)")

    # Sanity check visualization
    canvas = np.hstack([masked_a, masked_b])
    cv2.putText(canvas, f"{prefix} A (Base)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)
    cv2.putText(canvas, f"{prefix} B (gamma={gamma}, shadow-inverted subregion)", (W + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,), 2)

    check_path = out_dir / f"sanity_check_{prefix}_pair.png"
    cv2.imwrite(str(check_path), canvas)
    print(f"Saved sanity check visualization to: {check_path}")
    print("\nIllumination stress test pair generation completed successfully!")


if __name__ == "__main__":
    main()
