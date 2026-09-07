"""
scripts/make_verified_test_pair.py
==================================
Build Ground-Truth-Verified Lunar Test Pair for LUNA-MATCH.

Takes real lunar image (image_1.png), masks annotation overlays, and generates
a mathematically calibrated Moving/Sensed Image (Image B) using a known,
recorded rigid/affine transformation and illumination shift:
  - Rotation: 8.0 degrees
  - Scale: 0.60x (GSD_B = GSD_A / 0.6)
  - Translation: dx = +15.0 px, dy = -10.0 px
  - Sun-angle illumination shift: I_B = clip(I_A * 1.25 + 10 + N(0, 2), 0, 255)

Exports:
  - data/samples/verified_a.tif (GSD = 131.6 m/px)
  - data/samples/verified_b.tif (GSD = 219.33 m/px)
  - data/samples/ground_truth_transform.json
  - data/samples/sanity_check_verified_pair.png
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

# Add parent directory for module imports
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from scripts.prepare_real_samples import find_source_image, mask_and_fill


import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate Ground-Truth-Verified Lunar Test Pair")
    parser.add_argument("--rot", type=float, default=8.0, help="Rotation angle in degrees")
    parser.add_argument("--scale", type=float, default=0.60, help="Scale factor")
    parser.add_argument("--dx", type=float, default=15.0, help="Translation dx in pixels")
    parser.add_argument("--dy", type=float, default=-10.0, help="Translation dy in pixels")
    parser.add_argument("--gain", type=float, default=1.25, help="Photometric gain multiplier")
    parser.add_argument("--bias", type=float, default=10.0, help="Photometric bias offset")
    parser.add_argument("--prefix", type=str, default="verified", help="Output filename prefix")
    args = parser.parse_args()

    workspace_root = root_dir.parent

    # Candidate paths for base lunar image 1
    p1_candidates = [
        str(workspace_root / "image.png"),
        str(workspace_root / "image_1.png"),
        str(root_dir / "data" / "samples" / "image_1.png"),
    ]

    p1 = find_source_image(p1_candidates)
    print(f"Loading base lunar image from: {p1}")

    raw_1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    if raw_1 is None:
        raise ValueError(f"Failed to read source image from {p1}")

    # Verified annotation overlay mask boxes for Image 1 (556x437):
    #   Scale bar at bottom-left: y:[400, 437], x:[0, 105]
    #   North arrow at top-right: y:[0, 75], x:[505, 556]
    box_1_bl = (0, 105, 400, 437)
    box_1_tr = (505, 556, 0, 75)
    masked_a = mask_and_fill(raw_1, box_1_bl, box_1_tr)

    H, W = masked_a.shape
    print(f"Base image dimensions: {W}x{H} (WxH)")

    # -----------------------------------------------------------------------
    # Ground Truth Transformation Parameters
    # -----------------------------------------------------------------------
    rotation_deg = args.rot
    scale_factor = args.scale
    dx = args.dx
    dy = args.dy
    cx = W / 2.0
    cy = H / 2.0

    # OpenCV rotation + scale matrix around center
    M_rot = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale_factor)
    M_gt = M_rot.copy()
    M_gt[0, 2] += dx
    M_gt[1, 2] += dy

    # 3x3 Homogeneous representation
    M_3x3 = np.eye(3, dtype=np.float64)
    M_3x3[:2, :] = M_gt

    # Warp image A to create image B
    warped_b = cv2.warpAffine(
        masked_a,
        M_gt,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Apply sun-angle / photometric disparity + subtle sensor noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.5, warped_b.shape)
    gain = args.gain
    bias = args.bias
    masked_b = np.clip(warped_b.astype(np.float32) * gain + bias + noise, 0, 255).astype(np.uint8)

    # -----------------------------------------------------------------------
    # Export Artifacts
    # -----------------------------------------------------------------------
    out_dir = root_dir / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    gt_filename = "ground_truth_transform.json" if prefix == "verified" else f"ground_truth_{prefix}.json"
    gt_info = {
        "dataset": "LUNA-MATCH Verified Ground-Truth Pair",
        "base_image": Path(p1).name,
        "rotation_deg": rotation_deg,
        "scale_factor": scale_factor,
        "translation_dx_px": dx,
        "translation_dy_px": dy,
        "center_x_px": cx,
        "center_y_px": cy,
        "photometric_gain": gain,
        "photometric_bias": bias,
        "gsd_a_m_per_px": 131.6,
        "gsd_b_m_per_px": float(round(131.6 / scale_factor, 4)),
        "ground_truth_transform_3x3": M_3x3.tolist(),
        "affine_2x3": M_gt.tolist(),
    }

    gt_json_path = out_dir / gt_filename
    with open(gt_json_path, "w") as f:
        json.dump(gt_info, f, indent=2)
    print(f"Saved ground truth parameters to: {gt_json_path}")

    # 2. Export GeoTIFFs
    # Image A
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

    # Image B: GSD_B = GSD_A / scale_factor
    gsd_b = float(round(gsd_a / scale_factor, 4))
    tif_path_b = out_dir / f"{prefix}_b.tif"
    norm_b = (masked_b.astype(np.float32) / 255.0)
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
        dst.write(norm_b, 1)
    print(f"Exported Image B GeoTIFF: {tif_path_b} (GSD={gsd_b} m/px)")

    # 3. Visual sanity check side-by-side
    canvas = np.hstack([masked_a, masked_b])
    cv2.putText(canvas, f"{prefix} Image A (Base)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)
    cv2.putText(canvas, f"{prefix} Image B (rot={rotation_deg}deg, s={scale_factor})", (W + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,), 2)

    check_path = out_dir / f"sanity_check_{prefix}_pair.png"
    cv2.imwrite(str(check_path), canvas)
    print(f"Saved sanity check visualization to: {check_path}")
    print("\nVerified test pair generation completed successfully!")


if __name__ == "__main__":
    main()
