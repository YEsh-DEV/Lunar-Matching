"""
scripts/prepare_real_samples.py
===============================
Prepare Real Lunar Sample Datasets for the LUNA-MATCH Pipeline.

Loads raw lunar screenshots (image_1.png and image_2.png), masks out annotation
overlays (scale bar, text, north arrow) in-place using nearby local background
mean fill, and exports single-band GeoTIFFs with exact GSDs:
  - image_1.tif: GSD = 131.6 m/px
  - image_2.tif: GSD = 555.6 m/px
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

def find_source_image(candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"None of candidate paths found: {candidates}")

def mask_and_fill(img: np.ndarray, bl_box: tuple, tr_box: tuple) -> np.ndarray:
    """
    Mask bottom-left and top-right annotation overlays in-place.
    Fill with nearby local terrain mean using Gaussian feathering to avoid
    creating fake high-gradient step edges that attract feature matchers.
    """
    H, W = img.shape[:2]
    img_f = img.astype(np.float32)
    fill_canvas = img_f.copy()
    mask = np.zeros((H, W), dtype=np.float32)

    # 1. Bottom-Left Box: (x_min, x_max, y_min, y_max)
    bl_x0, bl_x1, bl_y0, bl_y1 = bl_box
    # Sample unmasked terrain immediately above the scale bar
    sample_y0 = max(0, bl_y0 - 45)
    sample_y1 = max(0, bl_y0 - 5)
    sample_x0 = bl_x0
    sample_x1 = min(W, bl_x1)
    bl_sample = img[sample_y0:sample_y1, sample_x0:sample_x1]
    bl_mean = float(np.mean(bl_sample)) if bl_sample.size > 0 else float(np.mean(img))

    fill_canvas[bl_y0:bl_y1, bl_x0:bl_x1] = bl_mean
    mask[bl_y0:bl_y1, bl_x0:bl_x1] = 1.0

    # 2. Top-Right Box: (x_min, x_max, y_min, y_max)
    tr_x0, tr_x1, tr_y0, tr_y1 = tr_box
    # Sample unmasked terrain immediately adjacent to the left of north arrow
    sample_y0_tr = tr_y0
    sample_y1_tr = min(H, tr_y1)
    sample_x0_tr = max(0, tr_x0 - 50)
    sample_x1_tr = max(0, tr_x0 - 5)
    tr_sample = img[sample_y0_tr:sample_y1_tr, sample_x0_tr:sample_x1_tr]
    tr_mean = float(np.mean(tr_sample)) if tr_sample.size > 0 else float(np.mean(img))

    fill_canvas[tr_y0:tr_y1, tr_x0:tr_x1] = tr_mean
    mask[tr_y0:tr_y1, tr_x0:tr_x1] = 1.0

    # Gaussian feather the mask boundary to avoid hard step-edges
    feathered_mask = cv2.GaussianBlur(mask, (9, 9), 3.0)

    # Seamless alpha blend
    blended = img_f * (1.0 - feathered_mask) + fill_canvas * feathered_mask
    return np.clip(blended, 0, 255).astype(np.uint8)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    workspace_root = root_dir.parent

    # Candidate source paths
    p1_candidates = [
        str(workspace_root / "image.png"),
        str(workspace_root / "image_1.png"),
        str(root_dir / "data" / "samples" / "image_1.png"),
    ]
    p2_candidates = [
        str(workspace_root / "image copy.png"),
        str(workspace_root / "image_2.png"),
        str(root_dir / "data" / "samples" / "image_2.png"),
    ]

    p1 = find_source_image(p1_candidates)
    p2 = find_source_image(p2_candidates)

    print(f"Loading Image 1 from: {p1}")
    print(f"Loading Image 2 from: {p2}")

    raw_1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    raw_2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    if raw_1 is None:
        raise ValueError(f"Failed to read image 1 from {p1}")
    if raw_2 is None:
        raise ValueError(f"Failed to read image 2 from {p2}")

    print(f"Image 1 dimensions: {raw_1.shape[1]}x{raw_1.shape[0]} (WxH)")
    print(f"Image 2 dimensions: {raw_2.shape[1]}x{raw_2.shape[0]} (WxH)")

    # Define verified mask boxes (x_min, x_max, y_min, y_max)
    # Image 1 (556x437):
    #   Scale bar at bottom left: y:[400, 437], x:[0, 105]
    #   North arrow at top right: y:[0, 75], x:[505, 556]
    box_1_bl = (0, 105, 400, 437)
    box_1_tr = (505, 556, 0, 75)

    # Image 2 (555x442):
    #   Scale bar at bottom left: y:[396, 442], x:[0, 110]
    #   North arrow at top right: y:[0, 60], x:[485, 555]
    box_2_bl = (0, 110, 396, 442)
    box_2_tr = (485, 555, 0, 60)

    masked_1 = mask_and_fill(raw_1, box_1_bl, box_1_tr)
    masked_2 = mask_and_fill(raw_2, box_2_bl, box_2_tr)

    # Output directory
    out_dir = root_dir / "data" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save visual sanity check side-by-side
    canvas_1 = np.hstack([raw_1, masked_1])
    canvas_2 = np.hstack([raw_2, masked_2])

    cv2.putText(canvas_1, "Original 1", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)
    cv2.putText(canvas_1, "Masked 1 (No Overlays)", (raw_1.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)
    cv2.putText(canvas_2, "Original 2", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)
    cv2.putText(canvas_2, "Masked 2 (No Overlays)", (raw_2.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)

    check_path_1 = out_dir / "sanity_check_image_1.png"
    check_path_2 = out_dir / "sanity_check_image_2.png"
    cv2.imwrite(str(check_path_1), canvas_1)
    cv2.imwrite(str(check_path_2), canvas_2)
    print(f"Sanity check visual saved to: {check_path_1} and {check_path_2}")

    # Export GeoTIFFs
    # Image 1: GSD = 131.6 m/px
    gsd_1 = 131.6
    H1, W1 = masked_1.shape
    transform_1 = from_origin(0.0, 0.0, float(gsd_1), float(gsd_1))
    tif_path_1 = out_dir / "image_1.tif"

    # Normalize to float32 [0.0, 1.0] for pipeline ingestion
    norm_1 = (masked_1.astype(np.float32) / 255.0)

    with rasterio.open(
        str(tif_path_1),
        "w",
        driver="GTiff",
        height=H1,
        width=W1,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform_1,
    ) as dst:
        dst.write(norm_1, 1)

    print(f"Exported Image 1 GeoTIFF: {tif_path_1} (GSD={gsd_1} m/px)")

    # Image 2: GSD = 555.6 m/px
    gsd_2 = 555.6
    H2, W2 = masked_2.shape
    transform_2 = from_origin(0.0, 0.0, float(gsd_2), float(gsd_2))
    tif_path_2 = out_dir / "image_2.tif"

    norm_2 = (masked_2.astype(np.float32) / 255.0)

    with rasterio.open(
        str(tif_path_2),
        "w",
        driver="GTiff",
        height=H2,
        width=W2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform_2,
    ) as dst:
        dst.write(norm_2, 1)

    print(f"Exported Image 2 GeoTIFF: {tif_path_2} (GSD={gsd_2} m/px)")
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()

