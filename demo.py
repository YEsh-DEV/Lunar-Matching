"""
demo.py
=======
LUNA-MATCH Live Demonstration & Benchmark Runner.

Executes the full 7-stage registration pipeline on sample lunar image pairs
and generates visual validation figures (checkerboard alignment, tie-point matches,
side-by-side comparison).

Usage:
  python demo.py --img-a data/samples/image_1.tif --img-b data/samples/image_2.tif --out demo_output/real_pair
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.orchestrator import LunaMatchPipeline, JobState
from core.ingest_preprocess import read_raster
from core.dense_matcher import _HAS_KORNIA, select_matcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("luna_demo")


def create_checkerboard(img1: np.ndarray, img2: np.ndarray, tile_size: int = 32) -> np.ndarray:
    """
    Create a checkerboard mosaic interleaving img1 and img2 in blocks of tile_size x tile_size.
    Essential for judges to visually verify crater rim continuity across seams.
    """
    H, W = img1.shape[:2]
    # Ensure same shape
    if img2.shape[:2] != (H, W):
        img2 = cv2.resize(img2, (W, H))

    checkerboard = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            y_end = min(H, y + tile_size)
            x_end = min(W, x + tile_size)

            tile_y = (y // tile_size) % 2
            tile_x = (x // tile_size) % 2

            if (tile_y + tile_x) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = img1[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]

    return checkerboard


def draw_tie_points(
    img_a: np.ndarray,
    img_b: np.ndarray,
    matches: np.ndarray,
    max_draw: int = 80,
) -> np.ndarray:
    """
    Draw side-by-side correspondence lines connecting matched keypoints.
    """
    H_a, W_a = img_a.shape[:2]
    H_b, W_b = img_b.shape[:2]

    # Canvas height = max(H_a, H_b), width = W_a + W_b
    canvas_h = max(H_a, H_b)
    canvas_w = W_a + W_b

    def to_u8(img):
        norm = (img - np.nanmin(img)) / max(np.nanmax(img) - np.nanmin(img), 1e-6)
        return (norm * 255.0).astype(np.uint8)

    c_a = to_u8(img_a)
    c_b = to_u8(img_b)

    # Convert to RGB canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:H_a, :W_a] = cv2.cvtColor(c_a, cv2.COLOR_GRAY2BGR)
    canvas[:H_b, W_a:canvas_w] = cv2.cvtColor(c_b, cv2.COLOR_GRAY2BGR)

    # Sample matches evenly if too many
    if len(matches) > max_draw:
        step = len(matches) // max_draw
        draw_matches = matches[::step][:max_draw]
    else:
        draw_matches = matches

    for m in draw_matches:
        x1, y1, x2, y2 = int(round(m[0])), int(round(m[1])), int(round(m[2])), int(round(m[3]))
        pt1 = (x1, y1)
        pt2 = (x2 + W_a, y2)
        conf = float(m[4]) if len(m) > 4 else 0.8

        # Color: Emerald green for high confidence, Cyan for moderate
        color = (0, 240, 100) if conf > 0.6 else (255, 200, 0)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="LUNA-MATCH Demo Runner")
    parser.add_argument("--img-a", required=True, help="Moving image (source) path")
    parser.add_argument("--img-b", required=True, help="Reference image path")
    parser.add_argument("--out", default="demo_output/real_pair", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" LUNA-MATCH: Planetary Image Registration Pipeline — Live Demo")
    print(" Problem Statement ID: SIH26166 | Space Technology")
    print("=" * 75)

    # Ingest metadata for display
    raw_a, meta_a = read_raster(args.img_a)
    raw_b, meta_b = read_raster(args.img_b)

    gsd_a = meta_a.get('gsd', 131.6)
    gsd_b = meta_b.get('gsd', 555.6)
    gsd_ratio = max(gsd_a, gsd_b) / max(min(gsd_a, gsd_b), 1e-6)

    # Determine matcher architecture used
    selected = select_matcher(gsd_ratio)
    matcher_desc = "Classical SIFT with Lowe's ratio test (0.8) + ORB safety net"
    if selected == 'loftr' and _HAS_KORNIA:
        matcher_desc = "LoFTR Detector-Free Local Feature Transformer (Kornia / PyTorch)"

    print(f"Moving Image (A)    : {args.img_a} (GSD: {gsd_a:.1f} m/px, Size: {raw_a.shape[1]}x{raw_a.shape[0]})")
    print(f"Reference Image (B) : {args.img_b} (GSD: {gsd_b:.1f} m/px, Size: {raw_b.shape[1]}x{raw_b.shape[0]})")
    print(f"Scale Disparity     : {gsd_ratio:.2f}x")
    print(f"Matcher Path Used   : {matcher_desc}")
    print("-" * 75)
    print("Executing full 7-stage registration pipeline...")

    # Run pipeline
    job_id = f"demo_{int(time.time())}"
    pipeline = LunaMatchPipeline(job_id, args.img_a, args.img_b)
    result = pipeline.run()

    status = result.get('status')
    is_fallback = result.get('is_synthetic_fallback', False)

    if status != 'DONE':
        print("=" * 75)
        print(" ❌ REGISTRATION FAILED")
        print(f" Failed Stage  : {result.get('stage', 'N/A')}")
        print(f" Error Details : {result.get('error', 'Unknown failure')}")
        if is_fallback:
            print(" ⚠️  WARNING: METRICS ARE NOT REAL — SYNTHETIC FALLBACK ACTIVE")
        print("=" * 75)

        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Failure report saved to: {metrics_path}")
        return

    if is_fallback:
        print("\n" + "!" * 75)
        print(" ⚠️  WARNING: METRICS ARE NOT REAL — SYNTHETIC FALLBACK ACTIVE")
        print("!" * 75 + "\n")

    print("=" * 75)
    print(" REGISTRATION METRICS & QUALITY REPORT")
    print("=" * 75)
    print(f"Status                 : {status}")
    print(f"Reprojection RMSE      : {result.get('rmse_px', 'N/A')} pixels")
    print(f"Inlier Verification    : {result.get('inlier_ratio', 0.0)*100:.1f}% ({result.get('n_inliers', 0)} / {result.get('n_total', 0)} verified)")
    print(f"Spatial Uniformity SDI : {result.get('sdi', 'N/A')} (Shannon Spatial Entropy)")
    print(f"Geometric Transform    : {str(result.get('transform', 'N/A')).upper()}")
    print(f"Synthetic Fallback     : {'ACTIVE ⚠️' if is_fallback else 'NO (Genuine matches only ✅)'}")
    print(f"Total Elapsed Time     : {result.get('elapsed_s', 'N/A')} seconds")
    print("=" * 75)

    # Save metrics JSON to output
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    # Copy transform_params.json to output directory if present
    job_dir = Path("data") / "jobs" / job_id
    transform_src = job_dir / "intermediate" / "transform_params.json"
    if transform_src.exists():
        import shutil
        shutil.copy(str(transform_src), str(out_dir / "transform_params.json"))
    matches_raw_path = job_dir / "intermediate" / "matches_raw.npy"
    matches_ver_path = job_dir / "intermediate" / "matches_verified.npy"
    registered_path = job_dir / "output" / "registered.tif"

    if not registered_path.exists():
        registered_path = job_dir / "output" / "registered.npy"

    if registered_path.name.endswith(".tif"):
        with rasterio.open(str(registered_path)) as s:
            registered_img = s.read(1).astype(np.float64)
    else:
        registered_img = np.load(str(registered_path))

    # 1. Generate Checkerboard Figure (Interleave Frame A with Warped Registered Image)
    checkerboard = create_checkerboard(raw_a, registered_img, tile_size=36)
    checkerboard_u8 = (np.clip(checkerboard, 0, 1) * 255.0).astype(np.uint8)
    cb_path = out_dir / "checkerboard_overlay.png"
    cv2.imwrite(str(cb_path), checkerboard_u8)
    print(f"[Visual 1] Checkerboard Overlay saved to: {cb_path}")

    # 2. Generate Tie-Points Correspondence Map
    if matches_ver_path.exists():
        verified_matches = np.load(str(matches_ver_path))
    elif matches_raw_path.exists():
        verified_matches = np.load(str(matches_raw_path))
    else:
        verified_matches = np.zeros((0, 5))

    if len(verified_matches) > 0:
        tie_points_img = draw_tie_points(raw_a, raw_b, verified_matches)
        tp_path = out_dir / "tie_points_matches.png"
        cv2.imwrite(str(tp_path), tie_points_img)
        print(f"[Visual 2] Tie-Points Correspondence Map saved to: {tp_path}")

    # 3. Generate 3-Panel Side-by-Side Comparison (Matplotlib)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0A0D14')
    for ax in axes:
        ax.set_facecolor('#0A0D14')
        ax.tick_params(colors='white')

    axes[0].imshow(raw_a, cmap='gray')
    axes[0].set_title(f"Target Frame A (GSD: {gsd_a} m/px)", color='#00F2FE', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(registered_img, cmap='gray')
    axes[1].set_title(f"Registered (Warped B to Frame A via {result.get('transform')})", color='#10B981', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(raw_b, cmap='gray')
    axes[2].set_title(f"Original Image B (GSD: {gsd_b} m/px)", color='#F59E0B', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    comparison_path = out_dir / "side_by_side_comparison.png"
    plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight', facecolor='#0A0D14')
    plt.close()
    print(f"[Visual 3] 3-Panel Comparison Figure saved to: {comparison_path}")

    # 4. Generate Checkerboard Comparison Plot with Seam Labels
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0A0D14')
    ax.imshow(checkerboard, cmap='gray')
    ax.set_title(f"Checkerboard Seam Continuity (Tile: 36px | RMSE: {result.get('rmse_px')}px)", color='white', fontsize=14, fontweight='bold')
    ax.axis('off')
    cb_fig_path = out_dir / "checkerboard_analysis.png"
    plt.savefig(str(cb_fig_path), dpi=150, bbox_inches='tight', facecolor='#0A0D14')
    plt.close()
    print(f"[Visual 4] Checkerboard Analysis Figure saved to: {cb_fig_path}")

    print("\nDemonstration execution completed successfully!")
    print(f"All validation artifacts stored in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
