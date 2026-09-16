"""
scripts/run_verified_stress_demo.py
===================================
Runs LUNA-MATCH on:
1. Verified Ground-Truth Pair (data/samples/verified_a.tif vs verified_b.tif)
2. Illumination Stress Pair (data/samples/stress_a.tif vs stress_b.tif)
Computes and reports:
- RMSE, Held-out RMSE, Overfit ratio, MAE, SSIM, NCC, SDI, Inlier Ratio,
  transform_type, condition_number (kappa), control points count.
- Runs check_against_ground_truth.py on the verified pair.
"""

import sys
import json
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pipeline.orchestrator import LunaMatchPipeline
import subprocess

def run_pair(name, img_a, img_b, job_id, out_copy_dir=None):
    print("=" * 80)
    print(f" RUNNING {name.upper()}: {img_a.name} vs {img_b.name}")
    print("=" * 80)
    t0 = time.time()
    pipeline = LunaMatchPipeline(job_id, str(img_a), str(img_b))
    result = pipeline.run()
    elapsed = round(time.time() - t0, 3)
    result["elapsed_s"] = elapsed

    print(f"[{name}] Status:          {result.get('status')}")
    if result.get("status") == "DONE":
        print(f"[{name}] RMSE:            {result.get('rmse_px')} px")
        print(f"[{name}] Held-Out RMSE:   {result.get('held_out_rmse_px')} px")
        print(f"[{name}] Overfit Ratio:   {result.get('overfit_ratio')}")
        print(f"[{name}] MAE:             {result.get('mae_px')} px")
        print(f"[{name}] SSIM:            {result.get('ssim')}")
        print(f"[{name}] NCC:             {result.get('ncc')}")
        print(f"[{name}] Inlier Ratio:    {result.get('inlier_ratio')*100:.2f}% ({result.get('n_inliers')}/{result.get('n_total')})")
        print(f"[{name}] SDI:             {result.get('sdi')}")
        print(f"[{name}] Transform:       {result.get('transform_type') or result.get('transform')}")
        print(f"[{name}] Condition (κ):   {result.get('condition_number')}")
        print(f"[{name}] Elapsed:         {elapsed}s")
        
        # Check control points CSV
        csv_path = pipeline.paths.get('control_points_csv')
        if csv_path and csv_path.exists():
            with open(csv_path) as f:
                lines = f.readlines()
            print(f"[{name}] Control Points:  {len(lines)-1} rows written to {csv_path.name}")
    else:
        print(f"[{name}] Stage:           {result.get('stage')}")
        print(f"[{name}] Error:           {result.get('error')}")

    if out_copy_dir:
        import shutil
        out_copy_dir = Path(out_copy_dir)
        out_copy_dir.mkdir(parents=True, exist_ok=True)
        for p in pipeline.paths.values():
            if p.exists() and p.is_file():
                shutil.copy2(p, out_copy_dir / p.name)

    return result, pipeline

def main():
    samples_dir = root_dir / "data" / "samples"
    demo_out = root_dir / "demo_output"

    # 1. Verified Pair
    ver_a = samples_dir / "verified_a.tif"
    ver_b = samples_dir / "verified_b.tif"
    ver_res, ver_pipe = run_pair(
        "Verified Ground-Truth Pair",
        ver_a, ver_b,
        "verified_pair",
        out_copy_dir=demo_out / "verified_pair"
    )

    # Run check_against_ground_truth.py
    print("\n" + "-" * 80)
    print(" GROUND TRUTH ACCURACY DECOMPOSITION")
    print("-" * 80)
    cmd = [
        sys.executable,
        str(root_dir / "scripts" / "check_against_ground_truth.py"),
        "--gt", str(samples_dir / "ground_truth_transform.json"),
        "--params", str(demo_out / "verified_pair" / "transform_params.json"),
        "--out", str(demo_out / "verified_pair" / "ground_truth_comparison.json"),
    ]
    gt_proc = subprocess.run(cmd, capture_output=True, text=True)
    print(gt_proc.stdout)
    if gt_proc.stderr:
        print(gt_proc.stderr)

    # 2. Stress Pair
    stress_a = samples_dir / "stress_a.tif"
    stress_b = samples_dir / "stress_b.tif"
    stress_res, stress_pipe = run_pair(
        "Illumination Stress Pair",
        stress_a, stress_b,
        "stress_run",
        out_copy_dir=demo_out / "stress_run"
    )

if __name__ == "__main__":
    main()
