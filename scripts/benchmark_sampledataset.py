"""
scripts/benchmark_sampledataset.py
==================================
Runs all 9 pairs from sampledataset/ through LunaMatchPipeline,
recording raw metrics, execution logs, and full tracebacks if any errors occur.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
import cv2
import numpy as np

# Ensure luna-match root is in python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pipeline.orchestrator import LunaMatchPipeline
from core.ingest_preprocess import read_raster


def main():
    dataset_dir = root_dir / "sampledataset"
    out_base = root_dir / "demo_output" / "sampledataset_benchmark"
    out_base.mkdir(parents=True, exist_ok=True)

    pairs = [
        # (name, path_a (source), path_b (reference), difficulty_hint)
        ("Pair 2", dataset_dir / "sou2.jpeg", dataset_dir / "res2.jpeg", "Easy (~1.1x scale, 82 SIFT)"),
        ("Pair 5", dataset_dir / "sou5.jpeg", dataset_dir / "res5.jpeg", "Easy/Mod (same dim, 53 SIFT)"),
        ("Pair 8", dataset_dir / "sou8.jpeg", dataset_dir / "res8.jpeg", "Easy/Mod (1.14x scale, 50 SIFT)"),
        ("Pair 3", dataset_dir / "sou3.jpeg", dataset_dir / "res3.jpeg", "Moderate (1.25x scale, 46 SIFT)"),
        ("Pair 6", dataset_dir / "sou6.jpeg", dataset_dir / "res6.jpeg", "Moderate (same dim, 37 SIFT)"),
        ("Pair 7", dataset_dir / "sou7.jpeg", dataset_dir / "res7.jpeg", "Moderate (low-res, 32 SIFT)"),
        ("Pair 9", dataset_dir / "image.png", dataset_dir / "image copy.png", "Hard (4.22x scale, 25 SIFT)"),
        ("Pair 4", dataset_dir / "sou4.jpeg", dataset_dir / "res4.jpeg", "Hard (1600x733, 16 SIFT)"),
        ("Pair 1", dataset_dir / "sou1.jpeg", dataset_dir / "res1.jpeg", "Hardest (1.67x scale, 4 SIFT)"),
    ]

    all_results = {}

    import argparse
    parser = argparse.ArgumentParser(description="Benchmark sampledataset pairs")
    parser.add_argument("--mode", choices=["fast", "standard"], default="fast", help="Performance tier (fast or standard)")
    args = parser.parse_args()
    mode = args.mode

    print("=" * 80)
    print(f" LUNA-MATCH: BENCHMARKING ALL 9 REAL PAIRS IN sampledataset/ (mode={mode})")
    print("=" * 80)

    for name, path_a, path_b, diff in pairs:
        print(f"\n>>> Running {name}: {path_a.name} vs {path_b.name} [{diff}] (mode={mode})")
        t0 = time.time()
        job_id = f"bench_{name.lower().replace(' ', '_')}"

        try:
            pipeline = LunaMatchPipeline(job_id, str(path_a), str(path_b), mode=mode)
            result = pipeline.run()
            elapsed = round(time.time() - t0, 3)
            result["bench_elapsed_s"] = elapsed
            all_results[name] = {
                "source": path_a.name,
                "reference": path_b.name,
                "difficulty": diff,
                "status": result.get("status"),
                "result": result,
            }
            print(f"[{name}] Result Status: {result.get('status')}")
            if result.get("status") == "DONE":
                print(f"    RMSE:            {result.get('rmse_px')} px")
                print(f"    Held-Out RMSE:   {result.get('held_out_rmse_px')} px")
                print(f"    Overfit Ratio:   {result.get('overfit_ratio')}")
                print(f"    MAE:             {result.get('mae_px')} px")
                print(f"    SSIM:            {result.get('ssim')}")
                print(f"    NCC:             {result.get('ncc')}")
                print(f"    Inlier Ratio:    {result.get('inlier_ratio')*100:.1f}% ({result.get('n_inliers')}/{result.get('n_total')})")
                print(f"    SDI:             {result.get('sdi')}")
                print(f"    Transform:       {result.get('transform_type') or result.get('transform')}")
                print(f"    Condition No:    {result.get('condition_number')}")
                print(f"    Elapsed:         {elapsed}s")
            else:
                print(f"    Stage:      {result.get('stage')}")
                if result.get('error_code'):
                    print(f"    Error Code: {result.get('error_code')}")
                print(f"    Error:      {result.get('error')}")
                print(f"    Elapsed:    {elapsed}s")

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            tb = traceback.format_exc()
            print(f"[{name}] CRASHED with exception: {e}")
            print(tb)
            all_results[name] = {
                "source": path_a.name,
                "reference": path_b.name,
                "difficulty": diff,
                "status": "CRASHED",
                "error": str(e),
                "traceback": tb,
                "bench_elapsed_s": elapsed,
            }

    # Save summary json
    summary_path = out_base / "all_pairs_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n" + "=" * 80)
    print(f"Saved comprehensive results to: {summary_path}")
    print("=" * 80)

    # Print markdown summary table
    print("\n" + "=" * 100)
    print("SUMMARY METRICS TABLE")
    print("=" * 100)
    headers = ["Pair", "Status", "Elapsed(s)", "RMSE(px)", "Held-Out RMSE", "Overfit Ratio", "MAE(px)", "SSIM", "NCC", "Inliers", "SDI", "Transform", "Kappa"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---'] * len(headers))} |")
    for name, path_a, path_b, diff in pairs:
        res_info = all_results.get(name, {})
        res = res_info.get("result", {})
        status = res_info.get("status", "FAIL")
        elapsed = res.get("bench_elapsed_s", res_info.get("bench_elapsed_s", "-"))
        elapsed_str = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else str(elapsed)
        if status == "DONE":
            row = [
                name,
                status,
                elapsed_str,
                f"{res.get('rmse_px', '-'):.4f}" if isinstance(res.get('rmse_px'), (int, float)) else str(res.get('rmse_px')),
                f"{res.get('held_out_rmse_px', '-'):.4f}" if isinstance(res.get('held_out_rmse_px'), (int, float)) else str(res.get('held_out_rmse_px')),
                f"{res.get('overfit_ratio', '-'):.4f}" if isinstance(res.get('overfit_ratio'), (int, float)) else str(res.get('overfit_ratio')),
                f"{res.get('mae_px', '-'):.4f}" if isinstance(res.get('mae_px'), (int, float)) else str(res.get('mae_px')),
                f"{res.get('ssim', '-'):.4f}" if isinstance(res.get('ssim'), (int, float)) else str(res.get('ssim')),
                f"{res.get('ncc', '-'):.4f}" if isinstance(res.get('ncc'), (int, float)) else str(res.get('ncc')),
                f"{res.get('n_inliers')}/{res.get('n_total')}",
                f"{res.get('sdi', '-'):.4f}" if isinstance(res.get('sdi'), (int, float)) else str(res.get('sdi')),
                str(res.get('transform_type') or res.get('transform')),
                f"{res.get('condition_number'):.2f}" if isinstance(res.get('condition_number'), (int, float)) else str(res.get('condition_number')),
            ]
        else:
            err_code = res.get("error_code")
            display_status = f"FAILED ({err_code})" if err_code else status
            row = [name, display_status, elapsed_str, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
        print(f"| {' | '.join(row)} |")
    print("=" * 100)

    # Print per-stage timing breakdown table
    print("\n" + "=" * 115)
    print(f"PER-STAGE TIMING BREAKDOWN TABLE (mode={mode})")
    print("=" * 115)
    stage_headers = ["Pair", "Status", "Preproc", "Overlap", "PC+MIM", "Crater", "Match", "ANMS", "Verif", "Refine", "ValSplit", "WarpEval", "Total Pipe", "Wall Time"]
    print(f"| {' | '.join(stage_headers)} |")
    print(f"| {' | '.join(['---'] * len(stage_headers))} |")
    for name, path_a, path_b, diff in pairs:
        res_info = all_results.get(name, {})
        res = res_info.get("result", {})
        status = res_info.get("status", "FAIL")
        st = res.get("stage_timings_ms", {})
        wall_s = res.get("bench_elapsed_s", "-")
        wall_str = f"{wall_s:.2f}s" if isinstance(wall_s, (int, float)) else str(wall_s)
        row = [
            name,
            status,
            f"{st.get('preprocessing_ms', '-')}ms",
            f"{st.get('overlap_check_ms', '-')}ms",
            f"{st.get('phase_congruency_ms', '-')}ms",
            f"{st.get('crater_detection_ms', '-')}ms",
            f"{st.get('dense_matching_ms', '-')}ms",
            f"{st.get('anms_ms', '-')}ms",
            f"{st.get('verification_ms', '-')}ms",
            f"{st.get('refinement_ms', '-')}ms",
            f"{st.get('validation_split_ms', '-')}ms",
            f"{st.get('warp_and_eval_ms', '-')}ms",
            f"{st.get('total_pipeline_ms', '-')}ms",
            wall_str,
        ]
        print(f"| {' | '.join(row)} |")
    print("=" * 115)


if __name__ == "__main__":
    main()

