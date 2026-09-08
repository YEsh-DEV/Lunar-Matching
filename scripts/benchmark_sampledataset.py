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

    print("=" * 80)
    print(" LUNA-MATCH: BENCHMARKING ALL 9 REAL PAIRS IN sampledataset/")
    print("=" * 80)

    for name, path_a, path_b, diff in pairs:
        print(f"\n>>> Running {name}: {path_a.name} vs {path_b.name} [{diff}]")
        t0 = time.time()
        job_id = f"bench_{name.lower().replace(' ', '_')}"

        try:
            pipeline = LunaMatchPipeline(job_id, str(path_a), str(path_b))
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
                print(f"    RMSE: {result.get('rmse_px')} px")
                print(f"    Inlier Ratio: {result.get('inlier_ratio')*100:.1f}% ({result.get('n_inliers')}/{result.get('n_total')})")
                print(f"    SDI: {result.get('sdi')}")
                print(f"    Transform: {result.get('transform')}")
                print(f"    Elapsed: {elapsed}s")
            else:
                print(f"    Stage: {result.get('stage')}")
                print(f"    Error: {result.get('error')}")

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


if __name__ == "__main__":
    main()
