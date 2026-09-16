"""
scripts/cleanup_old_jobs.py
===========================
Prunes completed/abandoned registration job directories in data/jobs/ older than
a specified threshold. Defaults to a dry run unless --apply is passed explicitly.
"""

from pathlib import Path
import shutil
import time
import argparse
from typing import List


def cleanup_jobs_older_than(
    hours: float = 24.0,
    jobs_dir: str = "data/jobs",
    apply: bool = False,
) -> List[str]:
    """
    Dry-run by default (apply=False): returns list of job dirs that WOULD
    be deleted (older than hours). Does NOT delete anything unless apply=True.
    Never touches data/samples/ regardless of apply flag.
    Skips directories with no status.json (safety guard).
    """
    jobs_path = Path(jobs_dir).resolve()
    if not jobs_path.exists():
        return []

    # Safety guard against targeting samples directory
    samples_path = (jobs_path.parent / "samples").resolve()

    now = time.time()
    cutoff_seconds = hours * 3600.0
    deleted_targets: List[str] = []

    for item in sorted(jobs_path.iterdir()):
        if not item.is_dir():
            continue

        resolved = item.resolve()
        # Never touch data/samples/
        if samples_path in resolved.parents or resolved == samples_path:
            continue

        status_file = item / "status.json"
        # Safety guard: skips directories with no status.json
        if not status_file.exists():
            continue

        # Check age using status.json mtime (or directory mtime)
        mtime = status_file.stat().st_mtime
        age_seconds = now - mtime

        if age_seconds >= cutoff_seconds:
            target_str = str(item)
            if apply:
                try:
                    shutil.rmtree(item, ignore_errors=False)
                    deleted_targets.append(target_str)
                except Exception as e:
                    # If error deleting, skip or log
                    pass
            else:
                deleted_targets.append(target_str)

    return deleted_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup old job directories in data/jobs")
    parser.add_argument("--hours", type=float, default=24.0, help="Age cutoff in hours")
    parser.add_argument("--apply", action="store_true", help="Perform deletion (dry-run by default)")
    args = parser.parse_args()

    targets = cleanup_jobs_older_than(hours=args.hours, apply=args.apply)
    if args.apply:
        print(f"Deleted {len(targets)} job directories.")
    else:
        print(f"Dry run: would delete {len(targets)} directories:")
        for t in targets:
            print(f"  {t}")
