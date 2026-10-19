#!/usr/bin/env python3
"""Append lidar distances onto each trajectory's robot_log.csv.

Matches rows by frame number. New columns are always last.
Re-running updates those columns instead of adding duplicates.
"""

from pathlib import Path
import argparse
import csv
import re


TRAJ_PATTERN = re.compile(r"^\d{8}_\d{6}$")
COLUMNS = ("filtered_min_distance", "lidar360_min_distance")


def load_distances(lidar_csv):
    distances = {}
    with open(lidar_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            raise ValueError(f"{lidar_csv.name} has no frame column")
        missing = [col for col in COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"{lidar_csv.name} has no {', '.join(missing)} column")
        for row in reader:
            distances[int(row["frame"])] = {col: row[col] for col in COLUMNS}
    return distances


def merge_trajectory(traj_dir):
    robot_csv = traj_dir / "robot_log.csv"
    lidar_csv = traj_dir / f"lidar_distance_analysis_{traj_dir.name}.csv"

    if not robot_csv.exists():
        print(f"[SKIP] {traj_dir.name}: no robot_log.csv")
        return None
    if not lidar_csv.exists():
        print(f"[SKIP] {traj_dir.name}: no {lidar_csv.name}")
        return None

    distances = load_distances(lidar_csv)

    with open(robot_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            print(f"[FAIL] {traj_dir.name}: robot_log.csv has no frame column")
            return None
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    fieldnames = [name for name in fieldnames if name not in COLUMNS]
    fieldnames.extend(COLUMNS)

    matched = 0
    missing = 0
    for row in rows:
        frame = int(row["frame"])
        if frame in distances:
            row.update(distances[frame])
            matched += 1
        else:
            for col in COLUMNS:
                row[col] = ""
            missing += 1

    extra = len(distances) - matched

    with open(robot_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    note = ""
    if missing or extra:
        note = f" | unmatched robot rows={missing}, extra lidar frames={extra}"
    print(f"[OK] {traj_dir.name}: {matched}/{len(rows)} frames{note}")

    return {
        "trajectory": traj_dir.name,
        "robot_rows": len(rows),
        "matched": matched,
        "missing": missing,
        "extra_lidar_frames": extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory",
        nargs="?",
        help="Optional trajectory folder, e.g. 20260630_052843. If omitted, merges all trajectories.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.trajectory:
        traj_dirs = [base_dir / args.trajectory]
    else:
        traj_dirs = [
            p for p in sorted(base_dir.iterdir())
            if p.is_dir() and TRAJ_PATTERN.match(p.name)
        ]

    results = []
    for traj_dir in traj_dirs:
        if not traj_dir.exists():
            print(f"[ERROR] Missing trajectory folder: {traj_dir}")
            continue
        result = merge_trajectory(traj_dir)
        if result is not None:
            results.append(result)

    print("\n=== Summary ===")
    print(f"Trajectories updated: {len(results)}")
    mismatched = [r for r in results if r["missing"] or r["extra_lidar_frames"]]
    if mismatched:
        print(f"Trajectories with frame mismatch: {len(mismatched)}")
        for row in mismatched:
            print(
                f"  {row['trajectory']}: robot={row['robot_rows']} "
                f"matched={row['matched']} missing={row['missing']} "
                f"extra_lidar={row['extra_lidar_frames']}"
            )
    else:
        print("Every robot_log row matched a lidar frame.")


if __name__ == "__main__":
    main()
