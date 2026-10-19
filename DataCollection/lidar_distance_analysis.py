# import numpy as np

# pts = np.load("20260612_015000/lidar_0001.npy")

# dist = np.sqrt(np.sum(pts[:,:3]**2, axis=1))

# idx = np.argmin(dist)

# print("closest distance:", dist[idx])
# print("closest point:", pts[idx])

#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import re
import numpy as np


TRAJ_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def load_points(lidar_file):
    pts = np.load(lidar_file)

    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Unexpected shape {pts.shape}")

    return pts


def closest_point(pts):
    dist = np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))
    idx = np.argmin(dist)
    return dist[idx], pts[idx]


def filter_obstacle_points(
    pts,
    z_min=-0.2,
    x_min=0.2,
    x_max=4.0,
    y_abs_max=2.0,
    min_distance=0.25
):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    dist = np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))

    mask = (
        (z > z_min) &              # remove floor-ish points
        (x > x_min) &              # remove robot body/behind robot
        (x < x_max) &              # ignore far background
        (np.abs(y) < y_abs_max) &  # ignore side walls
        (dist > min_distance)      # remove points too close to sensor
    )

    return pts[mask]


def filter_lidar360_points(pts, z_min=-0.2, min_distance=0.25):
    """Full-circle points with the robot body and floor removed.

    The dense body sits inside min_distance. A thinner shell remains just
    outside that ball: shoulders and the back of the head, with x near 0
    or slightly negative. Points in front of the sensor (x around 0.2 m)
    are left in, so a real close obstacle still sets the minimum.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    dist = np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))
    shoulder = (
        (x < 0.18)
        & (x > -0.30)
        & (np.abs(y) > 0.10)
        & (np.abs(y) < 0.36)
        & (z > 0.08)
        & (z < 0.38)
        & (dist < 0.46)
    )
    back_of_head = (
        (x < 0.0)
        & (x > -0.30)
        & (np.abs(y) < 0.12)
        & (z > 0.05)
        & (z < 0.30)
        & (dist < 0.40)
    )
    mask = (z > z_min) & (dist > min_distance) & ~shoulder & ~back_of_head
    return pts[mask]


def analyze_lidar_file(lidar_file, filters):
    pts = load_points(lidar_file)

    raw_dist, raw_pt = closest_point(pts)

    filtered_pts = filter_obstacle_points(pts, **filters)

    if len(filtered_pts) > 0:
        filt_dist, filt_pt = closest_point(filtered_pts)
    else:
        filt_dist, filt_pt = None, None

    scan_pts = filter_lidar360_points(
        pts,
        z_min=filters["z_min"],
        min_distance=filters["min_distance"],
    )
    if len(scan_pts) > 0:
        scan_dist, _scan_pt = closest_point(scan_pts)
    else:
        scan_dist = None

    return {
        "lidar_file": lidar_file.name,
        "frame": lidar_file.stem.replace("lidar_", ""),
        "num_points": len(pts),

        "raw_min_distance": raw_dist,
        "raw_x": raw_pt[0],
        "raw_y": raw_pt[1],
        "raw_z": raw_pt[2],
        "raw_reflectivity": raw_pt[3] if len(raw_pt) > 3 else "",

        "num_filtered_points": len(filtered_pts),
        "filtered_min_distance": filt_dist,
        "filtered_x": filt_pt[0] if filt_pt is not None else "",
        "filtered_y": filt_pt[1] if filt_pt is not None else "",
        "filtered_z": filt_pt[2] if filt_pt is not None else "",
        "filtered_reflectivity": filt_pt[3] if filt_pt is not None and len(filt_pt) > 3 else "",
        "lidar360_min_distance": scan_dist,
    }


def min_distance_row(rows, column):
    best = None
    best_val = None
    for row in rows:
        val = row.get(column, "")
        if val in ("", None):
            continue
        val = float(val)
        if best_val is None or val < best_val:
            best = row
            best_val = val
    return best


def summary_from_rows(traj_name, rows):
    best_row = min_distance_row(rows, "filtered_min_distance")
    scan_row = min_distance_row(rows, "lidar360_min_distance")
    if best_row is None and scan_row is None:
        return None
    return {
        "trajectory": traj_name,
        "num_lidar_files": len(rows),
        "closest_filtered_distance": best_row["filtered_min_distance"] if best_row else "",
        "closest_lidar_file": best_row["lidar_file"] if best_row else "",
        "closest_x": best_row["filtered_x"] if best_row else "",
        "closest_y": best_row["filtered_y"] if best_row else "",
        "closest_z": best_row["filtered_z"] if best_row else "",
        "closest_lidar360_distance": scan_row["lidar360_min_distance"] if scan_row else "",
        "closest_lidar360_file": scan_row["lidar_file"] if scan_row else "",
    }


def analyze_trajectory(traj_dir, filters, overwrite=False):
    lidar_files = sorted(traj_dir.glob("lidar_*.npy"))

    if not lidar_files:
        print(f"[SKIP] {traj_dir.name}: no lidar_*.npy files")
        return None

    out_csv = traj_dir / f"lidar_distance_analysis_{traj_dir.name}.csv"

    if out_csv.exists() and not overwrite:
        with open(out_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            has_scan = reader.fieldnames is not None and "lidar360_min_distance" in reader.fieldnames
        if not has_scan:
            print(f"[REDO] {traj_dir.name}: adding lidar360_min_distance")
        else:
            print(f"[READ] {traj_dir.name}: {out_csv.name} already exists")

            with open(out_csv, "r") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

            return summary_from_rows(traj_dir.name, existing_rows)

    rows = []
    best_row = None

    for lidar_file in lidar_files:
        try:
            row = analyze_lidar_file(lidar_file, filters)
            rows.append(row)

            if row["filtered_min_distance"] is not None:
                if best_row is None or row["filtered_min_distance"] < best_row["filtered_min_distance"]:
                    best_row = row

        except Exception as e:
            print(f"[ERROR] {traj_dir.name}/{lidar_file.name}: {e}")

    if not rows:
        print(f"[FAIL] {traj_dir.name}: no rows written")
        return None

    fieldnames = list(rows[0].keys())

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if best_row is not None:
        print(
            f"[OK] {traj_dir.name}: {len(rows)} lidar files | "
            f"closest filtered = {best_row['filtered_min_distance']:.3f} m "
            f"at {best_row['lidar_file']}"
        )
    else:
        print(f"[WARN] {traj_dir.name}: no filtered obstacle points found")

    return summary_from_rows(traj_dir.name, rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "trajectory",
        nargs="?",
        help="Optional trajectory folder, e.g. 20260508_081422. If omitted, analyzes all trajectories."
    )

    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--z-min", type=float, default=-0.2)
    parser.add_argument("--x-min", type=float, default=0.2)
    parser.add_argument("--x-max", type=float, default=4.0)
    parser.add_argument("--y-abs-max", type=float, default=2.0)
    parser.add_argument("--min-distance", type=float, default=0.25)

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    filters = {
        "z_min": args.z_min,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "y_abs_max": args.y_abs_max,
        "min_distance": args.min_distance,
    }

    print("Using filters:")
    for k, v in filters.items():
        print(f"  {k}: {v}")

    summary_rows = []

    if args.trajectory:
        traj_dirs = [base_dir / args.trajectory]
    else:
        traj_dirs = [
            p for p in sorted(base_dir.iterdir())
            if p.is_dir() and TRAJ_PATTERN.match(p.name)
        ]

    for traj_dir in traj_dirs:
        if not traj_dir.exists():
            print(f"[ERROR] Missing trajectory folder: {traj_dir}")
            continue

        result = analyze_trajectory(traj_dir, filters, overwrite=args.overwrite)
        if result is not None:
            summary_rows.append(result)

    if summary_rows:
        summary_csv = base_dir / "lidar_distance_summary_all_trajectories.csv"

        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n=== Summary ===")
        print(f"Trajectories analyzed: {len(summary_rows)}")
        print(f"Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()