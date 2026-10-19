#!/usr/bin/env python3
"""Pack DataCollection lidar scans into episode_*.npz for offline lidar FT.

Same episode layout as ``pack_real_humanoid_episodes.py``, with the camera
frames replaced by a filtered Mid-360 cloud.

Each frame:

* Load ``lidar_{frame:04d}.npy`` (xyz + reflectivity). Reflectivity is dropped.
* Drop floor and robot body with ``filter_lidar360_points`` — the same
  function that produced ``lidar360_min_distance``.
* Flip ``(x, y, z) -> (x, -y, -z)``. The real head mount is inverted relative
  to the native Mid-360 frame, so this puts points into the sim elevation
  span (-52 deg .. +7 deg).

``dist`` is ``lidar360_min_distance`` in meters. Fine-tune does
``h_s = dist - --lidar_distance_threshold`` (default 1.0), same as vision.

Output keys::

    lidar_xyz     float32 (P, 3)   filtered, sim-frame xyz, frames concatenated
    lidar_counts  int32 (T,)       points per frame; frame t is xyz[start:start+counts[t]]
    proprio       float32 (T, J)
    joint_names   str array (J,)
    act           float32 (T, 3)   [vx, vy, yaw_rate]
    dist          float32 (T,)     lidar360_min_distance in meters
    timestamps    float64 (T,)
    frames        int32 (T,)
    source_traj   str

Example::

    python scripts/pack_real_lidar_episodes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from pack_real_humanoid_episodes import (
    DEFAULT_PROPRIO_DIM,
    joint_columns,
    list_traj_dirs,
    pad_proprio,
    yaw_rate_from_yaw,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = REPO_ROOT / "DataCollection"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from lidar_distance_analysis import filter_lidar360_points  # noqa: E402

DIST_COL = "lidar360_min_distance"


def lidar_npy(traj_dir: Path, frame: int) -> Path:
    return traj_dir / f"lidar_{int(frame):04d}.npy"


def cloud_to_sim(pts: np.ndarray) -> np.ndarray:
    """Native Mid-360 points -> body/floor removed, sim sensor frame."""
    kept = filter_lidar360_points(pts)
    if len(kept) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.asarray(kept[:, :3], dtype=np.float32).copy()
    xyz[:, 1] *= -1.0
    xyz[:, 2] *= -1.0
    return xyz


def pack_trajectory(
    traj_dir: Path,
    *,
    min_frames: int,
    proprio_dim: int = DEFAULT_PROPRIO_DIM,
) -> dict | None:
    csv_path = traj_dir / "robot_log.csv"
    if not csv_path.exists():
        print(f"[SKIP] {traj_dir.name}: no robot_log.csv")
        return None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[SKIP] {traj_dir.name}: empty csv header")
            return None
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "frame" not in fieldnames or DIST_COL not in fieldnames:
        print(f"[SKIP] {traj_dir.name}: need frame + {DIST_COL}")
        return None
    joints = joint_columns(fieldnames)
    if not joints:
        print(f"[SKIP] {traj_dir.name}: no joint_* columns")
        return None

    frames: list[int] = []
    timestamps: list[float] = []
    vx: list[float] = []
    vy: list[float] = []
    yaw: list[float] = []
    dist: list[float] = []
    proprio: list[list[float]] = []
    clouds: list[np.ndarray] = []
    dropped = 0
    dist_mismatch = 0

    for row in rows:
        try:
            frame = int(row["frame"])
            npy = lidar_npy(traj_dir, frame)
            if not npy.exists():
                dropped += 1
                continue
            d_raw = row.get(DIST_COL, "")
            if d_raw == "":
                dropped += 1
                continue
            d_val = float(d_raw)
            xyz = cloud_to_sim(np.load(npy))
            if xyz.shape[0] > 0:
                cloud_min = float(np.linalg.norm(xyz, axis=1).min())
                if abs(cloud_min - d_val) > 1e-3:
                    dist_mismatch += 1
            frames.append(frame)
            timestamps.append(float(row["timestamp"]))
            vx.append(float(row["vx"]))
            vy.append(float(row["vy"]))
            yaw.append(float(row["yaw"]))
            dist.append(d_val)
            proprio.append([float(row[name]) for name in joints])
            clouds.append(xyz)
        except (KeyError, TypeError, ValueError, OSError):
            dropped += 1
            continue

    t = len(frames)
    if t < min_frames:
        print(f"[SKIP] {traj_dir.name}: T={t} < min_frames={min_frames} (dropped={dropped})")
        return None

    counts = np.asarray([c.shape[0] for c in clouds], dtype=np.int32)
    if int(counts.sum()) == 0:
        lidar_xyz = np.zeros((0, 3), dtype=np.float32)
    else:
        lidar_xyz = np.concatenate(clouds, axis=0)
    proprio_np = np.asarray(proprio, dtype=np.float32)
    proprio_np, joints = pad_proprio(proprio_np, joints, int(proprio_dim))
    ts = np.asarray(timestamps, dtype=np.float64)
    yaw_np = np.asarray(yaw, dtype=np.float32)
    act = np.stack(
        [
            np.asarray(vx, dtype=np.float32),
            np.asarray(vy, dtype=np.float32),
            yaw_rate_from_yaw(yaw_np, ts),
        ],
        axis=1,
    )
    dist_np = np.asarray(dist, dtype=np.float32)
    n_finite = int(np.isfinite(dist_np).sum())
    n_unsafe = int(np.sum(dist_np < 1.0))

    payload = {
        "lidar_xyz": lidar_xyz,
        "lidar_counts": counts,
        "proprio": proprio_np,
        "joint_names": np.asarray(joints),
        "act": act,
        "dist": dist_np,
        "timestamps": ts,
        "frames": np.asarray(frames, dtype=np.int32),
        "source_traj": np.asarray(traj_dir.name),
    }
    stats = {
        "trajectory": traj_dir.name,
        "T": t,
        "dropped": dropped,
        "dist_mismatch": dist_mismatch,
        "points": int(counts.sum()),
        "points_per_frame_min": int(counts.min()),
        "points_per_frame_max": int(counts.max()),
        "proprio_dim": int(proprio_np.shape[1]),
        "dist_min": float(np.nanmin(dist_np)),
        "dist_max": float(np.nanmax(dist_np)),
        "dist_finite": n_finite,
        "dist_lt_1m": n_unsafe,
        "act_vx_min": float(act[:, 0].min()),
        "act_vx_max": float(act[:, 0].max()),
        "act_yaw_abs_max": float(np.abs(act[:, 2]).max()),
        "proprio_abs_max": float(np.abs(proprio_np).max()),
    }
    return {"payload": payload, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack real G1 lidar scans into FT episode npz files."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(REPO_ROOT / "DataCollection"),
        help="Folder of YYYYMMDD_HHMMSS trajectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output folder for episode_*.npz. Default: <data_root>/ft_episodes_lidar.",
    )
    parser.add_argument("--min_frames", type=int, default=11, help="Need T >= z_fuse+1.")
    parser.add_argument(
        "--proprio_dim",
        type=int,
        default=DEFAULT_PROPRIO_DIM,
        help="Zero-pad / truncate joint vector to this dim (WM in_chans=37).",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Pack only this timestamp folder name.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_root / "ft_episodes_lidar"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.trajectory:
        traj_dirs = [data_root / args.trajectory]
    else:
        traj_dirs = list_traj_dirs(data_root)

    stats_all = []
    skipped = 0
    for i, traj_dir in enumerate(traj_dirs):
        if not traj_dir.is_dir():
            print(f"[SKIP] missing {traj_dir}")
            skipped += 1
            continue
        packed = pack_trajectory(
            traj_dir,
            min_frames=int(args.min_frames),
            proprio_dim=int(args.proprio_dim),
        )
        if packed is None:
            skipped += 1
            continue
        out_path = out_dir / f"episode_{i:03d}.npz"
        np.savez_compressed(out_path, **packed["payload"])
        packed["stats"]["npz"] = out_path.name
        stats_all.append(packed["stats"])
        s = packed["stats"]
        print(
            f"[OK] {s['trajectory']} -> {out_path.name} "
            f"T={s['T']} pts/frame[{s['points_per_frame_min']},{s['points_per_frame_max']}] "
            f"dist[{s['dist_min']:.2f},{s['dist_max']:.2f}] <1m={s['dist_lt_1m']} "
            f"mismatch={s['dist_mismatch']}"
        )

    t_total = int(sum(s["T"] for s in stats_all))
    unsafe = int(sum(s["dist_lt_1m"] for s in stats_all))
    zero_prop = int(sum(1 for s in stats_all if s["proprio_abs_max"] == 0.0))
    mismatch = int(sum(s["dist_mismatch"] for s in stats_all))
    meta = {
        "n_episodes": len(stats_all),
        "n_skipped": skipped,
        "n_frames": t_total,
        "n_transitions_est": max(0, t_total - len(stats_all)),
        "n_dist_lt_1m": unsafe,
        "frac_dist_lt_1m": (unsafe / t_total) if t_total else 0.0,
        "n_dist_cloud_mismatch": mismatch,
        "n_episodes_zero_proprio": zero_prop,
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "min_frames": int(args.min_frames),
        "proprio_dim": int(args.proprio_dim),
        "keys": [
            "lidar_xyz",
            "lidar_counts",
            "proprio",
            "joint_names",
            "act",
            "dist",
            "timestamps",
            "frames",
        ],
        "notes": [
            "lidar_xyz is filtered native xyz flipped (x, -y, -z) into the sim frame",
            "reflectivity dropped",
            "floor/body filter is filter_lidar360_points (same as lidar360_min_distance)",
            "dist is lidar360_min_distance in meters; FT computes h_s = dist - 1.0",
            "act = [logged vx, logged vy, d(yaw)/dt]; FT clips to env box",
            "proprio zero-padded to WM dim 37 (csv has 29 joints, currently all 0)",
            "joint_* copied as-is from robot_log.csv",
        ],
        "episodes": stats_all,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Summary ===")
    print(f"episodes: {len(stats_all)}  skipped: {skipped}")
    print(f"frames: {t_total}  ~transitions: {meta['n_transitions_est']}")
    print(f"dist<1m: {unsafe} ({meta['frac_dist_lt_1m']:.3f})")
    print(f"cloud min vs dist mismatches: {mismatch}")
    print(f"zero-proprio episodes: {zero_prop}/{len(stats_all)}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
