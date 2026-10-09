#!/usr/bin/env python3
"""Pack DataCollection timestamp folders into episode_*.npz for offline FT.

Each ``YYYYMMDD_HHMMSS`` folder is one trajectory. Frames are ``frame_XXXX.jpg``
aligned to ``robot_log.csv`` by the ``frame`` column.

Output keys (what ``finetune_HJ_humanoid_sac_path.py`` reads)::

    visual      uint8 (T, H, W, 3)   frame_{frame:04d}.jpg resized to WM HxW
    proprio     float32 (T, J)       joint_0 .. joint_K in csv order
    joint_names str array (J,)
    act         float32 (T, 3)       [vx, vy, yaw_rate] env-space
    dist        float32 (T,)         filtered_min_distance in meters
    timestamps  float64 (T,)         optional, ignored by FT
    frames      int32 (T,)           optional, ignored by FT

``filtered_min_distance`` is range in meters, not h_s. Fine-tune does
``h_s = dist - --lidar_distance_threshold`` (default 1.0).

``yaw_rate`` is not in the csv; it is wrapped Δyaw / Δt. Logged ``vx, vy``
are used as the executed action (FT later clips to the env box).

Example::

    python scripts/pack_real_humanoid_episodes.py
    python scripts/pack_real_humanoid_episodes.py --out_dir DataCollection/ft_episodes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


TRAJ_PATTERN = re.compile(r"^\d{8}_\d{6}$")
DIST_COL = "filtered_min_distance"
JOINT_RE = re.compile(r"^joint_(\d+)$")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WM_HW = (192, 256)  # wm_ckpt_18-27-17 / run_train_humanoid_wm_192x256.sh
DEFAULT_PROPRIO_DIM = 37  # WM proprio_encoder in_chans (G1 Isaac)


def load_wm_hw(ckpt_dir: Path) -> tuple[int, int] | None:
    """Read img_h / img_w from a WM run hydra.yaml."""
    hydra = ckpt_dir / "hydra.yaml"
    if not hydra.is_file():
        return None
    text = hydra.read_text(encoding="utf-8")
    mh = re.search(r"^img_h:\s*(\d+)", text, re.M)
    mw = re.search(r"^img_w:\s*(\d+)", text, re.M)
    if mh and mw:
        return int(mh.group(1)), int(mw.group(1))
    return None


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def joint_columns(fieldnames: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for name in fieldnames:
        m = JOINT_RE.match(name)
        if m:
            found.append((int(m.group(1)), name))
    found.sort(key=lambda x: x[0])
    return [name for _, name in found]


def frame_jpg(traj_dir: Path, frame: int) -> Path:
    return traj_dir / f"frame_{int(frame):04d}.jpg"


def load_rgb(path: Path, resize_hw: tuple[int, int] | None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        h, w = resize_hw
        img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def yaw_rate_from_yaw(yaw: np.ndarray, ts: np.ndarray) -> np.ndarray:
    t = int(yaw.shape[0])
    out = np.zeros((t,), dtype=np.float32)
    if t < 2:
        return out
    dt = np.diff(ts.astype(np.float64))
    dyaw = wrap_pi(np.diff(yaw.astype(np.float64)))
    dt = np.maximum(dt, 1e-3)
    rates = (dyaw / dt).astype(np.float32)
    out[:-1] = rates
    out[-1] = rates[-1]
    return out


def pad_proprio(values: np.ndarray, names: list[str], dim: int) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float32)
    src = int(values.shape[-1])
    names = list(names)
    if src == dim:
        return values, names
    if src < dim:
        pad = np.zeros((values.shape[0], dim - src), dtype=np.float32)
        values = np.concatenate([values, pad], axis=1)
        names = names + [f"joint_{i}" for i in range(src, dim)]
        return values, names
    return values[:, :dim].copy(), names[:dim]


def pack_trajectory(
    traj_dir: Path,
    *,
    min_frames: int,
    resize_hw: tuple[int, int] | None,
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
    visuals: list[np.ndarray] = []
    dropped = 0

    for row in rows:
        try:
            frame = int(row["frame"])
            jpg = frame_jpg(traj_dir, frame)
            if not jpg.exists():
                dropped += 1
                continue
            d_raw = row.get(DIST_COL, "")
            if d_raw == "":
                dropped += 1
                continue
            d_val = float(d_raw)
            visuals.append(load_rgb(jpg, resize_hw))
            frames.append(frame)
            timestamps.append(float(row["timestamp"]))
            vx.append(float(row["vx"]))
            vy.append(float(row["vy"]))
            yaw.append(float(row["yaw"]))
            dist.append(d_val)
            proprio.append([float(row[name]) for name in joints])
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue

    t = len(frames)
    if t < min_frames:
        print(f"[SKIP] {traj_dir.name}: T={t} < min_frames={min_frames} (dropped={dropped})")
        return None

    visual = np.stack(visuals, axis=0)
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
        "visual": visual,
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
        "visual_hw": [int(visual.shape[1]), int(visual.shape[2])],
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


def list_traj_dirs(data_root: Path) -> list[Path]:
    return [
        p
        for p in sorted(data_root.iterdir())
        if p.is_dir() and TRAJ_PATTERN.match(p.name)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack real G1 DataCollection folders into FT episode npz files."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "DataCollection"),
        help="Folder of YYYYMMDD_HHMMSS trajectories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output folder for episode_*.npz. Default: <data_root>/ft_episodes.",
    )
    parser.add_argument("--min_frames", type=int, default=11, help="Need T >= z_fuse+1.")
    parser.add_argument(
        "--dino_ckpt_dir",
        type=str,
        default=str(REPO_ROOT / "wm_ckpt_18-27-17"),
        help="WM run folder; img_h/img_w are read from hydra.yaml.",
    )
    parser.add_argument(
        "--img_h",
        type=int,
        default=None,
        help="Resize height. Default: hydra.yaml img_h, else 192.",
    )
    parser.add_argument(
        "--img_w",
        type=int,
        default=None,
        help="Resize width. Default: hydra.yaml img_w, else 256.",
    )
    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="Keep native jpg size (480x640). Do not use for FT.",
    )
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
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_root / "ft_episodes"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_resize:
        resize_hw = None
    else:
        if (args.img_h is None) ^ (args.img_w is None):
            raise SystemExit("Pass both --img_h and --img_w, or neither.")
        if args.img_h is not None:
            resize_hw = (int(args.img_h), int(args.img_w))
        else:
            wm_hw = load_wm_hw(Path(args.dino_ckpt_dir))
            resize_hw = wm_hw if wm_hw is not None else DEFAULT_WM_HW
        print(f"[INFO] resize visual to H={resize_hw[0]} W={resize_hw[1]} (WM training size)")

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
            resize_hw=resize_hw,
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
            f"T={s['T']} dist[{s['dist_min']:.2f},{s['dist_max']:.2f}] "
            f"<1m={s['dist_lt_1m']} proprio_abs_max={s['proprio_abs_max']:.3g}"
        )

    t_total = int(sum(s["T"] for s in stats_all))
    unsafe = int(sum(s["dist_lt_1m"] for s in stats_all))
    zero_prop = int(sum(1 for s in stats_all if s["proprio_abs_max"] == 0.0))
    meta = {
        "n_episodes": len(stats_all),
        "n_skipped": skipped,
        "n_frames": t_total,
        "n_transitions_est": max(0, t_total - len(stats_all)),
        "n_dist_lt_1m": unsafe,
        "frac_dist_lt_1m": (unsafe / t_total) if t_total else 0.0,
        "n_episodes_zero_proprio": zero_prop,
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "min_frames": int(args.min_frames),
        "resize_hw": list(resize_hw) if resize_hw else None,
        "wm_ckpt": str(Path(args.dino_ckpt_dir).resolve()),
        "proprio_dim": int(args.proprio_dim),
        "keys": ["visual", "proprio", "joint_names", "act", "dist", "timestamps", "frames"],
        "notes": [
            "visual resized to WM img_h x img_w (192x256 for wm_ckpt_18-27-17)",
            "proprio zero-padded to WM dim 37 (csv has 29 joints, currently all 0)",
            "dist is filtered_min_distance in meters; FT computes h_s = dist - 1.0",
            "act = [logged vx, logged vy, d(yaw)/dt]; FT clips to env box",
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
    print(f"zero-proprio episodes: {zero_prop}/{len(stats_all)}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
