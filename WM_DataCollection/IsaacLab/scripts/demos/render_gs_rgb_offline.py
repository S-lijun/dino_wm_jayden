#!/usr/bin/env python3
"""Render true 3DGS lab RGB offline from saved camera poses (H100-safe).

Isaac Sim rtx_rgb needs Vulkan ray_tracing; H100 only has CUDA. This script uses
gsplat on CUDA to rasterize a standard 3DGS .ply with poses from camera_poses.csv.

Usage:
  python scripts/demos/render_gs_rgb_offline.py \\
    --dataset_dir data/20260618_183746_304 \\
    --gs_ply scene_new/3dgs_lab.ply

Requires: pip install gsplat plyfile
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from camera_pose_utils import pinhole_intrinsics, quat_wxyz_to_rot
from lab_scene_utils import OUTPUT_IMG_RES, SENSOR_IMG_RES, rotate_sensor_ccw_to_landscape


def _load_ply(path: str):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise SystemExit("Install plyfile: pip install plyfile") from exc

    ply = PlyData.read(path)
    v = ply["vertex"]
    names = v.data.dtype.names or ()

    def col(name: str) -> np.ndarray:
        if name not in names:
            raise ValueError(f"PLY missing column {name!r} in {path}")
        return np.asarray(v[name], dtype=np.float32)

    xyz = np.stack([col("x"), col("y"), col("z")], axis=-1)
    opacity = col("opacity")
    scale = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=-1)
    rot = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=-1)

    sh_cols = [n for n in names if n.startswith("f_dc_") or n.startswith("f_rest_")]
    sh_cols.sort()
    if not sh_cols:
        raise ValueError(f"No SH columns in {path}")
    sh = np.stack([col(n) for n in sh_cols], axis=-1)
    return xyz, opacity, scale, rot, sh


def _apply_lab_scene_transform(xyz: np.ndarray) -> np.ndarray:
    """Match load_lab_scene_usd(): translate (2,-1,1.85) + rot Z 50°."""
    angle = np.deg2rad(50.0)
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = np.array([2.0, -1.0, 1.85], dtype=np.float64)
    return (xyz @ r.T) + t


def _world_to_cam(
    cam_pos: np.ndarray,
    cam_quat_wxyz: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    r = quat_wxyz_to_rot(cam_quat_wxyz)
    # camera looks +X in ROS optical frame; world points → camera
    return (points - cam_pos) @ r


def _render_frame_gsplat(
    means,
    quats,
    scales,
    opacities,
    colors,
    cam_pos,
    cam_quat,
    height,
    width,
    fx,
    fy,
    cx,
    cy,
    device,
):
    import torch
    from gsplat import rasterization

    view = torch.eye(4, device=device, dtype=torch.float32)
    r = torch.from_numpy(quat_wxyz_to_rot(cam_quat)).float().to(device)
    view[:3, :3] = r.T
    view[:3, 3] = -r.T @ torch.tensor(cam_pos, device=device, dtype=torch.float32)

    k = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )

    rgbd, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        view.unsqueeze(0),
        k.unsqueeze(0),
        width,
        height,
        sh_degree=0,
    )
    rgb = rgbd[0, ..., :3].clamp(0, 1)
    return (rgb.detach().cpu().numpy() * 255.0).astype(np.uint8)


def _read_poses(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline 3DGS RGB from camera_poses.csv")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--gs_ply",
        type=str,
        default=None,
        help="3DGS PLY (default: scene_new/3dgs_lab.ply next to IsaacLab)",
    )
    parser.add_argument("--out_subdir", type=str, default="images_gs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    demos_dir = os.path.dirname(os.path.abspath(__file__))
    isaaclab_root = os.path.abspath(os.path.join(demos_dir, "../.."))
    dataset_dir = os.path.abspath(args.dataset_dir)
    poses_csv = os.path.join(dataset_dir, "camera_poses.csv")
    if not os.path.isfile(poses_csv):
        raise SystemExit(
            f"Missing {poses_csv}\n"
            "Re-run DataCollection with depth_rgb (camera poses are logged automatically)."
        )

    gs_ply = args.gs_ply
    if gs_ply is None:
        gs_ply = os.path.join(isaaclab_root, "scene_new", "3dgs_lab.ply")
    gs_ply = os.path.abspath(gs_ply)
    if not os.path.isfile(gs_ply):
        raise SystemExit(
            f"3DGS PLY not found: {gs_ply}\n"
            "Export once from your lab scan (standard 3DGS training output) or from 3dgs_lab.usdz on an RTX machine."
        )

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch required") from exc

    try:
        import gsplat  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install gsplat: pip install gsplat") from exc

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, falling back to cpu (slow).")
        device = "cpu"

    print(f"[INFO] Loading GS PLY: {gs_ply}")
    xyz, opacity, scale, rot, sh = _load_ply(gs_ply)
    xyz = _apply_lab_scene_transform(xyz)

    means = torch.from_numpy(xyz).float().to(device)
    quats = torch.from_numpy(rot).float().to(device)
    scales = torch.exp(torch.from_numpy(scale).float().to(device))
    opacities = torch.sigmoid(torch.from_numpy(opacity).float().to(device))
    # DC SH only for speed (degree 0)
    colors = torch.from_numpy(sh[:, :3]).float().to(device)

    height, width = SENSOR_IMG_RES
    fx, fy, cx, cy = pinhole_intrinsics(height, width)
    out_dir = os.path.join(dataset_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    poses = _read_poses(poses_csv)
    if args.max_frames > 0:
        poses = poses[: args.max_frames]

    print(f"[INFO] Rendering {len(poses)} frames → {out_dir}")
    for i, row in enumerate(poses):
        cam_pos = np.array(
            [float(row["cam_px"]), float(row["cam_py"]), float(row["cam_pz"])],
            dtype=np.float64,
        )
        cam_quat = np.array(
            [
                float(row["cam_qw"]),
                float(row["cam_qx"]),
                float(row["cam_qy"]),
                float(row["cam_qz"]),
            ],
            dtype=np.float64,
        )
        rgb = _render_frame_gsplat(
            means,
            quats,
            scales,
            opacities,
            colors,
            cam_pos,
            cam_quat,
            height,
            width,
            fx,
            fy,
            cx,
            cy,
            device,
        )
        rgb = rotate_sensor_ccw_to_landscape(rgb)
        frame_idx = int(row.get("frame_idx", i))
        out_path = os.path.join(out_dir, f"rgb_{frame_idx:06d}.png")
        import imageio

        imageio.imwrite(out_path, rgb)
        if i % 50 == 0:
            print(f"  [{i+1}/{len(poses)}] {out_path} shape={rgb.shape}")

    print(f"[OK] GS RGB frames saved under {out_dir} (landscape {OUTPUT_IMG_RES[1]}×{OUTPUT_IMG_RES[0]})")


if __name__ == "__main__":
    main()
