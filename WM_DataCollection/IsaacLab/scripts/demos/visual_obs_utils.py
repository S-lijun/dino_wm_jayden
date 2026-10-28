"""Visual observations without RTX / --enable_cameras (cluster-safe).

On HPC nodes where ``--enable_cameras`` triggers NGX/Vulkan failures, use
``--visual_mode depth_rgb`` (default).  This uses Isaac Lab *RayCasterCamera*
(warp ray-mesh, same stack as LiDAR) and converts depth to 3-channel uint8.

Modes
-----
- ``off``       : no images
- ``depth_rgb`` : pinhole depth ray-cast, merged per obstacle mesh → pseudo-RGB
- ``lidar_rgb`` : Velodyne-style range image from merged LiDAR → pseudo-RGB
- ``rtx_rgb``   : true RTX ``TiledCamera`` (needs ``--enable_cameras``; often broken on cluster)
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

VISUAL_MODES = ("off", "depth_rgb", "lidar_rgb", "rtx_rgb")
DEFAULT_VISUAL_MODE = "depth_rgb"  # cluster-safe; DataCollection_test defaults to rtx_rgb
IMG_RES_LANDSCAPE = (480, 640)  # (height, width) = 640×480 landscape
LIDAR_CHANNELS = 45
LIDAR_HORIZONTAL_RES = 2.0
LIDAR_H_FOV = (-180.0, 180.0)


def resolve_visual_mode(args_cli) -> str:
    """Pick visual mode; never silently enable RTX on cluster."""
    mode = getattr(args_cli, "visual_mode", DEFAULT_VISUAL_MODE)
    if mode not in VISUAL_MODES:
        raise ValueError(f"visual_mode must be one of {VISUAL_MODES}, got {mode!r}")

    if getattr(args_cli, "enable_cameras", False) and mode != "rtx_rgb":
        warnings.warn(
            "--enable_cameras is set but visual_mode is not rtx_rgb. "
            f"Using {mode!r} without RTX (ignoring --enable_cameras).",
            stacklevel=2,
        )

    if mode == "rtx_rgb":
        args_cli.enable_cameras = True
    else:
        args_cli.enable_cameras = False

    return mode


def configure_app_for_visual(args_cli, visual_mode: str, *, verbose: bool = True) -> None:
    """Call before AppLauncher. Only rtx_rgb touches enable_cameras / kit_args."""
    if visual_mode != "rtx_rgb":
        args_cli.enable_cameras = False
        if verbose:
            print(
                f"[INFO] visual_mode={visual_mode!r} — RTX/NGX disabled "
                f"(no --enable_cameras). Safe for cluster headless."
            )
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        warnings.warn(
            "rtx_rgb + CUDA_VISIBLE_DEVICES may cause NGX errors. Try: unset CUDA_VISIBLE_DEVICES",
            stacklevel=2,
        )

    from cluster_camera_utils import configure_headless_cameras

    configure_headless_cameras(args_cli, verbose=verbose)


def lidar_range_image_shape() -> tuple[int, int]:
    """(vertical_channels, horizontal_bins) for default LidarPatternCfg."""
    h_span = LIDAR_H_FOV[1] - LIDAR_H_FOV[0]
    n_h = int(np.ceil(h_span / LIDAR_HORIZONTAL_RES)) - 1  # 360° excludes overlap
    return LIDAR_CHANNELS, n_h


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    """Ensure depth is 2D (H, W). RayCasterCamera returns (H, W, 1)."""
    d = np.asarray(depth, dtype=np.float32)
    while d.ndim > 2 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {d.shape}")
    return d


def depth_to_rgb(depth: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
    """Convert (H,W) or (H,W,1) depth to uint8 RGB via simple jet-like colormap."""
    d = _squeeze_depth(depth)
    d = np.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=max_depth)
    d = np.clip(d, 0.0, max_depth)
    inv = 1.0 - d / max_depth

    r = np.clip(1.5 * inv - 0.5, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(inv - 0.5) * 3.0, 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - inv) - 0.5, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def lidar_ranges_to_rgb(
    ranges: np.ndarray,
    valid: np.ndarray | None = None,
    out_size: tuple[int, int] = (224, 224),
    max_depth: float = 10.0,
) -> np.ndarray:
    """Reshape merged LiDAR ranges (N,) to range image and colormap to RGB."""
    v_ch, h_bins = lidar_range_image_shape()
    n = v_ch * h_bins
    r = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if r.size < n:
        r = np.pad(r, (0, n - r.size), constant_values=max_depth)
    elif r.size > n:
        r = r[:n]

    img = r.reshape(v_ch, h_bins)
    if valid is not None:
        v = np.asarray(valid, dtype=bool).reshape(-1)[:n].reshape(v_ch, h_bins)
        img = np.where(v, img, max_depth)

    from PIL import Image

    rgb = depth_to_rgb(img, max_depth=max_depth)
    pil = Image.fromarray(rgb)
    pil = pil.resize((out_size[1], out_size[0]), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def merge_depth_maps_multi(
    depth_list: list[np.ndarray],
    max_d: float,
) -> np.ndarray:
    """Per-pixel closest depth across k (H,W) depth maps."""
    if not depth_list:
        raise ValueError("depth_list is empty")
    merged = np.full(_squeeze_depth(depth_list[0]).shape, np.inf, dtype=np.float32)
    for depth in depth_list:
        d = _squeeze_depth(depth)
        valid = np.isfinite(d) & (d > 1e-4) & (d < max_d * 0.999)
        take = valid & (d < merged)
        merged[take] = d[take]
    merged[~np.isfinite(merged)] = max_d
    return merged


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def build_depth_camera_cfgs(
    mesh_paths: tuple[str, ...],
    *,
    img_res: tuple[int, int],
    update_period_s: float,
    patterns_mod,
    ray_caster_camera_cfg_cls,
    offset_pos: tuple[float, float, float] = (0.3, 0.0, 0.5),
    offset_rot: tuple[float, float, float, float] = (0.0, 0.924, 0.0, 0.383),
) -> list[tuple[str, Any]]:
    """One RayCasterCamera per mesh (Isaac only supports one mesh per caster)."""
    height, width = img_res
    pattern = patterns_mod.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=height,
        width=width,
    )
    cfgs: list[tuple[str, Any]] = []
    for i, mesh_path in enumerate(mesh_paths):
        name = f"depth_cam_{i}"
        cfg = ray_caster_camera_cfg_cls(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            mesh_prim_paths=[mesh_path],
            update_period=update_period_s,
            offset=ray_caster_camera_cfg_cls.OffsetCfg(
                pos=offset_pos,
                rot=offset_rot,
                convention="ros",
            ),
            debug_vis=False,
            pattern_cfg=pattern,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="max",
            max_distance=10.0,
        )
        cfgs.append((name, cfg))
    return cfgs


def build_tiled_camera_cfg(
    *,
    img_res: tuple[int, int],
    update_period_s: float,
    sim_utils,
    camera_cfg_cls,
):
    """RTX TiledCamera — fallback; prefer build_rtx_camera_cfg for 3DGS lab."""
    from cluster_camera_utils import build_tiled_camera_cfg as _build

    return _build(
        img_res=img_res,
        update_period_s=update_period_s,
        sim_utils=sim_utils,
        camera_cfg_cls=camera_cfg_cls,
    )


def build_rtx_camera_cfg(
    *,
    img_res: tuple[int, int],
    update_period_s: float,
    sim_utils,
    camera_cfg_cls,
):
    """RTX CameraCfg — same as DataCollection_loop (3DGS lab reference path)."""
    from cluster_camera_utils import build_rtx_camera_cfg as _build

    return _build(
        img_res=img_res,
        update_period_s=update_period_s,
        sim_utils=sim_utils,
        camera_cfg_cls=camera_cfg_cls,
    )
