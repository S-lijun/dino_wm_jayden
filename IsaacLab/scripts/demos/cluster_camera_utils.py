"""Helpers for RGB camera capture on headless Isaac Lab (HPC / Slurm)."""

from __future__ import annotations

import os
import warnings

DEFAULT_HEADLESS_CAMERA_KIT_ARGS = (
    "--/rtx/post/dlss/enabled=false "
    "--/rtx-transient/dldenoiser/enabled=false "
    "--/rtx-transient/dlssg/enabled=false "
    "--/rtx/domeLight/upperLowerStrategy=4"
)

# H100 shared nodes: single-GPU Vulkan, RaytracedLighting, no NGX/denoiser/aftermath.
HPC_GS_RGB_KIT_ARGS = (
    DEFAULT_HEADLESS_CAMERA_KIT_ARGS
    + " "
    + "--/rtx/rendermode=RaytracedLighting "
    + "--/rtx/reflections/denoiser/enabled=false "
    + "--/rtx/indirectDiffuse/denoiser/enabled=false "
    + "--/rtx/ambientOcclusion/enabled=false "
    + "--/renderer/multiGpu/enabled=false "
    + "--/renderer/multiGpu/autoEnable=false "
    + "--/renderer/multiGpu/maxGpuCount=1 "
    + "--/app/vulkan/enableValidationLayers=false "
    + "--/exts/omni.gpucompute.plugin/cudaDeviceMask=0"
)


def cameras_requested(args_cli) -> bool:
    if getattr(args_cli, "enable_cameras", False):
        return True
    return os.environ.get("ENABLE_CAMERAS", "0") == "1"


def configure_headless_cameras(args_cli, *, verbose: bool = True) -> None:
    """Apply HPC settings before ``AppLauncher(args_cli)``."""
    if not cameras_requested(args_cli):
        if verbose:
            print("[INFO] RGB disabled (no --enable_cameras).")
        return

    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True

    if getattr(args_cli, "headless", False) and not getattr(args_cli, "rendering_mode", None):
        args_cli.rendering_mode = "performance"
        #args_cli.rendering_mode = None

    use_hpc_gs = getattr(args_cli, "hpc_gs_rgb", True)
    extra = HPC_GS_RGB_KIT_ARGS if use_hpc_gs else DEFAULT_HEADLESS_CAMERA_KIT_ARGS
    
    existing = getattr(args_cli, "kit_args", "") or ""
    args_cli.kit_args = (existing + " " + extra).strip()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd or "," in cvd:
        warnings.warn(
            "For rtx_rgb on shared H100 nodes, pin ONE free GPU before launch, e.g. "
            "GS_GPU=3 bash scripts/demos/run_gs_rgb_capture.sh --smoke",
            stacklevel=2,
        )
    elif verbose:
        print(f"[INFO] Using single GPU CUDA_VISIBLE_DEVICES={cvd!r}")

    if verbose:
        print(
            f"[INFO] rtx_rgb headless: rendering_mode="
            f"{getattr(args_cli, 'rendering_mode', None)!r}, "
            f"hpc_kit={use_hpc_gs}"
        )
        print(f"[INFO] kit_args: {args_cli.kit_args}")


def build_rtx_camera_cfg(
    *,
    img_res: tuple[int, int],
    update_period_s: float,
    sim_utils,
    camera_cfg_cls,
):
    """CameraCfg — same as DataCollection_loop.py (3DGS lab)."""
    height, width = img_res
    return camera_cfg_cls(
        prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
        update_period=update_period_s,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=camera_cfg_cls.OffsetCfg(
            pos=(0.3, 0.0, 0.5),
            rot=(0.0, 0.924, 0.0, 0.383),
            convention="ros",
        ),
    )


def build_tiled_camera_cfg(
    *,
    img_res: tuple[int, int],
    update_period_s: float,
    sim_utils,
    camera_cfg_cls,
):
    height, width = img_res
    return camera_cfg_cls(
        prim_path="{ENV_REGEX_NS}/Robot/head_link/front_camera",
        update_period=update_period_s,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=camera_cfg_cls.OffsetCfg(
            pos=(0.3, 0.0, 0.5),
            rot=(0.0, 0.924, 0.0, 0.383),
            convention="ros",
        ),
    )
