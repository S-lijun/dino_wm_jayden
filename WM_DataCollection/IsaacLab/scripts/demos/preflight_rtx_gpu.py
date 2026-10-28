#!/usr/bin/env python3
"""Preflight: Isaac Sim rtx_rgb needs Vulkan ray_tracing (RTX-class GPU).

H100 / A100 datacenter GPUs expose CUDA but NOT Vulkan RT → gpu.foundation.plugin
reports "No device could be created" and PhysX follows with "no suitable CUDA GPU".

Exit 0 = OK to try rtx_rgb.  Exit 1 = use depth_rgb + offline GS renderer instead.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Substrings matched against nvidia-smi GPU product name (visible devices only).
NO_VULKAN_RT_GPUS = (
    "H100",
    "H200",
    "GH200",
    "A100",
    "A800",
)

ISAAC_RT_ERROR_HINT = """
[FATAL] This GPU cannot run Isaac Sim RTX / 3DGS camera (Vulkan ray_tracing unsupported).

Isaac error you saw:
  gpu.foundation.plugin: No device could be created
  → GPUs do not support RayTracing (Vulkan ray_tracing)
  → omni.gpu_foundation_factory: Failed to create any GPU devices
  → omni.physx.plugin: CUDA libs present, but no suitable CUDA GPU

Why H100 still fails: CUDA ≠ RTX rendering. H100 is compute-only for Isaac's RT pipeline.

What works on compute2 H100:
  1) Collect sim + poses (no RTX):
       bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless --visual_mode depth_rgb
  2) Render true GS lab RGB offline (gsplat on CUDA):
       bash scripts/demos/run_gs_rgb_offline.sh --dataset_dir data/<run_id>

Or run rtx_rgb on a workstation / node with GeForce RTX or RTX A6000/A40.
Override this check (not recommended on H100): GS_FORCE_RTX=1
"""


def _visible_gpu_names() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[WARN] nvidia-smi failed: {exc}")
        return []

    rows: list[tuple[int, str]] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx_s, name = line.split(",", 1)
        rows.append((int(idx_s.strip()), name.strip()))

    if not cvd:
        return [name for _, name in rows]

    allowed = {int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()}
    return [name for idx, name in rows if idx in allowed]


def main() -> int:
    if os.environ.get("GS_FORCE_RTX", "").strip() in ("1", "true", "yes"):
        print("[WARN] GS_FORCE_RTX set — skipping RTX GPU preflight (may crash).")
        return 0

    names = _visible_gpu_names()
    if not names:
        print("[WARN] Could not read GPU names; continuing (Isaac may still fail).")
        return 0

    blocked = [n for n in names if any(tag in n for tag in NO_VULKAN_RT_GPUS)]
    if blocked:
        print(ISAAC_RT_ERROR_HINT)
        print(f"[INFO] Visible GPU(s): {names}")
        print(f"[INFO] Blocked for rtx_rgb: {blocked}")
        return 1

    print(f"[OK] GPU(s) may support Isaac RTX: {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
