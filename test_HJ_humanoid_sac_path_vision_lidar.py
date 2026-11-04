"""Eval for cat(DINO CLS, SpUNet, proprio).

Same rollout, metrics, and controllers as ``test_HJ_humanoid_sac_path_lidar.py``.
This file only selects the fused observation used in training:

  z = cat(DINOv2 CLS 384, SpUNet 96, proprio 10)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _inject_defaults() -> None:
    extra: list[str] = []
    if "--obs_mode" not in sys.argv:
        extra += ["--obs_mode", "wm_spunet"]
    if "--visual_feature" not in sys.argv:
        extra += ["--visual_feature", "cls"]
    if "--lidar_pattern" not in sys.argv:
        extra += ["--lidar_pattern", "mid360"]
    if extra:
        sys.argv[1:1] = extra


if __name__ == "__main__":
    _inject_defaults()
    target = Path(__file__).resolve().parent / "test_HJ_humanoid_sac_path_lidar.py"
    runpy.run_path(str(target), run_name="__main__")
