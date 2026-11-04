"""SAC path training on cat(DINO z_vision, SpUNet z_lidar, proprio).

Same collector, replay buffer, critic fusion, and SAC updates as
``train_HJ_humanoid_sac_path_lidar.py``. This file only selects the fused
observation:

  z_vision = DINOv2 CLS from checkpoints/model_latest.pth (384-D)
  z_lidar  = frozen ScanNet SpUNet (96-D)
  z        = cat(z_vision, z_lidar, z_proprio)
  Q        = MLP(cat(z, (vx, vy, yaw_rate)))

Pass ``--visual_feature patch`` to use all DINO patch tokens instead of CLS.
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
    if extra:
        sys.argv[1:1] = extra


if __name__ == "__main__":
    _inject_defaults()
    target = Path(__file__).resolve().parent / "train_HJ_humanoid_sac_path_lidar.py"
    runpy.run_path(str(target), run_name="__main__")
