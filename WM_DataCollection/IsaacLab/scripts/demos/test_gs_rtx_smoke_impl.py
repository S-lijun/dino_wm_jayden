"""Smoke test: one GS lab RGB frame via rtx_rgb.

Run via:
  bash scripts/demos/run_gs_rgb_capture.sh --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test GS lab RTX RGB capture.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--out", type=str, default="/tmp/gs_rtx_smoke.png")
parser.add_argument(
    "--visual_mode",
    type=str,
    default="rtx_rgb",
    choices=["rtx_rgb"],
    help=argparse.SUPPRESS,
)
args_cli = parser.parse_args()

if not getattr(args_cli, "rendering_mode", None):
    args_cli.rendering_mode = "performance"

from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.camera import CameraCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
import isaaclab.sim as sim_utils

from cluster_camera_utils import build_rtx_camera_cfg
from lab_scene_utils import load_lab_scene_usd, rotate_sensor_ccw_to_landscape

IMG_RES = (640, 480)


def main() -> None:
    load_lab_scene_usd()

    env_cfg = G1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1000
    env_cfg.curriculum = None
    env_cfg.scene.camera = build_rtx_camera_cfg(
        img_res=IMG_RES,
        update_period_s=1.0 / 15.0,
        sim_utils=sim_utils,
        camera_cfg_cls=CameraCfg,
    )

    print("[INFO] Creating env (rtx_rgb + CameraCfg + lab.usda)...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    env.reset()
    robot = env.scene["robot"]
    n_act = robot.data.joint_pos.shape[1]
    action = torch.zeros(1, n_act, device=env.device)
    for _ in range(30):
        env.step(action)

    camera = env.scene["camera"]
    rgb_tensor = camera.data.output["rgb"][0]
    rgb_np = rgb_tensor[..., :3].detach().cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    rgb_np = rotate_sensor_ccw_to_landscape(rgb_np)

    import imageio

    out_path = os.path.abspath(args_cli.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.imwrite(out_path, rgb_np)
    print(f"[OK] Saved GS RGB smoke frame: {out_path} shape={rgb_np.shape}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
