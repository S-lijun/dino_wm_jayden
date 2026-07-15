"""Smoke test for LatentHumanoidEnv (Isaac G1 + DINO-WM encode interface)."""

import argparse
import os
import sys

# Repo root for env.isaac imports.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ISAACLAB_ROOT = os.path.join(REPO_ROOT, "IsaacLab")
if ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, ISAACLAB_ROOT)

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test LatentHumanoidEnv.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/storage1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/cargoal",
    help="DINO-WM checkpoint directory (encoder subfolder appended).",
)
parser.add_argument("--dino_encoder", type=str, default="dino")
parser.add_argument("--with_proprio", action="store_true")
parser.add_argument("--num_steps", type=int, default=10)
parser.add_argument("--num_resets", type=int, default=5)
parser.add_argument(
    "--visual_mode",
    type=str,
    default="depth_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
)
# --device is already registered by AppLauncher.add_app_launcher_args()
args_cli, _ = parser.parse_known_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from omegaconf import OmegaConf

from plan import load_model
from env.isaac.latent_humanoid_env import LatentHumanoidEnv


class _Args:
    seed = 0
    device = torch_device


def main():
    ckpt_root = os.path.join(args_cli.dino_ckpt_dir, args_cli.dino_encoder)
    hydra_cfg = os.path.join(ckpt_root, "hydra.yaml")
    snapshot = os.path.join(ckpt_root, "checkpoints", "model_latest.pth")

    if not os.path.isfile(hydra_cfg) or not os.path.isfile(snapshot):
        print(f"[WARN] WM checkpoint missing at {ckpt_root}; skipping encode test.")
        print("[INFO] Testing IsaacG1Wrapper only.")
        from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper

        wrapper = IsaacG1Wrapper(args_cli, visual_mode=_visual_mode)
        for r in range(args_cli.num_resets):
            info = wrapper.reset_scene(seed=r)
            print(f"[RESET {r}] active={info['active_obstacles']}, waypoint={info['waypoint']}")
            for t in range(args_cli.num_steps):
                action = np.random.uniform(-0.3, 0.3, size=3).astype(np.float32)
                wrapper.apply_velocity_command(action)
                diag = wrapper.get_safety_diagnostics()
                print(
                    f"  step={t} h_s={diag['h_s']:.1f} "
                    f"lidar_min={diag['lidar_min_distance']:.3f} "
                    f"lidar_xy={diag['lidar_min_distance_xy']:.3f} "
                    f"contact={diag['contact_collision']:.0f} "
                    f"lidar_unsafe={diag['lidar_unsafe']:.0f}"
                )
        simulation_app.close()
        return

    train_cfg = OmegaConf.load(hydra_cfg)
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=torch_device)
    wm.eval()

    args = _Args()
    args.device = torch_device

    env = LatentHumanoidEnv(
        args,
        wm,
        torch_device,
        args_cli,
        with_proprio=args_cli.with_proprio,
    )

    print(f"[INFO] observation_space={env.observation_space.shape}")
    print(f"[INFO] action_space={env.action_space.shape}")

    for r in range(args_cli.num_resets):
        z, info = env.reset(seed=r)
        print(
            f"[RESET {r}] z.shape={z.shape} active={info.get('active_obstacles')} "
            f"lidar_min={info.get('lidar_min_distance', float('nan')):.3f}"
        )
        for t in range(args_cli.num_steps):
            action = env.action_space.sample()
            z_next, h_s, term, trunc, step_info = env.step(action)
            diag = env.wrapper.get_safety_diagnostics()
            print(
                f"  step={t} h_s={h_s:.1f} z={z_next.shape} "
                f"lidar_min={diag['lidar_min_distance']:.3f} "
                f"contact={diag['contact_collision']:.0f} "
                f"lidar_unsafe={diag['lidar_unsafe']:.0f}"
            )
            if term or trunc:
                break

    env.close()
    simulation_app.close()
    print("[INFO] LatentHumanoidEnv smoke test complete.")


if __name__ == "__main__":
    main()
