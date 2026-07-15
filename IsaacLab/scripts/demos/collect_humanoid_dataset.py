"""Batch offline dataset collection for Isaac G1 humanoid world-model training."""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ISAACLAB_ROOT = os.path.join(REPO_ROOT, "IsaacLab")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ISAACLAB_ROOT)

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

# Matches DataCollection_test.py camera_fps / next_camera_time_s gating.
DEFAULT_CAMERA_FPS = 15.0
OBSES_15FPS_DIRNAME = "obses_15fps"

parser = argparse.ArgumentParser(description="Collect humanoid G1 offline trajectories.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=3000)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.environ.get(
        "DATASET_DIR",
        "/storage1/sibai/Active/ihab/research_new/datasets_dino",
    )
    + "/humanoid_g1",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--img_height",
    type=int,
    default=640,
    help="Sensor height in pixels (portrait 640×480 before CCW rotation → landscape PNG).",
)
parser.add_argument(
    "--img_width",
    type=int,
    default=480,
    help="Sensor width in pixels.",
)
parser.add_argument(
    "--visual_mode",
    type=str,
    default="depth_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
)
parser.add_argument(
    "--stuck_contact_steps",
    type=int,
    default=50,
    help="End episode if any link has contact force above threshold for this many consecutive steps.",
)
parser.add_argument(
    "--waypoint_stop_thresh",
    type=float,
    default=0.1,
    help="Distance (m) to current region waypoint before advancing (matches DataCollection_loop_test).",
)
parser.add_argument(
    "--max_speed",
    type=float,
    default=0.5,
    help="Nominal planar speed (m/s) for waypoint tracking.",
)
parser.add_argument(
    "--bin_x",
    type=float,
    default=2.0,
    help="Blue bin world X position (m).",
)
parser.add_argument(
    "--bin_y",
    type=float,
    default=0.0,
    help="Blue bin world Y position (m).",
)
parser.add_argument(
    "--camera_fps",
    type=float,
    default=DEFAULT_CAMERA_FPS,
    help="Sim-time rate for obses_15fps/ subsampling (matches DataCollection_test camera_fps).",
)
parser.add_argument(
    "--save_full_obses",
    action="store_true",
    help="Also save per-control-step obses/ (default: only obses_15fps/).",
)
args_cli, _ = parser.parse_known_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from pathlib import Path

from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper
from env.isaac.waypoint_utils import WaypointNavController


def subsample_frame_indices(num_frames: int, sim_dt: float, camera_fps: float) -> list[int]:
    """Indices into a per-step frame list sampled on uniform sim-time (like DataCollection_test)."""
    if num_frames <= 0:
        return []
    period_s = 1.0 / camera_fps
    indices: list[int] = []
    next_camera_time_s = 0.0
    for step in range(num_frames):
        sim_time_s = step * sim_dt
        if sim_time_s + 1e-12 >= next_camera_time_s:
            indices.append(step)
            next_camera_time_s += period_s
    return indices


def collect_episodes(num_episodes: int, max_steps: int, output_dir: str, seed: int):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    obses_dir = output_path / "obses"
    if args_cli.save_full_obses:
        obses_dir.mkdir(exist_ok=True)
    obses_15fps_dir = output_path / OBSES_15FPS_DIRNAME
    if _visual_mode != "off":
        obses_15fps_dir.mkdir(exist_ok=True)

    wrapper = IsaacG1Wrapper(
        args_cli,
        visual_mode=_visual_mode,
        img_res=(args_cli.img_height, args_cli.img_width),
        stuck_contact_steps=args_cli.stuck_contact_steps,
        waypoint_stop_thresh=args_cli.waypoint_stop_thresh,
        max_speed=args_cli.max_speed,
    )
    nav = WaypointNavController(
        max_speed=args_cli.max_speed,
        stop_thresh=args_cli.waypoint_stop_thresh,
    )
    rng = np.random.default_rng(seed)

    all_actions = []
    all_states = []
    all_costs = []
    seq_lengths = []
    n_finished_all_waypoints = 0
    n_stuck = 0
    n_max_steps = 0

    for ep in range(num_episodes):
        ep_seed = int(rng.integers(0, 2**31 - 1))
        reset_info = wrapper.reset_scene(seed=ep_seed)
        nav.reset()

        print(
            f"[EP {ep}] regions={reset_info.get('waypoint_region_names')} "
            f"waypoints={reset_info.get('waypoints')} "
            f"blue_bin={reset_info.get('obstacle_positions', {}).get('blue_bin_0')} "
            f"robot_xy={reset_info.get('robot_xy')}"
        )

        episode_actions = []
        episode_states = []
        episode_costs = []
        episode_obs = []
        end_reason = "max_steps"

        for step in range(max_steps):
            obs = wrapper.get_raw_obs()
            episode_obs.append(torch.from_numpy(obs["visual"]))

            robot = wrapper.env.unwrapped.scene["robot"]
            base_pos = robot.data.root_pos_w[0].cpu().numpy()
            base_quat = robot.data.root_quat_w[0].cpu().numpy()
            cmd = nav.compute_command(base_pos, base_quat, wrapper.waypoint)
            episode_actions.append(torch.from_numpy(cmd))
            _, _, _, step_info = wrapper.apply_velocity_command(cmd)

            # Post-step labels (matches DataCollection_test.py after env.step).
            episode_states.append(torch.from_numpy(wrapper.get_full_state()))
            episode_costs.append(torch.tensor(wrapper.calculate_cost(), dtype=torch.float32))

            if wrapper.advance_waypoint_if_reached():
                end_reason = "all_waypoints"
                break

            if step_info.get("stuck", False):
                end_reason = "stuck"
                break

        if end_reason == "all_waypoints":
            n_finished_all_waypoints += 1
        elif end_reason == "stuck":
            n_stuck += 1
        else:
            n_max_steps += 1

        if len(episode_actions) == 0:
            continue

        all_actions.append(torch.stack(episode_actions))
        all_states.append(torch.stack(episode_states))
        all_costs.append(torch.stack(episode_costs))
        seq_lengths.append(len(episode_actions))
        if args_cli.save_full_obses:
            torch.save(torch.stack(episode_obs).cpu(), obses_dir / f"episode_{ep}.pth")
        if _visual_mode != "off":
            idx_15 = subsample_frame_indices(
                len(episode_obs), wrapper.sim_dt, args_cli.camera_fps
            )
            obs_15 = torch.stack([episode_obs[i] for i in idx_15]).cpu()
            torch.save(obs_15, obses_15fps_dir / f"episode_{ep}.pth")

        if (ep + 1) % 50 == 0:
            print(
                f"[INFO] Collected {ep + 1}/{num_episodes} episodes "
                f"(finished={n_finished_all_waypoints}, stuck={n_stuck}, max_steps={n_max_steps})"
            )

    if not all_actions:
        raise RuntimeError("No episodes collected — check Isaac env / sensors.")

    max_len = max(seq_lengths)
    action_dim = all_actions[0].shape[-1]
    state_dim = all_states[0].shape[-1]

    padded_actions = torch.zeros(len(all_actions), max_len, action_dim)
    padded_states = torch.zeros(len(all_actions), max_len, state_dim)
    padded_costs = torch.zeros(len(all_actions), max_len)

    for i, (actions, states, costs) in enumerate(zip(all_actions, all_states, all_costs)):
        length = len(actions)
        padded_actions[i, :length] = actions
        padded_states[i, :length] = states
        padded_costs[i, :length] = costs

    torch.save(padded_actions, output_path / "actions.pth")
    torch.save(padded_states, output_path / "states.pth")
    torch.save(torch.tensor(seq_lengths), output_path / "seq_lengths.pth")
    torch.save(padded_costs, output_path / "costs.pth")
    torch.save(
        {
            "proprio_dim": wrapper.proprio_dim,
            "visual_size": wrapper.visual_output_size,
            "sensor_img_res": wrapper.img_res,
            "control_fps": 1.0 / wrapper.sim_dt,
            "visual_fps": args_cli.camera_fps,
            "save_full_obses": args_cli.save_full_obses,
            "obses_15fps_dir": OBSES_15FPS_DIRNAME,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "stuck_contact_steps": args_cli.stuck_contact_steps,
            "waypoint_stop_thresh": args_cli.waypoint_stop_thresh,
            "max_speed": args_cli.max_speed,
            "blue_bin_xy": (args_cli.bin_x, args_cli.bin_y),
            "state_lidar_min_field": "lidar_min_range_nonzero_m",
            "end_reason_counts": {
                "all_waypoints": n_finished_all_waypoints,
                "stuck": n_stuck,
                "max_steps": n_max_steps,
            },
        },
        output_path / "meta.pth",
    )

    print(f"[INFO] Saved {len(all_actions)} episodes to {output_path}")
    print(
        f"[INFO] End reasons: all_waypoints={n_finished_all_waypoints}, "
        f"stuck={n_stuck}, max_steps={n_max_steps}"
    )


def main():
    collect_episodes(
        args_cli.num_episodes,
        args_cli.max_steps,
        args_cli.output_dir,
        args_cli.seed,
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
