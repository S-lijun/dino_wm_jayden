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
# Saved frames match wm_ckpt_18-27-17 (img_h, img_w). Full sensor frames are ~6x larger.
SAVE_IMG_H = 192
SAVE_IMG_W = 256

parser = argparse.ArgumentParser(description="Collect humanoid G1 offline trajectories.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_episodes", type=int, default=3000)
parser.add_argument(
    "--max_steps",
    type=int,
    default=8000,
    help="Sim-step cap. start→goal paths are several meters; 8000 steps ≈ 40s at dt=0.005.",
)
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
    default="rtx_rgb",
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
    help="Unused. Bins are resampled beside the start-goal line each episode.",
)
parser.add_argument("--arena_x_min", type=float, default=-1.0)
parser.add_argument("--arena_x_max", type=float, default=6.0)
parser.add_argument("--arena_y_min", type=float, default=-4.0)
parser.add_argument("--arena_y_max", type=float, default=2.0)
parser.add_argument(
    "--perp_offset",
    type=float,
    default=1.5,
    help="Outer |offset| (m) of bins from the start-goal line.",
)
parser.add_argument(
    "--perp_keepout",
    type=float,
    default=0.0,
    help=(
        "Inner keep-out (m). Bin |offset| ~ U[this, perp_offset]. "
        "0 lets a bin sit on the start-goal line, directly ahead of the robot."
    ),
)
parser.add_argument("--min_start_goal_dist", type=float, default=5.0)
parser.add_argument("--max_n_obstacles", type=int, default=2)
parser.add_argument("--obstacle_absent_prob", type=float, default=0.1)
parser.add_argument("--two_obstacle_prob", type=float, default=0.5)
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
parser.add_argument(
    "--save_video",
    action="store_true",
    help="Write videos/episode_XXXX.mp4 at --camera_fps and a first-frame PNG.",
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
import torch.nn.functional as F
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


def _resize_uint8_hwc(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(T, H, W, C) uint8 -> (T, height, width, C) uint8."""
    if frames.ndim != 4:
        raise ValueError(f"expected (T,H,W,C), got {tuple(frames.shape)}")
    if int(frames.shape[1]) == height and int(frames.shape[2]) == width:
        return frames
    x = frames.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)


def _frame_uint8(frame: torch.Tensor) -> np.ndarray:
    arr = frame.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.max(arr)) <= 1.0 else 1.0
        arr = (arr * scale).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _save_episode_preview(frames: torch.Tensor, video_dir: Path, ep: int, fps: float) -> None:
    """15 Hz frames -> first PNG plus an mp4 that can be opened directly."""
    if frames.shape[0] == 0:
        return
    video_dir.mkdir(parents=True, exist_ok=True)
    images = [_frame_uint8(frames[i]) for i in range(int(frames.shape[0]))]
    png_path = video_dir / f"episode_{ep:04d}_first.png"
    from PIL import Image

    Image.fromarray(images[0]).save(png_path)
    mp4_path = video_dir / f"episode_{ep:04d}.mp4"
    import imageio.v2 as imageio

    imageio.mimsave(str(mp4_path), images, fps=max(1, int(round(fps))))
    print(f"[INFO] preview png={png_path} video={mp4_path} frames={len(images)}")


def collect_episodes(num_episodes: int, max_steps: int, output_dir: str, seed: int):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    obses_dir = output_path / "obses"
    if args_cli.save_full_obses:
        obses_dir.mkdir(exist_ok=True)
    obses_15fps_dir = output_path / OBSES_15FPS_DIRNAME
    video_dir = output_path / "videos"
    if _visual_mode != "off":
        obses_15fps_dir.mkdir(exist_ok=True)
    if args_cli.save_video and _visual_mode != "off":
        video_dir.mkdir(exist_ok=True)

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
            f"[EP {ep}] {reset_info.get('waypoint_region_names')} "
            f"start={reset_info.get('robot_xy')} goal={reset_info.get('goal_xy')} "
            f"bins={reset_info.get('n_obstacles')} side={reset_info.get('bin_side')} "
            f"obstacles={reset_info.get('obstacle_positions')}"
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
        if _visual_mode != "off":
            idx_15 = subsample_frame_indices(
                len(episode_obs), wrapper.sim_dt, args_cli.camera_fps
            )
            obs_15 = torch.stack([episode_obs[i] for i in idx_15]).cpu()
            obs_15 = _resize_uint8_hwc(obs_15, SAVE_IMG_H, SAVE_IMG_W)
            torch.save(obs_15, obses_15fps_dir / f"episode_{ep}.pth")
            if args_cli.save_video:
                _save_episode_preview(obs_15, video_dir, ep, args_cli.camera_fps)
        if args_cli.save_full_obses:
            full = _resize_uint8_hwc(torch.stack(episode_obs).cpu(), SAVE_IMG_H, SAVE_IMG_W)
            torch.save(full, obses_dir / f"episode_{ep}.pth")
        # Drop the full-resolution frames before the next episode.
        episode_obs.clear()

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
    padded_states = torch.zeros(len(all_states), max_len, state_dim)
    padded_costs = torch.zeros(len(all_costs), max_len)

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
            "visual_size": (SAVE_IMG_H, SAVE_IMG_W),
            "sensor_img_res": wrapper.img_res,
            "control_fps": 1.0 / wrapper.sim_dt,
            "visual_fps": args_cli.camera_fps,
            "save_full_obses": args_cli.save_full_obses,
            "save_video": bool(args_cli.save_video),
            "obses_15fps_dir": OBSES_15FPS_DIRNAME,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "stuck_contact_steps": args_cli.stuck_contact_steps,
            "waypoint_stop_thresh": args_cli.waypoint_stop_thresh,
            "max_speed": args_cli.max_speed,
            "scene": "start_goal",
            "arena_xy": (
                args_cli.arena_x_min,
                args_cli.arena_x_max,
                args_cli.arena_y_min,
                args_cli.arena_y_max,
            ),
            "perp_offset": args_cli.perp_offset,
            "perp_keepout": args_cli.perp_keepout,
            "min_start_goal_dist": args_cli.min_start_goal_dist,
            "max_n_obstacles": args_cli.max_n_obstacles,
            "obstacle_absent_prob": args_cli.obstacle_absent_prob,
            "two_obstacle_prob": args_cli.two_obstacle_prob,
            "bin_side": "random_left_or_right",
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
