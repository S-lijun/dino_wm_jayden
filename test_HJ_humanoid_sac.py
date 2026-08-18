"""Test SAC HJ safety filter on Isaac G1 latent humanoid.

Default: each trial freezes one scene and runs three controllers
  - safe_only (SF only)
  - waypoint_only (WaypointNavController)
  - switching (Q-gate)
then writes 3 videos + one overlay traj PNG.

Test waypoint layout (not used in training):
  start → front → left|right sampled in danger disk (bin center, r=0.3)
  → back disk at (5.5, 0) r=1.0
  Every 5 trials, exactly one has no blue bin; other sampling is unchanged.

Usage::

  python test_HJ_humanoid_sac.py --headless --visual_mode rtx_rgb \\
    --dino_ckpt_dir /workspace --dino_encoder wm_ckpt_18-27-17 --with_proprio \\
    --policy_path runs/sac_hj_humanoid/.../epoch_id_N/policy.pth \\
    --num_runs 5

  # WM dynamics look-ahead: gate on V(z_{t+1}) instead of Q(z_t, a_nom)
  python test_HJ_humanoid_sac.py ... --look_ahead
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ISAACLAB_ROOT = os.path.join(REPO_ROOT, "IsaacLab")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ISAACLAB_ROOT)

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

COMPARE_MODES = ("safe_only", "waypoint_only", "switching")
MODE_LABELS = {
    "safe_only": "SF only",
    "waypoint_only": "Waypoint only",
    "switching": "Switching",
}

parser = argparse.ArgumentParser("Test SAC HJ filter on latent Humanoid (Isaac G1)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/workspace",
    help="Parent of encoder folder",
)
parser.add_argument(
    "--dino_encoder",
    type=str,
    default="wm_ckpt_18-27-17",
)
parser.add_argument(
    "--config",
    type=str,
    default="train_HJ_configs.yaml",
)
parser.add_argument("--with_proprio", action="store_true")
parser.add_argument(
    "--visual_mode",
    type=str,
    default="rtx_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
)
parser.add_argument(
    "--policy_path",
    type=str,
    required=True,
    help="Path to SAC training checkpoint .../epoch_id_N/policy.pth",
)
parser.add_argument(
    "--mode",
    type=str,
    default="compare",
    choices=["compare", "switching", "waypoint_only", "safe_only"],
    help="compare (default) = SF / waypoint / switching on the same scene.",
)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument(
    "--no_bin_every",
    type=int,
    default=5,
    help=(
        "In each block of this many trials, exactly one has no blue bin "
        "(waypoints / spawn otherwise unchanged). 0 disables."
    ),
)
parser.add_argument(
    "--safety_threshold",
    type=float,
    default=0.0,
    help=(
        "Switch to SF when the gate value < this. Default gate is "
        "min(Q1,Q2)(z_t, a_nom). With --look_ahead, gate is V(z_{t+1}) "
        "= min(Q1,Q2)(z_hat, pi(z_hat)) from the WM predictor."
    ),
)
parser.add_argument(
    "--look_ahead",
    action="store_true",
    help=(
        "Use WM predictor look-ahead: gate and SF actor both use predicted "
        "z_{t+1} given a_nom, not current z_t. Gate is V(z_hat); SF is pi(z_hat)."
    ),
)
parser.add_argument("--max_visual_steps", type=int, default=400)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--save_video", action="store_true", default=True)
parser.add_argument("--no_save_video", action="store_true")
parser.add_argument(
    "--goal_radius",
    type=float,
    default=0.1,
    help="Waypoint stop / goal success radius (m).",
)
parser.add_argument(
    "--pass_radius",
    type=float,
    default=0.3,
    help="Danger pass disk radius around obstacle for left|right vias (m).",
)
parser.add_argument(
    "--back_center_x",
    type=float,
    default=5.5,
    help="Back-region disk center x.",
)
parser.add_argument(
    "--back_center_y",
    type=float,
    default=-2.0,
    help="Back-region disk center y (aisle at y=-2).",
)
parser.add_argument(
    "--back_radius",
    type=float,
    default=1.0,
    help="Back-region disk radius (m).",
)

args_cli, remaining = parser.parse_known_args()

sys.path.insert(0, os.path.join(ISAACLAB_ROOT, "scripts/demos"))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, REPO_ROOT)

import gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from PyHJ.policy import avoid_SACPolicy_annealing
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import ActorProb, Critic

from wm_load import load_model
from env.isaac.latent_humanoid_env import LatentHumanoidEnv


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def get_args_and_merge_config():
    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    args = argparse.Namespace(**vars(args_cli))
    for key, val in vars(cfg_args).items():
        setattr(args, key.replace("-", "_"), val)
    return args


def _visual_to_uint8_hwc(visual) -> np.ndarray:
    if isinstance(visual, torch.Tensor):
        arr = visual.detach().cpu().numpy()
    else:
        arr = np.asarray(visual)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC visual, got {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr[..., :3]


def _video_frame(env: LatentHumanoidEnv, hj_val: float | None, l_val: float | None) -> np.ndarray:
    hwc = _visual_to_uint8_hwc(env.wrapper.get_raw_obs()["visual"])
    return env._overlay_metrics_hwc(hwc, hj_val, l_val)


def build_policy(env: LatentHumanoidEnv, args, device: str):
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    policy_action_space = gym.spaces.Box(
        low=np.asarray(env.action_space.low, dtype=np.float32),
        high=np.asarray(env.action_space.high, dtype=np.float32),
        dtype=np.float32,
    )

    def _make_critic():
        net = Net(
            state_shape,
            action_shape,
            hidden_sizes=args.critic_net,
            activation=getattr(torch.nn, args.critic_activation),
            concat=True,
            device=device,
        )
        critic = Critic(net, device=device).to(device)
        optim = torch.optim.AdamW(critic.parameters(), lr=float(args.critic_lr))
        return critic, optim

    critic1, critic1_optim = _make_critic()
    critic2, critic2_optim = _make_critic()

    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=device,
    )
    actor1 = ActorProb(
        actor_net, action_shape, max_action=1.0, unbounded=True, device=device
    ).to(device)
    actor1_optim = torch.optim.AdamW(actor1.parameters(), lr=float(args.actor_lr))

    policy = avoid_SACPolicy_annealing(
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=float(args.tau),
        gamma=float(args.gamma_pyhj),
        alpha=0.2,
        exploration_noise=None,
        deterministic_eval=True,
        reward_normalization=bool(args.rew_norm),
        estimation_step=int(args.n_step),
        action_space=policy_action_space,
        actor1=actor1,
        actor1_optim=actor1_optim,
    )
    policy.critic = policy.critic1
    return policy


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--policy_path not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys (ok if alpha-only): {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    policy.eval()
    policy.actor1.eval()
    policy.critic1.eval()
    policy.critic2.eval()
    print(f"[INFO] Loaded SAC policy from {ckpt_path}")


@torch.no_grad()
def q_value(policy, z: np.ndarray, act_policy: np.ndarray, device: str) -> float:
    """min(Q1, Q2)(z, a) in policy space."""
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    a_t = torch.as_tensor(act_policy, dtype=torch.float32, device=device)
    if a_t.ndim == 1:
        a_t = a_t.unsqueeze(0)
    q1 = policy.critic1(z_t, a_t)
    q2 = policy.critic2(z_t, a_t)
    q = torch.min(q1, q2)
    return float(q.reshape(-1)[0].item())


@torch.no_grad()
def v_value(policy, z: np.ndarray, device: str) -> float:
    """V(z) = min(Q1, Q2)(z, pi(z)) with the deterministic SAC actor."""
    from PyHJ.data import Batch

    was_training = policy.training
    policy.eval()
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    act_pol = policy(Batch(obs=z_t, info=Batch())).act
    if isinstance(act_pol, torch.Tensor):
        a_t = act_pol
    else:
        a_t = torch.as_tensor(act_pol, dtype=torch.float32, device=device)
    if a_t.ndim == 1:
        a_t = a_t.unsqueeze(0)
    q1 = policy.critic1(z_t, a_t)
    q2 = policy.critic2(z_t, a_t)
    q = torch.min(q1, q2)
    policy.train(was_training)
    return float(q.reshape(-1)[0].item())


@torch.no_grad()
def safe_action_env(policy, z: np.ndarray, device: str) -> np.ndarray:
    """Deterministic SAC actor (mean) → env-space (vx, vy, yaw free)."""
    from PyHJ.data import Batch

    was_training = policy.training
    policy.eval()
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    batch = Batch(obs=z_t, info=Batch())
    act_pol = policy(batch).act
    if isinstance(act_pol, torch.Tensor):
        act_pol = act_pol.detach().cpu().numpy()
    act_pol = np.asarray(act_pol, dtype=np.float32).reshape(-1)
    act_env = np.asarray(policy.map_action(act_pol), dtype=np.float32).reshape(3)
    policy.train(was_training)
    return act_env


def _robot_xy(env: LatentHumanoidEnv) -> np.ndarray:
    return np.asarray(env.wrapper.get_robot_xy_local(), dtype=np.float64).reshape(2)


def _capture_layout(env: LatentHumanoidEnv) -> dict:
    layout = getattr(env, "_last_reset_layout", None)
    if layout is None:
        raise RuntimeError("No reset layout captured; call env.reset() first.")
    return {
        "blue_bin_xy": (
            float(layout["blue_bin_xy"][0]),
            float(layout["blue_bin_xy"][1]),
        ),
        "obstacle_present": bool(layout["obstacle_present"]),
        "waypoints": np.asarray(layout["waypoints"], dtype=np.float64).copy(),
        "waypoint_region_names": list(layout["waypoint_region_names"]),
        "robot_xy": np.asarray(layout["robot_xy"], dtype=np.float64).copy(),
    }


def _no_bin_run_ids(
    num_runs: int, every: int, rng: np.random.Generator
) -> set[int]:
    """Pick exactly one no-bin trial in each block of ``every`` runs."""
    if every <= 0 or num_runs <= 0:
        return set()
    ids: set[int] = set()
    for start in range(0, int(num_runs), int(every)):
        block = np.arange(start, min(start + int(every), int(num_runs)))
        ids.add(int(rng.choice(block)))
    return ids


def configure_test_waypoints(
    env: LatentHumanoidEnv,
    *,
    pass_radius: float,
    back_center_x: float,
    back_center_y: float,
    back_radius: float,
    goal_radius: float,
) -> None:
    """Eval layout: start→front→danger left|right@bin→back disk."""
    bin_xy = env.wrapper._blue_bin_xy_fixed
    env.wrapper.include_middle_pass = False
    env.wrapper.alternate_left_right = True
    env.wrapper.obstacle_absent_prob = 0.0
    env.wrapper.randomize_obstacle = False
    env.wrapper.y_bound = 0.0
    # Allow reaching back disk (~5.5±1); training wall at 4.5 would clip early.
    env.wrapper.x_bound_max = float(back_center_x + back_radius + 1.0)
    env.wrapper.trajectory_region_sequence = [
        "start",
        "front",
        ("left", "right"),
        "back",
    ]
    danger = {
        "center": np.array([float(bin_xy[0]), float(bin_xy[1])], dtype=np.float64),
        "r": float(pass_radius),
    }
    env.wrapper.trajectory_regions["left"] = {
        "center": danger["center"].copy(),
        "r": float(pass_radius),
    }
    env.wrapper.trajectory_regions["right"] = {
        "center": danger["center"].copy(),
        "r": float(pass_radius),
    }
    env.wrapper.trajectory_regions["back"] = {
        "center": np.array(
            [float(back_center_x), float(back_center_y)], dtype=np.float64
        ),
        "r": float(back_radius),
    }
    env.wrapper.waypoint_stop_thresh = float(goal_radius)
    env.waypoint_nav.stop_thresh = float(goal_radius)
    print(
        f"[INFO] test waypoints: sequence={env.wrapper.trajectory_region_sequence} "
        f"danger_pass=bin{tuple(bin_xy)} r={pass_radius} "
        f"back=({back_center_x},{back_center_y}) r={back_radius} "
        f"goal_radius={goal_radius} x_bound_max={env.wrapper.x_bound_max}"
    )


def _save_compare_traj(
    *,
    out_path: Path,
    mode_trajs: dict[str, np.ndarray],
    mode_ends: dict[str, str],
    waypoints: np.ndarray,
    obstacle_xy: np.ndarray,
    pass_radius: float,
    goal_radius: float,
    run_id: int,
    obstacle_present: bool = True,
) -> None:
    """Overlay SF / waypoint / switching paths on one top-down PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    wps = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    goal_xy = wps[-1]
    start_xy = wps[0]
    vias = wps[1:-1] if wps.shape[0] > 2 else wps[1:]

    style = {
        "safe_only": {"color": "#1f77b4", "lw": 1.8, "z": 3},
        "waypoint_only": {"color": "0.45", "lw": 1.6, "z": 2},
        "switching": {"color": "#d62728", "lw": 1.8, "z": 4},
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    for mode in COMPARE_MODES:
        traj = mode_trajs.get(mode)
        if traj is None or len(traj) == 0:
            continue
        traj = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
        st = style[mode]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            "-",
            color=st["color"],
            linewidth=st["lw"],
            zorder=st["z"],
            label=MODE_LABELS[mode],
        )

    ax.scatter(
        float(start_xy[0]),
        float(start_xy[1]),
        c="green",
        s=45,
        zorder=5,
        label="start",
    )
    if vias.size:
        ax.scatter(
            vias[:, 0],
            vias[:, 1],
            c="cyan",
            marker="x",
            s=55,
            zorder=5,
            label="vias",
        )
    if obstacle_present:
        ax.scatter(
            float(obstacle_xy[0]),
            float(obstacle_xy[1]),
            c="blue",
            s=70,
            zorder=5,
            label="obstacle",
        )
        ax.add_patch(
            Circle(
                (float(obstacle_xy[0]), float(obstacle_xy[1])),
                float(pass_radius),
                fill=False,
                edgecolor="blue",
                linestyle=":",
                linewidth=1.2,
                label=f"pass region r={pass_radius:.2f}",
            )
        )
    ax.add_patch(
        Circle(
            (float(goal_xy[0]), float(goal_xy[1])),
            float(goal_radius),
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.2,
            label=f"goal r={goal_radius:.2f}",
        )
    )
    ax.scatter(
        float(goal_xy[0]),
        float(goal_xy[1]),
        c="red",
        s=70,
        marker="*",
        zorder=6,
        label="goal",
    )

    ends = ", ".join(
        f"{MODE_LABELS[m]}={mode_ends.get(m, '?')}" for m in COMPARE_MODES
    )
    bin_tag = "bin" if obstacle_present else "no bin"
    ax.set_title(f"run{run_id:02d} | {bin_tag} | {ends}", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Saved compare traj: {out_path}")


def simulate_one(
    env: LatentHumanoidEnv,
    policy,
    *,
    mode: str,
    device: str,
    safety_threshold: float,
    max_visual_steps: int,
    save_video: bool,
    out_dir: Path,
    run_id: int,
    fixed_layout: dict | None = None,
    seed: int | None = None,
    look_ahead: bool = False,
) -> dict:
    if fixed_layout is None:
        z, info = env.reset(seed=seed)
        layout = _capture_layout(env)
    else:
        z, info = env.reset(seed=seed, options={"fixed_layout": fixed_layout})
        layout = fixed_layout

    frames: list[np.ndarray] = []
    traj = [_robot_xy(env)]
    h_s_last = float(env.wrapper.calculate_cost())
    hj_exec = env._hj_value(z)
    if save_video:
        frames.append(_video_frame(env, hj_exec, h_s_last))

    hj_interventions = 0
    total_switches = 0
    constraint_violations = 0
    min_hj_nom = float("inf")
    last_controller = None
    end_reason = None

    for step in range(max_visual_steps):
        a_nom_env = env.compute_waypoint_nav_action()
        a_nom_pol = np.asarray(
            policy.map_action_inverse(a_nom_env), dtype=np.float32
        ).reshape(-1)

        if look_ahead:
            z_hat = env.predict_next_latent(a_nom_env)
            q_nom = v_value(policy, z_hat, device)
            z_sf = z_hat
            gate_name = "V(z_hat)"
        else:
            q_nom = q_value(policy, z, a_nom_pol, device)
            z_sf = z
            gate_name = "Q(a_nom)"

        if mode == "waypoint_only":
            action = a_nom_env
        elif mode == "safe_only":
            action = safe_action_env(policy, z_sf, device)
            hj_interventions += 1
        else:  # switching
            if q_nom < safety_threshold:
                action = safe_action_env(policy, z_sf, device)
                hj_interventions += 1
                if last_controller == "waypoint":
                    total_switches += 1
                last_controller = "hj"
                if step < 30 or hj_interventions <= 5:
                    print(
                        f"  step {step}: SF intervene {gate_name}={q_nom:.3f} "
                        f"a_nom={a_nom_env} a_sf={action}"
                    )
            else:
                action = a_nom_env
                if last_controller == "hj":
                    total_switches += 1
                last_controller = "waypoint"

        min_hj_nom = min(min_hj_nom, q_nom)

        z, h_s, terminated, truncated, info = env.step(action)
        traj.append(_robot_xy(env))
        h_s_last = float(h_s)
        if h_s < 0:
            constraint_violations += 1

        if save_video:
            hj_exec = env._hj_value(
                z, action_env=np.asarray(action, dtype=np.float64)
            )
            frames.append(_video_frame(env, hj_exec, h_s_last))

        if terminated or truncated:
            code = int(info.get("end_reason", 0))
            end_reason = {
                0: "ongoing",
                1: "all_waypoints",
                2: "stuck",
                3: "max_steps",
                4: "out_of_bounds",
            }.get(code, f"code_{code}")
            break
    else:
        end_reason = "max_visual_steps"

    video_path = None
    if save_video and frames:
        try:
            import imageio.v2 as imageio

            video_path = str(out_dir / f"{mode}_run{run_id:02d}.mp4")
            imageio.mimsave(video_path, frames, fps=int(round(env.visual_fps)))
            print(f"[INFO] Saved video: {video_path} ({len(frames)} frames)")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] video save failed: {exc}")

    success = end_reason == "all_waypoints"
    result = {
        "mode": mode,
        "run_id": run_id,
        "end_reason": end_reason,
        "success": success,
        "visual_steps": int(info.get("episode_step", step + 1)),
        "hj_interventions": hj_interventions,
        "total_switches": total_switches,
        "constraint_violations": constraint_violations,
        "min_q_nom": min_hj_nom if np.isfinite(min_hj_nom) else None,
        "final_h_s": h_s_last,
        "stuck": float(info.get("stuck", 0.0)),
        "video": video_path,
        "traj": np.asarray(traj, dtype=np.float64),
        "layout": layout,
    }
    print(
        f"[RUN {run_id}] mode={mode} end={end_reason} success={success} "
        f"steps={result['visual_steps']} interventions={hj_interventions} "
        f"violations={constraint_violations} minQ={result['min_q_nom']}"
    )
    return result


def main():
    args = get_args_and_merge_config()
    args.device = torch_device
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    save_video = bool(args.save_video) and not bool(args.no_save_video)

    np.random.seed(int(getattr(args, "seed", 0)))
    torch.manual_seed(int(getattr(args, "seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(getattr(args, "seed", 0)))

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("humanoid_test_sac") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = list(COMPARE_MODES) if args.mode == "compare" else [args.mode]
    look_ahead = bool(getattr(args, "look_ahead", False))
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] modes={modes} policy={args.policy_path}")
    gate_desc = (
        "gate=V(z_t+1) from WM predictor" if look_ahead else "gate=Q(z_t, a_nom)"
    )
    print(f"[INFO] look_ahead={look_ahead} ({gate_desc})")

    ckpt_dir = Path(args.dino_ckpt_dir)
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
    if not hydra_cfg.is_file():
        raise FileNotFoundError(f"WM hydra.yaml not found: {hydra_cfg}")
    train_cfg = OmegaConf.load(str(hydra_cfg))
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=args.device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    env = LatentHumanoidEnv(
        args,
        wm,
        args.device,
        args_cli,
        with_proprio=bool(args.with_proprio),
        latent_h=False,
        wandb_video_every=0,
    )
    configure_test_waypoints(
        env,
        pass_radius=float(args.pass_radius),
        back_center_x=float(args.back_center_x),
        back_center_y=float(args.back_center_y),
        back_radius=float(args.back_radius),
        goal_radius=float(args.goal_radius),
    )
    policy = build_policy(env, args, args.device)
    load_policy_checkpoint(policy, args.policy_path, args.device)
    env.policy_for_log = policy

    results = []
    base_seed = int(getattr(args, "seed", 0))
    no_bin_every = int(getattr(args, "no_bin_every", 5))
    no_bin_ids = _no_bin_run_ids(
        int(args.num_runs),
        no_bin_every,
        np.random.default_rng(base_seed + 17),
    )
    if no_bin_ids:
        pretty = ", ".join(str(i + 1) for i in sorted(no_bin_ids))
        print(
            f"[INFO] no blue bin on trial(s) {pretty} "
            f"(1 per {no_bin_every} trials; waypoints otherwise unchanged)"
        )
    else:
        print("[INFO] blue bin present on every trial")

    for run_id in range(int(args.num_runs)):
        no_bin = run_id in no_bin_ids
        # Sample spawn / waypoints as usual (bin always present in RNG), then
        # optionally hide the bin so the rest of the trial is unchanged.
        env.wrapper.obstacle_absent_prob = 0.0
        trial_seed = base_seed + run_id
        env.reset(seed=trial_seed)
        layout = _capture_layout(env)
        if no_bin:
            layout["obstacle_present"] = False
        print(
            f"\n=== Trial {run_id + 1}/{args.num_runs} modes={modes} "
            f"blue_bin={'absent' if no_bin else 'present'} ==="
        )
        mode_trajs: dict[str, np.ndarray] = {}
        mode_ends: dict[str, str] = {}
        for mode in modes:
            r = simulate_one(
                env,
                policy,
                mode=mode,
                device=args.device,
                safety_threshold=float(args.safety_threshold),
                max_visual_steps=int(args.max_visual_steps),
                save_video=save_video,
                out_dir=out_dir,
                run_id=run_id,
                fixed_layout=layout,
                seed=trial_seed,
                look_ahead=look_ahead,
            )
            if mode == modes[0]:
                print(
                    f"[INFO] frozen layout bin={layout['blue_bin_xy']} "
                    f"obstacle_present={layout['obstacle_present']} "
                    f"waypoints={layout['waypoints'].tolist()} "
                    f"names={layout['waypoint_region_names']}"
                )
            results.append(r)
            mode_trajs[mode] = r["traj"]
            mode_ends[mode] = r["end_reason"]

        if args.mode == "compare" and layout is not None and len(modes) > 1:
            _save_compare_traj(
                out_path=out_dir / f"run{run_id:02d}_compare_traj.png",
                mode_trajs=mode_trajs,
                mode_ends=mode_ends,
                waypoints=layout["waypoints"],
                obstacle_xy=np.asarray(layout["blue_bin_xy"], dtype=np.float64),
                pass_radius=float(args.pass_radius),
                goal_radius=float(args.goal_radius),
                run_id=run_id,
                obstacle_present=bool(layout["obstacle_present"]),
            )

    # Per-mode summary
    lines = [
        f"policy={args.policy_path}",
        f"algo=SAC",
        f"eval_mode={args.mode}",
        f"look_ahead={look_ahead}",
        f"safety_threshold={float(args.safety_threshold)}",
        f"modes={modes}",
        f"pass_radius={float(args.pass_radius)}",
        f"back=({float(args.back_center_x)},{float(args.back_center_y)}) "
        f"r={float(args.back_radius)}",
        f"goal_radius={float(args.goal_radius)}",
        f"trials={int(args.num_runs)}",
        f"no_bin_every={no_bin_every} no_bin_trials="
        f"{sorted(i + 1 for i in no_bin_ids)}",
        "",
    ]
    for mode in modes:
        rs = [r for r in results if r["mode"] == mode]
        n = len(rs)
        n_ok = sum(1 for r in rs if r["success"])
        n_stuck = sum(1 for r in rs if r["end_reason"] == "stuck")
        n_oob = sum(1 for r in rs if r["end_reason"] == "out_of_bounds")
        mean_int = float(np.mean([r["hj_interventions"] for r in rs])) if n else 0.0
        mean_viol = (
            float(np.mean([r["constraint_violations"] for r in rs])) if n else 0.0
        )
        lines.extend(
            [
                f"[{MODE_LABELS.get(mode, mode)}]",
                f"  success={n_ok}/{n} ({100.0 * n_ok / max(n, 1):.1f}%)",
                f"  stuck={n_stuck} out_of_bounds={n_oob}",
                f"  mean_hj_interventions={mean_int:.2f}",
                f"  mean_constraint_violations={mean_viol:.2f}",
            ]
        )
        for r in rs:
            lines.append(
                f"  run{r['run_id']:02d}: end={r['end_reason']} success={r['success']} "
                f"bin={'absent' if not r['layout']['obstacle_present'] else 'present'} "
                f"steps={r['visual_steps']} int={r['hj_interventions']} "
                f"viol={r['constraint_violations']} minQ={r['min_q_nom']}"
            )
        lines.append("")

    text = "\n".join(lines)
    summary_path = out_dir / (
        "summary_compare.txt" if args.mode == "compare" else f"summary_{args.mode}.txt"
    )
    summary_path.write_text(text, encoding="utf-8")
    print("\n" + "=" * 60)
    print(text)
    print(f"[INFO] Wrote {summary_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
