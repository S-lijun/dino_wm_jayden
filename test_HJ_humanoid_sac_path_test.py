"""Train-collect replica of test_HJ_humanoid_sac_path.py.

Does not replace test_HJ_humanoid_sac_path.py. Env / a_nom / h_s / critic
flags match the CLS+concat train command. You only set waypoints and bins.

Eval a_nom uses fixed |v|=0.5 and no (vx,vy,yaw) coverage noise.
Pass --nom_act_noise to add N(0, --nom_act_noise_std). SF action is eval tanh(μ).

Custom layout (4 waypoints required; 0/1/2 bins)::

  --start_xy 0,0 --via1_xy 2,0 --via2_xy 4,-0.25 --goal_xy 5.5,0 \\
    --bin_xy 4,0
  --bin_xy 4,0 --bin_xy 4,-2
  --bin_xy none

Multiple scenes in one launch (repeat --scene; each is start via1 via2 goal [bin...])::

  --scene 0,0 2,0 4.0,0.0 5.5,0 4,0 \\
  --scene 0,0 2,0 4.0,0.5 5.5,0 4,0 \\
  --scene 0,0 2,0 4.0,0.0 5.5,0 none

Usage::

    python test_HJ_humanoid_sac_path_test.py --headless --visual_mode rtx_rgb \\
      --dino_ckpt_dir C:\\dino_wm_jayden --dino_encoder wm_ckpt_18-27-17 \\
      --policy_path runs/sac_hj_humanoid_path/.../epoch_id_N/policy.pth \\
      --start_xy 0,0 --via1_xy 2,0 --via2_xy 4,-0.25 --goal_xy 5.5,0 \\
      --bin_xy 4,0
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

COMPARE_MODES = ("waypoint_only", "safe_only", "switching")
MODE_LABELS = {
    "safe_only": "SF only",
    "waypoint_only": "Waypoint only",
    "switching": "Switching",
}
# start=0, trans1=1, trans2=2, goal=3
THIRD_WAYPOINT_IDX = 2

# Easy straight-line eval: start → wp2 → (wp3 within 1 m of bin) → goal.
# Bin is fixed on the aisle; wp3 is sampled inside a 1 m disk (not the 1.5 m
# h_s danger radius used for plotting / non-easy eval).
EASY_START_REGION = {"center": (0.0, -2.0), "r": 1.0}
EASY_WP2_REGION = {"center": (2.0, -2.0), "r": 1.5}
EASY_BIN_XY = (4.5, -2.0)
EASY_GOAL_REGION = {"center": (5.5, -2.0), "r": 1.0}
EASY_WP3_MAX_DIST = 1.0

parser = argparse.ArgumentParser(
    "Train-collect replica test for SAC HJ path filter"
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--dino_ckpt_dir", type=str, default=REPO_ROOT)
parser.add_argument("--dino_encoder", type=str, default="wm_ckpt_18-27-17")
parser.add_argument(
    "--visual_feature",
    type=str,
    default="cls",
    choices=["patch", "cls"],
    help="Must match training (this script defaults to cls).",
)
parser.add_argument(
    "--critic_fusion",
    type=str,
    default="concat",
    choices=["late", "concat"],
    help="Must match training (this script defaults to concat).",
)
parser.add_argument(
    "--z_fuse_frames",
    type=int,
    default=1,
    help=(
        "Average the last N full z vectors (visual + proprio). "
        "Must match training. 1 = single-frame z. Default 1."
    ),
)
parser.add_argument("--config", type=str, default="train_HJ_configs.yaml")
parser.add_argument(
    "--with_proprio",
    action=argparse.BooleanOptionalAction,
    default=True,
)
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
    help=(
        "SAC path checkpoint: .../epoch_id_N/policy.pth, or the epoch_id_N "
        "directory (policy.pth is inferred)."
    ),
)
parser.add_argument(
    "--mode",
    type=str,
    default="compare",
    choices=["compare", "switching", "waypoint_only", "safe_only"],
    help="compare (default) = waypoint / SF / switching on the same scene.",
)
parser.add_argument("--num_runs", type=int, default=1)
parser.add_argument(
    "--no_bin_every",
    type=int,
    default=2,
    help=(
        "In each block of this many trials, exactly one has no bin "
        "(waypoints otherwise sampled the same way). 0 = always one bin. "
        "Ignored with --easy."
    ),
)
parser.add_argument(
    "--easy",
    action="store_true",
    help=(
        "Easy straight-line layout: start disk (0,-2) r=1, wp2 disk (2,-2) r=1.5, "
        "fixed bin (4.5,-2), wp3 within 1 m of the bin, goal disk (5.5,-2) r=1. "
        "Always one bin; ignores --no_bin_every. Mutually exclusive with custom XY flags."
    ),
)
parser.add_argument(
    "--start_xy",
    type=str,
    default=None,
    metavar="X,Y",
    help="Custom start XY as 'x,y'. Requires --via1_xy --via2_xy --goal_xy.",
)
parser.add_argument(
    "--via1_xy",
    type=str,
    default=None,
    metavar="X,Y",
    help="Custom first via (trans1) as 'x,y'.",
)
parser.add_argument(
    "--via2_xy",
    type=str,
    default=None,
    metavar="X,Y",
    help="Custom second via (trans2) as 'x,y'.",
)
parser.add_argument(
    "--goal_xy",
    type=str,
    default=None,
    metavar="X,Y",
    help="Custom goal XY as 'x,y'.",
)
parser.add_argument(
    "--bin_xy",
    action="append",
    default=None,
    metavar="X,Y",
    help=(
        "Blue-bin XY as 'x,y'. Repeat for a second bin, or 'x1,y1;x2,y2'. "
        "'none' = no bin. Only valid with custom waypoints. "
        "Omit: no bin. Ignored if --scene is set."
    ),
)
parser.add_argument(
    "--scene",
    action="append",
    nargs="+",
    default=None,
    metavar="XY",
    help=(
        "One eval scene: start via1 via2 goal [bin [bin2]]. Repeat the flag "
        "to queue more scenes; this process runs them in order (one trial "
        "each). Example: --scene 0,0 2,0 4,0 5.5,0 4,0. Use 'none' for no "
        "bin. Mutually exclusive with --start_xy / --easy. Do not use ';' "
        "inside the spec (PowerShell treats it as a new statement)."
    ),
)
parser.add_argument(
    "--spawn_yaw",
    type=float,
    default=None,
    help="Spawn yaw in radians. Default: heading from start toward via1.",
)
parser.add_argument(
    "--safety_threshold",
    type=float,
    default=0.0,
    help="Switch to SF when min(Q1,Q2)(z, a_nom) < this.",
)
parser.add_argument(
    "--look_ahead",
    action="store_true",
    help="Gate/SF on WM-predicted z_{t+1} instead of current z_t.",
)
parser.add_argument("--max_visual_steps", type=int, default=800)
parser.add_argument("--max_episode_steps", type=int, default=20000)
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
    "--danger_radius",
    type=float,
    default=1.0,
    help="Plot / failure-set disk (m). Default 1 to match h_s = d_min-1.",
)
parser.add_argument(
    "--lidar_distance_threshold",
    type=float,
    default=1.0,
    help="h_s = lidar_d_min - this. Dashed obstacle circle uses the same radius.",
)
parser.add_argument("--lidar_h_half_fov_deg", type=float, default=60.0)
parser.add_argument("--include_contact_in_hs", action="store_true", default=True)
parser.add_argument("--no_include_contact_in_hs", action="store_true")
parser.add_argument("--contact_hs", type=float, default=-2.5)
parser.add_argument("--yaw_limit", type=float, default=1.0)
parser.add_argument(
    "--freeze_yaw",
    action="store_true",
    default=False,
    help=(
        "Force yaw (dim 2) to 0 on the SF action only; a_nom keeps its yaw. "
        "Q(a_sf) is probed on the same zeroed action. Default: off."
    ),
)
parser.add_argument("--y_bound", type=float, default=0.0)
parser.add_argument("--use_arena_bounds", action="store_true", default=True)
parser.add_argument("--arena_x_min", type=float, default=-1.0)
parser.add_argument("--arena_x_max", type=float, default=6.0)
parser.add_argument("--arena_y_min", type=float, default=-4.0)
parser.add_argument("--arena_y_max", type=float, default=2.0)
parser.add_argument("--waypoint_layout", type=str, default="start_goal_perp")
parser.add_argument("--perp_offset", type=float, default=2.5)
parser.add_argument("--min_start_goal_dist", type=float, default=4.0)
parser.add_argument("--path_obstacle_layout", action="store_true", default=True)
parser.add_argument("--max_n_obstacles", type=int, default=2)
parser.add_argument("--obstacle_absent_prob", type=float, default=0.1)
parser.add_argument("--two_obstacle_prob", type=float, default=0.5)
parser.add_argument("--skip_arena_oob_from_buffer", action="store_true", default=True)
parser.add_argument(
    "--randomize_nom_speed",
    action="store_true",
    default=False,
    help="Eval default off: |v|=0.5. On: Uniform[min,max] once per episode.",
)
parser.add_argument("--no_randomize_nom_speed", action="store_true")
parser.add_argument("--nom_speed_min", type=float, default=0.05)
parser.add_argument("--nom_speed_max", type=float, default=0.95)
parser.add_argument("--nom_speed_mean", type=float, default=0.5)
parser.add_argument("--nom_speed_std", type=float, default=0.3)
parser.add_argument(
    "--nom_act_noise",
    action="store_true",
    default=False,
    help="Off by default. On: N(0, --nom_act_noise_std) on (vx,vy,yaw) after a_nom.",
)
parser.add_argument("--no_nom_act_noise", action="store_true")
parser.add_argument("--nom_act_noise_std", type=float, default=0.1)
parser.add_argument("--a_good_hold_steps", type=int, default=20)

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
from PyHJ.utils.net.continuous import ActorProb

from wm_load import load_model
from env.isaac.latent_humanoid_env import LatentHumanoidEnv
from env.isaac.waypoint_utils import (
    _point_in_rect,
    sample_perp_points_on_segment,
    sample_point_in_region,
    sample_start_goal_perp_path,
)
from env.isaac.late_fusion_critic import make_q_critic
from env.isaac.switch_traj_plot import _plot_colored_traj


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
        name = key.replace("-", "_")
        # Keep CLI path-eval flags; fill the rest from train_HJ_configs.yaml.
        if not hasattr(args_cli, name):
            setattr(args, name, val)
    if bool(getattr(args, "no_include_contact_in_hs", False)):
        args.include_contact_in_hs = False
    if bool(getattr(args, "no_randomize_nom_speed", False)):
        args.randomize_nom_speed = False
    if bool(getattr(args, "no_nom_act_noise", False)):
        args.nom_act_noise = False
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
        critic = make_q_critic(
            getattr(args, "critic_fusion", "late"),
            state_shape,
            action_shape,
            hidden_sizes=args.critic_net,
            activation=getattr(torch.nn, args.critic_activation),
            device=device,
        )
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


def resolve_policy_path(ckpt_path: str | Path) -> Path:
    """Accept policy.pth, or an epoch_id_N directory that contains it."""
    raw = str(ckpt_path).strip().strip('"').strip("'")
    path = Path(raw).expanduser()
    search = [path]
    if not path.is_absolute():
        search.append(Path(REPO_ROOT) / path)

    tried: list[Path] = []
    for cand in search:
        tried.append(cand)
        if cand.is_file():
            return cand
        pth = cand / "policy.pth"
        tried.append(pth)
        if pth.is_file():
            return pth

    pretty = "\n".join(f"  {p}" for p in tried)
    raise FileNotFoundError(
        f"--policy_path not found: {ckpt_path}\n"
        "Pass .../epoch_id_N/policy.pth or the epoch_id_N folder.\n"
        f"Tried:\n{pretty}"
    )


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str, critic_fusion: str = "late") -> None:
    ckpt_path = resolve_policy_path(ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    fusion = str(critic_fusion).lower()
    late_missing = [k for k in missing if "critic" in k and "z_mlp" in k]
    concat_as_late = any("preprocess" in k for k in unexpected)
    late_as_concat = any("z_mlp" in k for k in unexpected) or any(
        "preprocess" in k and "critic" in k for k in missing
    )
    if fusion == "late" and (late_missing or concat_as_late):
        raise RuntimeError(
            "This checkpoint is concat critic (cat(z, a)), not LateFusionCritic. "
            "Pass --critic_fusion concat --visual_feature cls (matching training). "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if fusion == "concat" and late_as_concat:
        raise RuntimeError(
            "This checkpoint is LateFusionCritic, not concat. "
            "Drop --critic_fusion concat (default late + patch). "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    if missing:
        print(f"[WARN] missing keys (ok if alpha-only): {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected}")
    policy.eval()
    policy.actor1.eval()
    policy.critic1.eval()
    policy.critic2.eval()
    print(f"[INFO] Loaded SAC policy from {ckpt_path}")


FREEZE_YAW = False


def _zero_yaw_act(act: np.ndarray) -> np.ndarray:
    """Force yaw (dim 2) to 0 when --freeze_yaw is on. No-op otherwise.

    Policy- and env-space yaw share a symmetric range, so 0 maps to 0 in both.
    """
    out = np.asarray(act, dtype=np.float32)
    if not FREEZE_YAW or out.ndim < 1 or out.shape[-1] < 3:
        return out
    out = out.copy()
    out[..., 2] = 0.0
    return out


@torch.no_grad()
def q_value(policy, z: np.ndarray, act_policy: np.ndarray, device: str) -> float:
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    a_t = torch.as_tensor(act_policy, dtype=torch.float32, device=device)
    if a_t.ndim == 1:
        a_t = a_t.unsqueeze(0)
    q = torch.min(policy.critic1(z_t, a_t), policy.critic2(z_t, a_t))
    return float(q.reshape(-1)[0].item())


@torch.no_grad()
def v_value(policy, z: np.ndarray, device: str) -> float:
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
    q = torch.min(policy.critic1(z_t, a_t), policy.critic2(z_t, a_t))
    policy.train(was_training)
    return float(q.reshape(-1)[0].item())


@torch.no_grad()
def actor_policy_and_env(policy, z: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic SAC mean: policy-space tanh(μ) and env-space action."""
    from PyHJ.data import Batch

    was_training = policy.training
    policy.eval()
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    act_pol = policy(Batch(obs=z_t, info=Batch())).act
    if isinstance(act_pol, torch.Tensor):
        act_pol = act_pol.detach().cpu().numpy()
    act_pol = _zero_yaw_act(np.asarray(act_pol, dtype=np.float32).reshape(-1))
    act_env = _zero_yaw_act(
        np.asarray(policy.map_action(act_pol), dtype=np.float32).reshape(3)
    )
    policy.train(was_training)
    return act_pol, act_env


@torch.no_grad()
def safe_action_env(policy, z: np.ndarray, device: str) -> np.ndarray:
    _, act_env = actor_policy_and_env(policy, z, device)
    return act_env


@torch.no_grad()
def probe_action_qs(
    policy,
    z: np.ndarray,
    a_nom_pol: np.ndarray,
    device: str,
    rng: np.random.Generator,
) -> dict:
    """Same z: Q(a_nom) vs Q(a_sf) vs Q(uniform[-1,1]). Policy-space actions."""
    a_sf_pol, a_sf_env = actor_policy_and_env(policy, z, device)
    a_rand_pol = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
    z_norm = float(np.linalg.norm(np.asarray(z, dtype=np.float64).reshape(-1)))
    return {
        "q_nom": q_value(policy, z, a_nom_pol, device),
        "q_sf": q_value(policy, z, a_sf_pol, device),
        "q_rand": q_value(policy, z, a_rand_pol, device),
        "z_norm": z_norm,
        "a_sf_pol": a_sf_pol,
        "a_sf_env": a_sf_env,
    }


def _robot_xy(env: LatentHumanoidEnv) -> np.ndarray:
    return np.asarray(env.wrapper.get_robot_xy_local(), dtype=np.float64).reshape(2)


def _no_bin_run_ids(num_runs: int, every: int, rng: np.random.Generator) -> set[int]:
    if every <= 0 or num_runs <= 0:
        return set()
    ids: set[int] = set()
    for start in range(0, int(num_runs), int(every)):
        block = np.arange(start, min(start + int(every), int(num_runs)))
        ids.add(int(rng.choice(block)))
    return ids


def configure_path_eval(env: LatentHumanoidEnv, args) -> None:
    global FREEZE_YAW
    FREEZE_YAW = bool(getattr(args, "freeze_yaw", False))
    env.wrapper.waypoint_layout = "start_goal_perp"
    env.wrapper.path_obstacle_layout = True
    env.wrapper.filter_lidar_to_active_obstacles = True
    env.wrapper.include_middle_pass = False
    env.wrapper.alternate_left_right = False
    env.wrapper.obstacle_absent_prob = float(args.obstacle_absent_prob)
    env.wrapper.two_obstacle_prob = float(args.two_obstacle_prob)
    env.wrapper.y_bound = 0.0
    env.wrapper.use_arena_bounds = True
    env.wrapper.arena_x_min = float(args.arena_x_min)
    env.wrapper.arena_x_max = float(args.arena_x_max)
    env.wrapper.arena_y_min = float(args.arena_y_min)
    env.wrapper.arena_y_max = float(args.arena_y_max)
    env.wrapper.perp_offset = float(args.perp_offset)
    env.wrapper.min_start_goal_dist = float(args.min_start_goal_dist)
    env.wrapper.lidar_distance_threshold = float(args.lidar_distance_threshold)
    env.wrapper.lidar_h_half_fov_deg = float(args.lidar_h_half_fov_deg)
    env.wrapper.include_contact_in_hs = bool(args.include_contact_in_hs)
    env.wrapper.contact_hs = float(args.contact_hs)
    env.wrapper.waypoint_stop_thresh = float(args.goal_radius)
    env.waypoint_nav.stop_thresh = float(args.goal_radius)
    env.randomize_nom_speed = bool(args.randomize_nom_speed) and not bool(
        getattr(args, "no_randomize_nom_speed", False)
    )
    env.nom_speed_min = float(args.nom_speed_min)
    env.nom_speed_max = float(args.nom_speed_max)
    env.nom_speed_mean = float(args.nom_speed_mean)
    env.nom_speed_std = float(args.nom_speed_std)
    env.nom_act_noise = bool(args.nom_act_noise) and not bool(
        getattr(args, "no_nom_act_noise", False)
    )
    env.nom_act_noise_std = float(args.nom_act_noise_std)
    env.nom_act_noise_stds = []
    env.z_fuse_frames = int(getattr(args, "z_fuse_frames", 1))
    if not env.randomize_nom_speed:
        env._episode_nom_speed = 0.5
        env.waypoint_nav.max_speed = 0.5
    if not env.nom_act_noise:
        env._episode_nom_act_noise_std = 0.0
    custom = _custom_waypoints_complete(args)
    if custom:
        layout_tag = "custom"
    elif bool(getattr(args, "easy", False)):
        layout_tag = "easy-line"
    else:
        layout_tag = "start+trans1+trans2+goal"
    nom_v = (
        f"|v|~U[{float(args.nom_speed_min):g},{float(args.nom_speed_max):g}]/ep"
        if bool(args.randomize_nom_speed)
        else "|v|=0.5"
    )
    nom_a = (
        f" act+=N(0,{float(args.nom_act_noise_std):g})"
        if bool(args.nom_act_noise)
        else " act noise off"
    )
    print(
        f"[INFO] path eval: layout={layout_tag} "
        f"arena=x[{args.arena_x_min:g},{args.arena_x_max:g}] "
        f"y[{args.arena_y_min:g},{args.arena_y_max:g}] "
        f"danger_r={float(args.lidar_distance_threshold):g} "
        f"goal_r={float(args.goal_radius):g} "
        f"lidar_cone=±{float(args.lidar_h_half_fov_deg):g}° "
        f"contact_hs={float(args.contact_hs):g} "
        f"nom {nom_v}{nom_a} a_sf=eval mean "
        f"freeze_yaw={FREEZE_YAW}"
    )


def _place_bin_covering_waypoint(
    rng: np.random.Generator,
    waypoint: np.ndarray,
    *,
    danger_radius: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    start: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    min_offset: float = 0.35,
    max_tries: int = 128,
) -> np.ndarray:
    """Sample a bin XY such that ``waypoint`` is strictly inside the danger disk."""
    wp = np.asarray(waypoint, dtype=np.float64).reshape(2)
    r_hi = max(0.2, float(danger_radius) - 0.08)
    r_lo = min(float(min_offset), 0.6 * r_hi)
    last = wp.copy()
    for _ in range(int(max_tries)):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = float(np.sqrt(rng.uniform(r_lo * r_lo, r_hi * r_hi)))
        xy = wp + rad * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        last = xy
        if not _point_in_rect(xy, x_min, x_max, y_min, y_max, margin=0.2):
            continue
        if float(np.linalg.norm(xy - wp)) >= float(danger_radius) - 1e-6:
            continue
        if start is not None and float(np.linalg.norm(xy - start)) < 1.0:
            continue
        if goal is not None and float(np.linalg.norm(xy - goal)) < 1.0:
            continue
        return xy
    if not _point_in_rect(last, x_min, x_max, y_min, y_max, margin=0.05):
        last = np.array(
            [
                float(np.clip(wp[0], x_min + 0.3, x_max - 0.3)),
                float(np.clip(wp[1], y_min + 0.3, y_max - 0.3)),
            ],
            dtype=np.float64,
        )
        delta = last - wp
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-6:
            last = wp + np.array([min(0.5, 0.5 * r_hi), 0.0], dtype=np.float64)
    return last


def _sample_disk_in_arena(
    rng: np.random.Generator,
    region: dict,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_tries: int = 64,
) -> np.ndarray:
    last = sample_point_in_region(region, rng)
    for _ in range(int(max_tries)):
        pt = sample_point_in_region(region, rng)
        if _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=0.15):
            return pt
        last = pt
    return np.array(
        [
            float(np.clip(last[0], x_min + 0.2, x_max - 0.2)),
            float(np.clip(last[1], y_min + 0.2, y_max - 0.2)),
        ],
        dtype=np.float64,
    )


def _sample_in_danger_disk(
    rng: np.random.Generator,
    bin_xy: np.ndarray,
    *,
    danger_radius: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    start: np.ndarray | None = None,
    max_tries: int = 128,
) -> np.ndarray:
    """Sample a waypoint strictly inside the bin's danger disk."""
    c = np.asarray(bin_xy, dtype=np.float64).reshape(2)
    r_hi = max(0.2, float(danger_radius) - 0.08)
    r_lo = min(0.35, 0.6 * r_hi)
    last = c.copy()
    for _ in range(int(max_tries)):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        rad = float(np.sqrt(rng.uniform(r_lo * r_lo, r_hi * r_hi)))
        xy = c + rad * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        last = xy
        if not _point_in_rect(xy, x_min, x_max, y_min, y_max, margin=0.15):
            continue
        if float(np.linalg.norm(xy - c)) >= float(danger_radius) - 1e-6:
            continue
        if start is not None and float(np.linalg.norm(xy - start)) < 1.0:
            continue
        return xy
    return last


def _parse_bin_list(val) -> list[np.ndarray]:
    """Parse 0/1/2 bin XYs from CLI.

    Accepts None, 'none', 'x,y', 'x1,y1;x2,y2', or a list of those strings
    (from ``--bin_xy`` action=append).
    """
    if val is None:
        return []
    chunks: list[str] = []
    if isinstance(val, (list, tuple)):
        for item in val:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                chunks.extend(part.strip() for part in s.split(";") if part.strip())
    else:
        s = str(val).strip()
        if s:
            chunks.extend(part.strip() for part in s.split(";") if part.strip())
    out: list[np.ndarray] = []
    for chunk in chunks:
        if chunk.lower() in ("", "none", "null"):
            continue
        xy = _parse_xy(chunk)
        if xy is None:
            continue
        out.append(xy)
    if len(out) > 2:
        raise ValueError(f"at most 2 bins (training max), got {len(out)}")
    return out


def _parse_xy(val) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.size != 2:
            raise ValueError(f"expected 2 XY values, got {val!r}")
        return arr.astype(np.float64)
    s = str(val).strip().lower()
    if s in ("", "none", "null"):
        return None
    parts = str(val).replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"expected 'x,y', got {val!r}")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float64)


def _xy2(val) -> np.ndarray:
    xy = _parse_xy(val)
    if xy is None:
        raise ValueError("expected 'x,y', got empty / none")
    return xy


def _scene_flag_list(args) -> list[list[str]]:
    raw = getattr(args, "scene", None)
    if not raw:
        return []
    out: list[list[str]] = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            tokens = [str(t).strip() for t in item if str(t).strip()]
        else:
            tokens = [str(item).strip()] if str(item).strip() else []
        if tokens:
            out.append(tokens)
    return out


def _parse_scene_tokens(
    tokens: list[str], *, max_n_obstacles: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """start via1 via2 goal [bin [bin2] | none]."""
    parts = [str(t).strip() for t in tokens if str(t).strip()]
    if len(parts) == 1:
        parts = [
            p.strip()
            for p in parts[0].replace("|", " ").replace(";", " ").split()
            if p.strip()
        ]
    if len(parts) < 4:
        raise ValueError(
            "--scene needs start via1 via2 goal [bin...], got "
            + repr(" ".join(parts))
        )
    start = _xy2(parts[0])
    trans1 = _xy2(parts[1])
    trans2 = _xy2(parts[2])
    goal = _xy2(parts[3])
    bins = _parse_bin_list(parts[4:] if len(parts) > 4 else None)
    if len(bins) > int(max_n_obstacles):
        raise ValueError(
            f"--scene has {len(bins)} bins but --max_n_obstacles={max_n_obstacles}"
        )
    return start, trans1, trans2, goal, bins


def build_layout_from_waypoints(
    start,
    trans1,
    trans2,
    goal,
    bins: list[np.ndarray] | None = None,
    *,
    spawn_yaw: float | None = None,
    log: bool = True,
) -> dict:
    """Freeze start, via1, via2, goal, and 0/1/2 bins."""
    start = np.asarray(start, dtype=np.float64).reshape(2)
    trans1 = np.asarray(trans1, dtype=np.float64).reshape(2)
    trans2 = np.asarray(trans2, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    wps = np.stack([start, trans1, trans2, goal], axis=0)
    names = ["start", "trans1", "trans2", "goal"]
    heading = trans1 - start
    if float(np.linalg.norm(heading)) < 1e-6:
        heading = np.array([1.0, 0.0], dtype=np.float64)
    yaw = (
        float(spawn_yaw)
        if spawn_yaw is not None
        else float(np.arctan2(heading[1], heading[0]))
    )
    bins = list(bins or [])
    obstacle_xys = [(float(b[0]), float(b[1])) for b in bins]
    if log:
        if obstacle_xys:
            dists = [
                float(np.linalg.norm(np.asarray(xy) - trans2)) for xy in obstacle_xys
            ]
            print(
                f"[INFO] custom layout: start={start.tolist()} via1={trans1.tolist()} "
                f"via2={trans2.tolist()} goal={goal.tolist()} "
                f"bins={obstacle_xys} via2–bin={dists} "
                f"spawn_yaw={yaw:.3f}"
            )
        else:
            print(
                f"[INFO] custom layout: start={start.tolist()} via1={trans1.tolist()} "
                f"via2={trans2.tolist()} goal={goal.tolist()} bin=none "
                f"spawn_yaw={yaw:.3f}"
            )
    blue = obstacle_xys[0] if obstacle_xys else (0.0, 0.0)
    return {
        "waypoints": wps,
        "waypoint_region_names": names,
        "spawn_yaw": yaw,
        "obstacle_xys": obstacle_xys,
        "n_obstacles": int(len(obstacle_xys)),
        "obstacle_present": bool(obstacle_xys),
        "blue_bin_xy": blue,
        "robot_xy": np.asarray(start, dtype=np.float64).copy(),
    }


def _copy_layout(layout: dict) -> dict:
    return {
        **layout,
        "waypoints": np.asarray(layout["waypoints"], dtype=np.float64).copy(),
        "robot_xy": np.asarray(layout["robot_xy"], dtype=np.float64).copy(),
        "obstacle_xys": list(layout["obstacle_xys"]),
        "waypoint_region_names": list(layout["waypoint_region_names"]),
    }


def _custom_waypoint_flags(args) -> dict[str, bool]:
    return {
        "start_xy": _parse_xy(getattr(args, "start_xy", None)) is not None,
        "via1_xy": _parse_xy(getattr(args, "via1_xy", None)) is not None,
        "via2_xy": _parse_xy(getattr(args, "via2_xy", None)) is not None,
        "goal_xy": _parse_xy(getattr(args, "goal_xy", None)) is not None,
    }


def _custom_waypoints_complete(args) -> bool:
    flags = _custom_waypoint_flags(args)
    return all(flags.values())


def _validate_custom_layout_args(args) -> None:
    flags = _custom_waypoint_flags(args)
    any_wp = any(flags.values())
    all_wp = all(flags.values())
    scenes = _scene_flag_list(args)
    bins = _parse_bin_list(getattr(args, "bin_xy", None))
    bin_set = bool(bins) or (
        getattr(args, "bin_xy", None) is not None
        and str(args.bin_xy).strip() != ""
    )
    spawn_set = getattr(args, "spawn_yaw", None) is not None
    easy = bool(getattr(args, "easy", False))
    if scenes and (easy or any_wp or bin_set):
        raise ValueError(
            "--scene is mutually exclusive with --easy / --start_xy / "
            "--via1_xy / --via2_xy / --goal_xy / --bin_xy"
        )
    if easy and (any_wp or bin_set or spawn_set):
        raise ValueError(
            "--easy is mutually exclusive with --start_xy/--via1_xy/--via2_xy/"
            "--goal_xy/--bin_xy/--spawn_yaw"
        )
    if any_wp and not all_wp:
        missing = [k for k, v in flags.items() if not v]
        raise ValueError(
            "custom layout requires all four waypoints; missing "
            + ", ".join(f"--{k}" for k in missing)
        )
    if (bin_set or spawn_set) and not all_wp and not scenes:
        raise ValueError(
            "--bin_xy / --spawn_yaw require a full custom layout "
            "(--start_xy --via1_xy --via2_xy --goal_xy) or --scene"
        )
    if scenes:
        max_n = int(getattr(args, "max_n_obstacles", 2))
        for tokens in scenes:
            _parse_scene_tokens(tokens, max_n_obstacles=max_n)
        return
    if len(bins) > int(getattr(args, "max_n_obstacles", 2)):
        raise ValueError(
            f"got {len(bins)} bins but --max_n_obstacles="
            f"{int(getattr(args, 'max_n_obstacles', 2))}"
        )


def build_custom_path_eval_layout(args) -> dict:
    """Freeze start, via1, via2, goal, and 0/1/2 bins from CLI XY flags."""
    bins = _parse_bin_list(getattr(args, "bin_xy", None))
    return build_layout_from_waypoints(
        _xy2(args.start_xy),
        _xy2(args.via1_xy),
        _xy2(args.via2_xy),
        _xy2(args.goal_xy),
        bins,
        spawn_yaw=getattr(args, "spawn_yaw", None),
        log=True,
    )


def build_scene_eval_layouts(args) -> list[dict]:
    """One frozen layout per --scene (queued in CLI order)."""
    yaw = getattr(args, "spawn_yaw", None)
    max_n = int(getattr(args, "max_n_obstacles", 2))
    layouts: list[dict] = []
    for i, tokens in enumerate(_scene_flag_list(args)):
        start, trans1, trans2, goal, bins = _parse_scene_tokens(
            tokens, max_n_obstacles=max_n
        )
        print(f"[INFO] scene {i:02d}")
        layouts.append(
            build_layout_from_waypoints(
                start,
                trans1,
                trans2,
                goal,
                bins,
                spawn_yaw=yaw,
                log=True,
            )
        )
    return layouts


def sample_easy_path_eval_layout(rng: np.random.Generator, args) -> dict:
    """Straight aisle: start → wp2 → wp3 within 1 m of bin → goal, bin at (4.5,-2)."""
    x_min = float(args.arena_x_min)
    x_max = float(args.arena_x_max)
    y_min = float(args.arena_y_min)
    y_max = float(args.arena_y_max)
    wp3_r = float(EASY_WP3_MAX_DIST)
    start = _sample_disk_in_arena(
        rng, EASY_START_REGION, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    trans1 = _sample_disk_in_arena(
        rng, EASY_WP2_REGION, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    bin_xy = np.asarray(EASY_BIN_XY, dtype=np.float64).reshape(2).copy()
    trans2 = _sample_in_danger_disk(
        rng,
        bin_xy,
        danger_radius=wp3_r,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        start=start,
    )
    goal = _sample_disk_in_arena(
        rng, EASY_GOAL_REGION, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    wps = np.stack([start, trans1, trans2, goal], axis=0)
    names = ["start", "trans1", "trans2", "goal"]
    heading = trans1 - start
    if float(np.linalg.norm(heading)) < 1e-6:
        heading = np.array([1.0, 0.0], dtype=np.float64)
    spawn_yaw = float(np.arctan2(heading[1], heading[0]))
    dist = float(np.linalg.norm(bin_xy - trans2))
    print(
        f"[INFO] easy layout: bin={bin_xy.tolist()} wp3 dist={dist:.2f}m "
        f"(wp3_r={wp3_r:g}, plot_danger_r={float(args.danger_radius):g})"
    )
    return {
        "waypoints": wps,
        "waypoint_region_names": names,
        "spawn_yaw": spawn_yaw,
        "obstacle_xys": [(float(bin_xy[0]), float(bin_xy[1]))],
        "n_obstacles": 1,
        "obstacle_present": True,
        "blue_bin_xy": (float(bin_xy[0]), float(bin_xy[1])),
        "robot_xy": np.asarray(start, dtype=np.float64).copy(),
    }


def sample_path_eval_layout(
    rng: np.random.Generator,
    args,
    *,
    with_bin: bool | None = None,
    n_bins: int | None = None,
) -> dict:
    """start + 2 perp vias + goal; 0/1/2 bins on the start-goal perpendicular."""
    wps, names, spawn_yaw = sample_start_goal_perp_path(
        rng,
        x_min=float(args.arena_x_min),
        x_max=float(args.arena_x_max),
        y_min=float(args.arena_y_min),
        y_max=float(args.arena_y_max),
        perp_offset=float(args.perp_offset),
        min_start_goal_dist=float(args.min_start_goal_dist),
        margin=0.5,
    )
    if wps.shape[0] <= THIRD_WAYPOINT_IDX:
        raise RuntimeError(f"expected 4 waypoints, got {wps.shape}")
    if n_bins is None:
        if with_bin is False:
            n_bins = 0
        elif with_bin is True:
            n_bins = 1
        else:
            absent_p = float(args.obstacle_absent_prob)
            if absent_p > 0.0 and float(rng.random()) < absent_p:
                n_bins = 0
            elif float(rng.random()) < float(args.two_obstacle_prob):
                n_bins = min(2, int(args.max_n_obstacles))
            else:
                n_bins = 1
    n_bins = max(0, min(int(n_bins), int(args.max_n_obstacles), 2))
    obstacle_xys: list[tuple[float, float]] = []
    if n_bins > 0:
        pts = sample_perp_points_on_segment(
            rng,
            wps[0],
            wps[-1],
            n_bins,
            perp_offset=float(args.perp_offset),
            x_min=float(args.arena_x_min),
            x_max=float(args.arena_x_max),
            y_min=float(args.arena_y_min),
            y_max=float(args.arena_y_max),
        )
        obstacle_xys = [(float(p[0]), float(p[1])) for p in pts]
        print(
            f"[INFO] train-like bins n={n_bins} xy={obstacle_xys} "
            f"(perp band ±{float(args.perp_offset):g} m)"
        )
    blue = obstacle_xys[0] if obstacle_xys else (0.0, 0.0)
    return {
        "waypoints": np.asarray(wps, dtype=np.float64).copy(),
        "waypoint_region_names": list(names),
        "spawn_yaw": float(spawn_yaw),
        "obstacle_xys": obstacle_xys,
        "n_obstacles": int(len(obstacle_xys)),
        "obstacle_present": bool(obstacle_xys),
        "blue_bin_xy": blue,
        "robot_xy": np.asarray(wps[0], dtype=np.float64).copy(),
    }


def _save_compare_traj(
    *,
    out_path: Path,
    mode_trajs: dict[str, np.ndarray],
    mode_ends: dict[str, str],
    layout: dict,
    danger_radius: float,
    goal_radius: float,
    arena: tuple[float, float, float, float],
    run_id: int,
    mode_use_sf: dict[str, np.ndarray] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    wps = np.asarray(layout["waypoints"], dtype=np.float64).reshape(-1, 2)
    start_xy = wps[0]
    goal_xy = wps[-1]
    vias = wps[1:-1] if wps.shape[0] > 2 else np.zeros((0, 2))
    obstacle_present = bool(layout.get("obstacle_present", False))
    obs_xy = []
    if obstacle_present:
        raw = layout.get("obstacle_xys") or []
        if raw:
            obs_xy = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
        else:
            obs_xy = np.asarray(layout["blue_bin_xy"], dtype=np.float64).reshape(1, 2)

    style = {
        "waypoint_only": {"color": "0.45", "lw": 1.6, "z": 2},
        "safe_only": {"color": "#1f77b4", "lw": 1.8, "z": 3},
        "switching": {"color": "#d62728", "lw": 1.8, "z": 4},
    }
    mode_use_sf = mode_use_sf or {}

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    x0, x1, y0, y1 = (float(v) for v in arena)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="0.55",
            linewidth=1.0,
            zorder=1,
            label="arena",
        )
    )
    ax.set_xlim(x0 - 0.4, x1 + 0.4)
    ax.set_ylim(y0 - 0.4, y1 + 0.4)

    for mode in COMPARE_MODES:
        traj = mode_trajs.get(mode)
        if traj is None or len(traj) == 0:
            continue
        traj = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
        st = style[mode]
        if mode == "switching":
            _plot_colored_traj(
                ax,
                traj,
                mode_use_sf.get(mode),
                nom_label="Switching / waypoint",
                sf_label="Switching / SF",
                linewidth=st["lw"],
                zorder=st["z"],
            )
            continue
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
        facecolors="limegreen",
        edgecolors="black",
        linewidths=0.8,
        s=55,
        zorder=6,
        label="start",
    )
    if vias.size:
        ax.scatter(
            vias[:, 0],
            vias[:, 1],
            c="cyan",
            marker="x",
            s=60,
            zorder=6,
            label="waypoint",
        )
        if vias.shape[0] >= 2:
            ax.scatter(
                float(vias[1, 0]),
                float(vias[1, 1]),
                facecolors="none",
                edgecolors="orange",
                s=90,
                zorder=7,
                label="wp3 (in danger if bin)",
            )
    ax.scatter(
        float(goal_xy[0]),
        float(goal_xy[1]),
        c="red",
        s=90,
        marker="*",
        zorder=7,
        label="goal",
    )
    ax.add_patch(
        Circle(
            (float(goal_xy[0]), float(goal_xy[1])),
            float(goal_radius),
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.1,
            zorder=5,
            label=f"goal r={goal_radius:.2f}",
        )
    )
    r = float(danger_radius)
    for i, xy in enumerate(obs_xy):
        ax.scatter(
            float(xy[0]),
            float(xy[1]),
            c="blue",
            s=70,
            zorder=6,
            label="obstacle" if i == 0 else None,
        )
        ax.add_patch(
            Circle(
                (float(xy[0]), float(xy[1])),
                r,
                fill=False,
                edgecolor="blue",
                linestyle="--",
                linewidth=1.3,
                zorder=5,
                label=f"danger r={r:g}" if i == 0 else None,
            )
        )

    ends = ", ".join(f"{MODE_LABELS[m]}={mode_ends.get(m, '?')}" for m in COMPARE_MODES)
    n_bin = int(len(obs_xy)) if obstacle_present else 0
    bin_tag = f"{n_bin} bin" if n_bin == 1 else f"{n_bin} bins"
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
    fixed_layout: dict,
    seed: int | None = None,
    look_ahead: bool = False,
    file_tag: str | None = None,
) -> dict:
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
    use_sf_hist: list[bool] = []
    probe_rng = np.random.default_rng(0)
    q_nom_hist: list[float] = []
    q_sf_hist: list[float] = []
    q_rand_hist: list[float] = []
    z_norm_hist: list[float] = []

    for step in range(max_visual_steps):
        a_nom_env = env.compute_waypoint_nav_action()
        a_nom_pol = np.asarray(
            policy.map_action_inverse(a_nom_env), dtype=np.float32
        ).reshape(-1)

        if look_ahead:
            z_hat = env.predict_next_latent(a_nom_env)
            z_sf = z_hat
            gate_name = "V(z_hat)"
        else:
            z_sf = z
            gate_name = "Q(a_nom)"

        probe = probe_action_qs(policy, z_sf, a_nom_pol, device, probe_rng)
        q_nom = float(probe["q_nom"])
        q_nom_hist.append(q_nom)
        q_sf_hist.append(float(probe["q_sf"]))
        q_rand_hist.append(float(probe["q_rand"]))
        z_norm_hist.append(float(probe["z_norm"]))
        if step < 5 or step % 10 == 0:
            print(
                f"  [Q-probe] step={step} z_norm={probe['z_norm']:.1f} "
                f"Qnom={probe['q_nom']:.3f} Qsf={probe['q_sf']:.3f} "
                f"Qrand={probe['q_rand']:.3f} "
                f"a_sf_pol={np.array2string(probe['a_sf_pol'], precision=3)}"
            )

        used_sf = False
        if mode == "waypoint_only":
            action = a_nom_env
        elif mode == "safe_only":
            action = probe["a_sf_env"]
            used_sf = True
            hj_interventions += 1
        else:
            if q_nom < safety_threshold:
                action = probe["a_sf_env"]
                used_sf = True
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

        use_sf_hist.append(used_sf)
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

    tag = file_tag or f"run{run_id:02d}"
    video_path = None
    if save_video and frames:
        try:
            import imageio.v2 as imageio

            video_path = str(out_dir / f"{mode}_{tag}.mp4")
            imageio.mimsave(video_path, frames, fps=int(round(env.visual_fps)))
            print(f"[INFO] Saved video: {video_path} ({len(frames)} frames)")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] video save failed: {exc}")

    success = end_reason == "all_waypoints"
    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    result = {
        "mode": mode,
        "run_id": run_id,
        "file_tag": tag,
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
        "use_sf": np.asarray(use_sf_hist, dtype=bool),
        "layout": layout,
        "q_probe_mean_nom": _mean(q_nom_hist),
        "q_probe_mean_sf": _mean(q_sf_hist),
        "q_probe_mean_rand": _mean(q_rand_hist),
        "q_probe_mean_z_norm": _mean(z_norm_hist),
    }
    print(
        f"[RUN {run_id}] mode={mode} end={end_reason} success={success} "
        f"steps={result['visual_steps']} interventions={hj_interventions} "
        f"violations={constraint_violations} minQ={result['min_q_nom']}"
    )
    if q_nom_hist:
        dn = result["q_probe_mean_nom"] - result["q_probe_mean_rand"]
        ds = result["q_probe_mean_sf"] - result["q_probe_mean_rand"]
        print(
            f"[Q-probe summary] n={len(q_nom_hist)} mean z_norm="
            f"{result['q_probe_mean_z_norm']:.1f} | "
            f"Qnom={result['q_probe_mean_nom']:.3f} "
            f"Qsf={result['q_probe_mean_sf']:.3f} "
            f"Qrand={result['q_probe_mean_rand']:.3f} | "
            f"Qnom-Qrand={dn:.3f} Qsf-Qrand={ds:.3f}"
        )
    return result


def main():
    args = get_args_and_merge_config()
    _validate_custom_layout_args(args)
    args.policy_path = str(resolve_policy_path(args.policy_path))
    args.device = torch_device
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    args.visual_feature = str(getattr(args, "visual_feature", "cls")).lower()
    args.critic_fusion = str(getattr(args, "critic_fusion", "concat")).lower()
    args_cli.max_n_obstacles = int(args.max_n_obstacles)
    save_video = bool(args.save_video) and not bool(args.no_save_video)

    look_ahead = bool(getattr(args, "look_ahead", False))
    if look_ahead and args.visual_feature == "cls":
        raise ValueError(
            "--look_ahead is incompatible with --visual_feature cls "
            "(WM predictor is patch-token only)."
        )

    np.random.seed(int(getattr(args, "seed", 0)))
    torch.manual_seed(int(getattr(args, "seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(getattr(args, "seed", 0)))

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("humanoid_test_sac_path_trainlike") / stamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = list(COMPARE_MODES) if args.mode == "compare" else [args.mode]
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] modes={modes} policy={args.policy_path}")
    print(
        f"[INFO] visual_feature={args.visual_feature} "
        f"critic_fusion={args.critic_fusion} "
        f"z_fuse_frames={int(getattr(args, 'z_fuse_frames', 1))} "
        f"freeze_yaw={bool(getattr(args, 'freeze_yaw', False))}"
    )
    print(
        "[INFO] look_ahead="
        f"{look_ahead} (gate={'V(z_t+1)' if look_ahead else 'Q(z_t, a_nom)'})"
    )
    print(
        "[INFO] eval a_nom: "
        f"|v|={'U[' + str(args.nom_speed_min) + ',' + str(args.nom_speed_max) + ']/ep' if bool(args.randomize_nom_speed) else '0.5'} "
        f"act_noise={'N(0,' + str(args.nom_act_noise_std) + ')' if bool(args.nom_act_noise) else 'off'}; "
        "a_sf=eval mean (no coverage noise); "
        f"max_n_obstacles={int(args.max_n_obstacles)}"
    )

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
    configure_path_eval(env, args)
    policy = build_policy(env, args, args.device)
    load_policy_checkpoint(
        policy, args.policy_path, args.device, critic_fusion=args.critic_fusion
    )
    env.policy_for_log = policy
    env.log_rollout_to_wandb = False

    results = []
    base_seed = int(getattr(args, "seed", 0))
    layout_rng = np.random.default_rng(base_seed + 91)
    easy = bool(getattr(args, "easy", False))
    scene_layouts = build_scene_eval_layouts(args) if _scene_flag_list(args) else []
    custom = _custom_waypoints_complete(args)
    custom_layout = build_custom_path_eval_layout(args) if custom else None
    if scene_layouts:
        n_trials = len(scene_layouts)
        if int(args.num_runs) != 1:
            print(
                f"[INFO] --scene: {n_trials} queued layouts; "
                f"ignoring --num_runs={int(args.num_runs)}"
            )
        print(
            f"[INFO] {n_trials} scene(s) from --scene; one compare trial each"
        )
    elif custom:
        n_trials = int(args.num_runs)
        print(
            "[INFO] custom layout frozen for all trials "
            f"(n_bins={int(custom_layout['n_obstacles'])}); "
            "ignores --easy and --no_bin_every"
        )
    elif easy:
        n_trials = int(args.num_runs)
        print(
            "[INFO] --easy: straight line start(0,-2)r=1 → wp2(2,-2)r=1.5 → "
            "wp3 within 1 m of bin; bin fixed (4.5,-2); goal(5.5,-2)r=1. Always one bin."
        )
    else:
        n_trials = int(args.num_runs)
        print(
            "[INFO] no custom XY: sample train-like path "
            f"(absent_p={float(args.obstacle_absent_prob):g}, "
            f"P(2|present)={float(args.two_obstacle_prob):g}, "
            f"max_n={int(args.max_n_obstacles)})"
        )

    arena = (
        float(args.arena_x_min),
        float(args.arena_x_max),
        float(args.arena_y_min),
        float(args.arena_y_max),
    )

    for run_id in range(n_trials):
        if scene_layouts:
            layout = _copy_layout(scene_layouts[run_id])
            with_bin = bool(layout["obstacle_present"])
            file_tag = f"scene{run_id:02d}"
        elif custom:
            layout = _copy_layout(custom_layout)
            with_bin = bool(layout["obstacle_present"])
            file_tag = f"run{run_id:02d}"
        else:
            if easy:
                layout = sample_easy_path_eval_layout(layout_rng, args)
            else:
                layout = sample_path_eval_layout(layout_rng, args)
            with_bin = bool(layout["obstacle_present"])
            file_tag = f"run{run_id:02d}"
        trial_seed = base_seed + run_id
        print(
            f"\n=== Trial {run_id + 1}/{n_trials} tag={file_tag} modes={modes} "
            f"bin={'present' if with_bin else 'absent'} "
            f"waypoints={layout['waypoints'].tolist()} "
            f"bins={layout['obstacle_xys']} ==="
        )
        mode_trajs: dict[str, np.ndarray] = {}
        mode_ends: dict[str, str] = {}
        mode_use_sf: dict[str, np.ndarray] = {}
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
                file_tag=file_tag,
            )
            results.append(r)
            mode_trajs[mode] = r["traj"]
            mode_ends[mode] = r["end_reason"]
            mode_use_sf[mode] = r["use_sf"]

        if args.mode == "compare" and len(modes) > 1:
            _save_compare_traj(
                out_path=out_dir / f"{file_tag}_compare_traj.png",
                mode_trajs=mode_trajs,
                mode_ends=mode_ends,
                layout=layout,
                danger_radius=float(args.lidar_distance_threshold),
                goal_radius=float(args.goal_radius),
                arena=arena,
                run_id=run_id,
                mode_use_sf=mode_use_sf,
            )

    lines = [
        f"policy={args.policy_path}",
        f"algo=SAC path",
        f"eval_mode={args.mode}",
        f"look_ahead={look_ahead}",
        f"safety_threshold={float(args.safety_threshold)}",
        f"modes={modes}",
        f"easy={easy}",
        f"custom={custom}",
        f"n_scenes={len(scene_layouts)}",
        f"danger_radius={float(args.lidar_distance_threshold)}",
        f"goal_radius={float(args.goal_radius)}",
        f"trials={n_trials}",
        f"nom_speed={bool(args.randomize_nom_speed)} "
        f"nom_act_noise={bool(args.nom_act_noise)}",
    ]
    if scene_layouts:
        for i, lay in enumerate(scene_layouts):
            wps = np.asarray(lay["waypoints"], dtype=np.float64)
            bin_txt = list(lay["obstacle_xys"]) if lay["obstacle_present"] else "none"
            lines.append(
                f"scene{i:02d}="
                f"start={wps[0].tolist()} via1={wps[1].tolist()} "
                f"via2={wps[2].tolist()} goal={wps[3].tolist()} bin={bin_txt}"
            )
    elif custom and custom_layout is not None:
        wps = np.asarray(custom_layout["waypoints"], dtype=np.float64)
        bin_txt = (
            list(custom_layout["obstacle_xys"])
            if custom_layout["obstacle_present"]
            else "none"
        )
        lines.append(
            "custom_layout="
            f"start={wps[0].tolist()} via1={wps[1].tolist()} "
            f"via2={wps[2].tolist()} goal={wps[3].tolist()} bin={bin_txt}"
        )
    lines.append("")
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
            qline = ""
            if r.get("q_probe_mean_nom") is not None:
                qline = (
                    f" Qnom={r['q_probe_mean_nom']:.3f} Qsf={r['q_probe_mean_sf']:.3f} "
                    f"Qrand={r['q_probe_mean_rand']:.3f}"
                )
            tag = r.get("file_tag") or f"run{int(r['run_id']):02d}"
            bin_txt = (
                "absent"
                if not r["layout"]["obstacle_present"]
                else r["layout"]["obstacle_xys"]
            )
            lines.append(
                f"  {tag}: end={r['end_reason']} success={r['success']} "
                f"bin={bin_txt} "
                f"steps={r['visual_steps']} int={r['hj_interventions']} "
                f"viol={r['constraint_violations']} minQ={r['min_q_nom']}{qline}"
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
