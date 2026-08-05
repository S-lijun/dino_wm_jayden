"""Critic-only HJ training for QP safety filter on Isaac G1 latent space.

Does NOT train a DDPG actor for control. Collects with WaypointNavController,
fits the avoid critic, and saves ``policy.pth`` (critic used at test time by
``test_HJ_humanoid_qp.py`` discrete QP filter).

Does not modify ``train_HJ_humanoid.py`` (DDPG actor path).
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# Isaac Sim must start before isaaclab / env imports.
# ---------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ISAACLAB_ROOT = os.path.join(REPO_ROOT, "IsaacLab")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ISAACLAB_ROOT)

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("QP/critic HJ on DINO latent Humanoid (Isaac G1)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/workspace",
    help="Parent dir of the WM run folder (joins with --dino_encoder)",
)
parser.add_argument(
    "--config",
    type=str,
    default="train_HJ_configs.yaml",
    help="Path to flat YAML of hyperparameters",
)
parser.add_argument(
    "--with_proprio",
    action="store_true",
    help="Include proprioceptive embeddings in latent state",
)
parser.add_argument(
    "--dino_encoder",
    type=str,
    default="wm_ckpt_18-27-17",
    help="Encoder / WM run folder under dino_ckpt_dir",
)
parser.add_argument("--latent_h", default=False, action="store_true")
parser.add_argument(
    "--visual_mode",
    type=str,
    default="depth_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
)
parser.add_argument(
    "--resume_policy",
    type=str,
    default=None,
    help=(
        "Path to a saved .../epoch_id_N/policy.pth. "
        "Loads weights and continues training from epoch N+1 up to --total-episodes "
        "(from YAML / CLI, e.g. 120)."
    ),
)
parser.add_argument(
    "--wandb_video_every",
    type=int,
    default=5,
    help=(
        "Upload a rollout video to wandb every N finished episodes after buffer "
        "warm-up (0=off; default 5). Buffer warm-up never records video."
    ),
)
parser.add_argument(
    "--y_bound",
    type=float,
    default=0.0,
    help="Soft |y| corridor (meters). <=0 disables (default). >0 truncates when |y|>bound.",
)
parser.add_argument(
    "--critic_warmup_updates",
    type=int,
    default=1000,
    help=(
        "After buffer warm-up collect, run this many critic-only updates "
        "before the epoch loop (actor stays frozen throughout)."
    ),
)
parser.add_argument(
    "--critic_action_samples",
    type=int,
    default=64,
    help="Candidates for Bellman next-action max_a Q (full vx,vy,yaw unless --freeze_yaw).",
)
parser.add_argument(
    "--train_collect_noise",
    type=float,
    default=0.1,
    help="Gaussian noise sigma on waypoint cmds during post-buffer collect (policy space).",
)
parser.add_argument(
    "--freeze_yaw",
    action="store_true",
    help="Freeze yaw=0 in collect + Bellman max_a (legacy). Default: yaw free.",
)
# --device is already registered by AppLauncher.add_app_launcher_args()

args_cli, remaining = parser.parse_known_args()

sys.path.insert(0, os.path.join(ISAACLAB_ROOT, "scripts/demos"))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")
if not bool(getattr(args_cli, "headless", False)):
    print(
        "[WARN] Running WITHOUT --headless. GUI + rtx_rgb will freeze as "
        "'Not Responding' on Windows. Keep rtx_rgb, but MUST add --headless:\n"
        "  python train_HJ_humanoid_qp.py --headless --visual_mode rtx_rgb ..."
    )
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Prefer repo root after AppLauncher mutates sys.path (Isaac cv2/utils shadowing).
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Imports after AppLauncher
# ---------------------------------------------------------------------
import numpy as np
import torch
import yaml
import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from PyHJ.data import Batch, Collector, VectorReplayBuffer
from PyHJ.trainer import offpolicy_trainer
from PyHJ.trainer.offpolicy import OffpolicyTrainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing

from env.isaac.critic_select import select_actions_max_q

# Progress bar: critic only (actor unused for QP deployment).
_PROGRESS_BAR_LOSS_KEYS = ("loss/critic",)


def _log_update_data_bar_filter(self, data, losses):
    for k in losses.keys():
        self.stat[k].add(losses[k])
        losses[k] = self.stat[k].get()
        if k in _PROGRESS_BAR_LOSS_KEYS:
            data[k] = f"{losses[k]:.3f}"
    self.logger.log_update_data(losses, self.gradient_step)


OffpolicyTrainer.log_update_data = _log_update_data_bar_filter  # type: ignore[method-assign]

from wm_load import load_model  # not plan.py — avoids env.venv / utils collision
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


def _parse_resume_epoch_from_path(policy_path: str | Path) -> int:
    """Infer N from '.../epoch_id_N/policy.pth'."""
    import re

    for part in Path(policy_path).resolve().parts[::-1]:
        m = re.fullmatch(r"epoch_id_(\d+)", part)
        if m:
            return int(m.group(1))
    raise ValueError(
        f"Cannot parse epoch from --resume_policy path (expected .../epoch_id_N/policy.pth): "
        f"{policy_path}"
    )


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str) -> None:
    """Load a previously saved policy.state_dict() into ``policy``."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--resume_policy not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(state, strict=True)
    print(f"[INFO] Resumed policy from {ckpt_path}")


def main():
    args = get_args_and_merge_config()

    args.critic_lr = float(args.critic_lr)
    args.actor_lr = float(args.actor_lr)
    args.tau = float(args.tau)
    args.gamma_pyhj = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.update_per_step = float(args.update_per_step)
    args.step_per_epoch = int(args.step_per_epoch)
    args.step_per_collect = int(args.step_per_collect)
    args.test_num = int(args.test_num)
    args.training_num = int(args.training_num)
    args.total_episodes = int(args.total_episodes)
    args.batch_size_pyhj = int(args.batch_size_pyhj)
    args.buffer_size = int(args.buffer_size)
    # Latent dim ~74k; buffer_size=40k ⇒ tens of GiB and looks "hung" on first add.
    # Critic-only QP does not need a huge buffer.
    if args.buffer_size > 10000:
        print(
            f"[INFO] Capping buffer_size {args.buffer_size} → 10000 "
            "(large latent; first buffer.add otherwise allocates ~20GiB+ and stalls).",
            flush=True,
        )
        args.buffer_size = 10000
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    args.device = torch_device

    if args.training_num > 1:
        print("[WARN] Isaac Sim supports one env instance; forcing training_num=1")
        args.training_num = 1
    if args.test_num > 1:
        args.test_num = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ.setdefault("MPLCONFIGDIR", "/storage1/sibai/Active/ihab/tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%m%d_%H%M")
    wandb.init(
        project="qp-hj-latent-humanoid",
        name=f"qp-critic-{args.dino_encoder}-{timestamp}",
        config=vars(args),
    )
    # Rollout curves vs env_step; loss curves vs update (env_step freezes during
    # critic-only warmup — binding loss to env_step makes wandb Charts look empty).
    wandb.define_metric("trainer/env_step")
    wandb.define_metric("trainer/update")
    wandb.define_metric("safety/*", step_metric="trainer/env_step")
    wandb.define_metric("actor_action/*", step_metric="trainer/env_step")
    wandb.define_metric("rollout/*", step_metric="trainer/env_step")
    wandb.define_metric("loss/*", step_metric="trainer/update")
    wandb.define_metric("train/*", step_metric="trainer/update")
    writer = SummaryWriter(log_dir=f"runs/qp_hj_humanoid/{args.dino_encoder}-{timestamp}/logs")
    # update_interval=1: log every grad step (default 1000 would skip almost everything).
    wb_logger = WandbLogger(update_interval=1, train_interval=10**9, test_interval=10**9)
    wb_logger.load(writer)
    logger = wb_logger

    ckpt_dir = Path(args.dino_ckpt_dir)
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
    if not hydra_cfg.is_file():
        raise FileNotFoundError(
            f"WM hydra.yaml not found: {hydra_cfg}\n"
            f"  Expected layout: <dino_ckpt_dir>/<dino_encoder>/hydra.yaml\n"
            f"  On this machine use e.g. --dino_ckpt_dir /workspace "
            f"--dino_encoder wm_ckpt_18-27-17"
        )
    if not snapshot.is_file():
        raise FileNotFoundError(f"WM checkpoint not found: {snapshot}")
    print(f"[INFO] Loading WM from {ckpt_dir}")
    train_cfg = OmegaConf.load(str(hydra_cfg))
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=args.device)
    for p in wm.parameters():
        p.requires_grad = True

    def make_env():
        return LatentHumanoidEnv(
            args,
            wm,
            args.device,
            args_cli,
            with_proprio=args.with_proprio,
            latent_h=args.latent_h,
            wandb_video_every=args.wandb_video_every,
        )

    # Isaac Sim allows only one SimulationContext per process. Do not create a
    # second DummyVectorEnv for "test" — offpolicy_trainer already uses
    # test_collector=None below.
    if args.training_num != 1:
        print(f"[WARN] Forcing training_num=1 (was {args.training_num})")
        args.training_num = 1
    train_envs = DummyVectorEnv([make_env])

    state_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape = state_space.shape
    action_shape = action_space.shape or action_space.n
    # Actor outputs tanh in [-1, 1]; map_action linearly scales to env bounds
    # (vx [0,0.8], vy [-0.5,0.5], yaw [-0.5,0.5]). Do NOT use high as max_action —
    # that Dubins/CarGoal pattern assumes symmetric [-high, high] actions.
    max_action = 1.0
    # PyHJ map_action checks gym.spaces.Box; env uses gymnasium.Box.
    import gym

    policy_action_space = gym.spaces.Box(
        low=np.asarray(action_space.low, dtype=np.float32),
        high=np.asarray(action_space.high, dtype=np.float32),
        dtype=np.float32,
    )

    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=args.critic_net,
        activation=getattr(torch.nn, args.critic_activation),
        concat=True,
        device=args.device,
    )
    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj
    )

    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device,
    )
    actor = Actor(
        actor_net,
        action_shape,
        max_action=max_action,
        device=args.device,
    ).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

    policy = avoid_DDPGPolicy_annealing(
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=policy_action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )
    # Critic-only training for QP deployment (actor kept in ckpt but frozen).
    policy.new_expl = False
    n_act_samples = int(getattr(args, "critic_action_samples", 64))
    act_dim = int(np.prod(action_shape))
    train_collect_noise = float(getattr(args, "train_collect_noise", 0.1))
    freeze_yaw = bool(getattr(args, "freeze_yaw", False))
    bellman_fixed_dims = {2: 0.0} if freeze_yaw else None

    def _zero_yaw_act(act):
        if isinstance(act, torch.Tensor):
            if act.shape[-1] < 3:
                return act
            out = act.clone()
            out[..., 2] = 0.0
            return out
        out = np.array(act, dtype=np.float32, copy=True)
        if out.ndim >= 1 and out.shape[-1] >= 3:
            out[..., 2] = 0.0
        return out

    def _waypoint_acts_policy(*, zero_yaw: bool = False) -> np.ndarray:
        fns = train_envs.get_env_attr("compute_waypoint_nav_action")
        acts_env = np.stack(
            [np.asarray(fn(), dtype=np.float32).reshape(-1) for fn in fns],
            axis=0,
        )
        acts = np.asarray(policy.map_action_inverse(acts_env), dtype=np.float32)
        if zero_yaw:
            acts = _zero_yaw_act(acts)
        return acts

    def _target_q_sample_max(buffer, indices):
        """Bellman next-action: max_a Q_target(s', a); yaw free unless --freeze_yaw."""
        batch = buffer[indices]
        best_act, _ = select_actions_max_q(
            policy.critic_old,
            batch.obs_next,
            act_dim=act_dim,
            n_candidates=n_act_samples,
            device=args.device,
            fixed_dims=bellman_fixed_dims,
        )
        return policy.critic_old(batch.obs_next, best_act)

    def learn_critic_only(batch, **kwargs):
        del kwargs
        td, critic_loss = policy._mse_optimizer(
            batch, policy.critic, policy.critic_optim
        )
        batch.weight = td
        policy.sync_weight()
        return {
            "loss/actor": 0.0,
            "loss/critic": float(critic_loss.item()),
        }

    policy._target_q = _target_q_sample_max  # type: ignore[method-assign]
    policy.learn = learn_critic_only  # type: ignore[method-assign]
    policy.actor.eval()
    for p in policy.actor.parameters():
        p.requires_grad = False
    print(
        f"[INFO] QP/critic-only train: gamma={args.gamma_pyhj}, "
        f"critic_warmup_updates={args.critic_warmup_updates}, "
        f"critic_action_samples={n_act_samples}, "
        f"train_collect_noise={train_collect_noise}, "
        f"freeze_yaw={freeze_yaw}, "
        f"Bellman=max_a Q (yaw={'0' if freeze_yaw else 'free'}); "
        f"actor frozen; deploy via test_HJ_humanoid_qp.py"
    )
    # Video every episode is extremely expensive on RTX (resize+GIF encode can
    # make the first collect chunk look hung for 20–40 min). Default: off during
    # buffer warm-up; restore after. CLI --wandb_video_every 0 keeps video off.
    video_every_train = int(getattr(args, "wandb_video_every", 0))
    if video_every_train < 0:
        video_every_train = 0
    log_state = {"env_step": 0, "update": 0}
    # Buffer warm-up: no rollout video (scalars still log).
    train_envs.set_env_attr("wandb_video_every", 0)
    train_envs.set_env_attr("log_state", log_state)
    # Enable per-frame HJ overlay + safety/l, safety/hj wandb scalars in the env.
    train_envs.set_env_attr("policy_for_log", policy)
    policy.log_state = log_state
    policy.last_clean_act_env = None

    # Warm-start from a previous safety-filter checkpoint (actor+critic+targets).
    # Continues until args.total_episodes (same target as a fresh run, e.g. 120).
    resume_epoch = 0
    if args.resume_policy:
        load_policy_checkpoint(policy, args.resume_policy, args.device)
        resume_epoch = _parse_resume_epoch_from_path(args.resume_policy)
        if resume_epoch >= args.total_episodes:
            raise ValueError(
                f"resume epoch {resume_epoch} already >= total-episodes "
                f"{args.total_episodes}; nothing to train."
            )
        print(
            f"[INFO] Resumed at epoch {resume_epoch}; "
            f"continuing through epoch {args.total_episodes}"
        )

    start_epoch = resume_epoch + 1
    end_epoch = args.total_episodes

    # Progress-bar losses are MovAvg-smoothed inside OffpolicyTrainer.log_update_data.
    # Log THAT same smoothed dict to wandb (not the raw per-learn instantaneous loss).
    _orig_log_update = logger.log_update_data

    def log_update_data_synced(update_result, step):
        log_state["update"] = int(step)
        env_step = int(log_state.get("env_step", 0))
        payload = {
            "trainer/env_step": float(env_step),
            "trainer/update": float(step),
            "loss/critic": float(update_result["loss/critic"]),
            "loss/actor": float(update_result.get("loss/actor", 0.0)),
        }
        wandb.log(payload)
        return _orig_log_update(update_result, step)

    logger.log_update_data = log_update_data_synced  # type: ignore[method-assign]

    def _stash_act(act, batch):
        del batch
        try:
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            policy.last_clean_act_env = np.asarray(act_env, dtype=np.float64).reshape(-1, 3)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] act stash failed: {exc}")
            policy.last_clean_act_env = None
        return act

    def train_fn(epoch: int, step_idx: int):
        del epoch, step_idx
        wandb.log(
            {
                "trainer/env_step": float(log_state.get("env_step", 0)),
                "trainer/update": float(log_state.get("update", 0)),
                "train/collect_noise": float(train_collect_noise),
            }
        )

    def _waypoint_forward_buffer(batch, state=None, **kwargs):
        del batch, state, kwargs
        acts = _waypoint_acts_policy(zero_yaw=False)
        return Batch(act=acts, state=None)

    def _waypoint_forward_train(batch, state=None, **kwargs):
        """Training collect: waypoint goals; yaw free unless --freeze_yaw."""
        del batch, state, kwargs
        acts = _waypoint_acts_policy(zero_yaw=freeze_yaw)
        if train_collect_noise > 0:
            noise = np.random.randn(*acts.shape).astype(np.float32) * train_collect_noise
            if freeze_yaw:
                noise[..., 2] = 0.0
            acts = np.clip(acts + noise, -1.0, 1.0)
            if freeze_yaw:
                acts = _zero_yaw_act(acts)
        return Batch(act=acts, state=None)

    def _gpu_compute_pids() -> list[int]:
        """Return PIDs currently holding GPU compute contexts (best-effort)."""
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=5,
            )
            pids = []
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
            return pids
        except Exception:  # noqa: BLE001
            return []

    # Avoid building Collector before waypoint forward is installed (reset is fine
    # either way, but keep action path consistent from the first collect).
    policy.forward = _waypoint_forward_buffer  # type: ignore[method-assign]
    policy.exploration_noise = _stash_act  # type: ignore[method-assign]

    print(
        "[INFO] Collecting initial transitions with WaypointNavController "
        "(start=(0,0) -> front=(1.5,0) -> left|right|middle; bin FIXED at (3.5,0); "
        "middle = collision demos for critic)..."
    )
    train_envs.set_env_attr("randomize_obstacle", False)
    train_envs.set_env_attr("include_middle_pass", True)
    train_envs.set_env_attr("wandb_video_every", 0)
    # Keep safety/l + safety/hj during buffer (needed for wandb Charts).
    train_envs.set_env_attr("log_rollout_to_wandb", True)
    train_envs.set_env_attr("debug_step_timing", True)
    train_envs.set_env_attr("_debug_steps_left", 3)

    # nvidia-smi PIDs may be host-namespace; ignore if only our process holds GPU.
    other_gpu = [p for p in _gpu_compute_pids() if p != os.getpid()]
    if other_gpu:
        try:
            import subprocess

            mem_mib = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                .splitlines()[0]
                .strip()
            )
        except Exception:  # noqa: BLE001
            mem_mib = -1
        # False positive: Isaac often appears as a different PID in nvidia-smi.
        if mem_mib > 8000:
            print(
                f"[WARN] GPU busy (~{mem_mib} MiB); nvidia pids={other_gpu}. "
                "If warm-up hangs, pkill -9 -f train_HJ_humanoid_qp.py and rerun.",
                flush=True,
            )

    obs_dim = int(np.prod(state_space.shape))
    est_gb = args.buffer_size * obs_dim * 4 * 2 / (1024**3)
    print(
        f"[INFO] Replay buffer size={args.buffer_size}, obs_dim={obs_dim} "
        f"(~{est_gb:.1f} GiB for obs+obs_next; allocated on first collect).",
        flush=True,
    )
    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    print("[INFO] Creating collector (env reset)...", flush=True)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    print("[INFO] Collector ready.", flush=True)

    warmup_n_step = 1000
    # Print every env step — RTX can take several seconds/step; large chunks look hung.
    warmup_chunk = 1
    print(
        f"[INFO] Warm-up collect n_step={warmup_n_step} "
        f"(progress every {warmup_chunk} step; video OFF + wandb scalars OFF during buffer).",
        flush=True,
    )
    collected = 0
    ep_total = 0
    t0 = time.time()
    print(
        "[INFO] buffer warm-up: starting first collect step "
        "(RTX step ~1s after warm; then buffer may allocate for ~10–60s — "
        "wait for '1/1000' print)...",
        flush=True,
    )
    while collected < warmup_n_step:
        n = min(warmup_chunk, warmup_n_step - collected)
        chunk_t0 = time.time()
        if collected == 0:
            print(
                "[INFO] buffer warm-up: calling collector.collect(1) "
                "(after step exits, first buffer.add may pause — not RTX stuck)...",
                flush=True,
            )
        stats = train_collector.collect(n)
        collected += int(stats.get("n/st", n))
        ep_total += int(stats.get("n/ep", 0))
        elapsed = time.time() - t0
        chunk_dt = time.time() - chunk_t0
        sps = collected / max(elapsed, 1e-6)
        # Throttle console a bit after the first few steps.
        if collected <= 5 or collected % 10 == 0 or collected >= warmup_n_step:
            print(
                f"[INFO] buffer warm-up: {collected}/{warmup_n_step} steps "
                f"({100.0 * collected / warmup_n_step:.0f}%), "
                f"episodes={ep_total}, {sps:.2f} steps/s, "
                f"last_step={chunk_dt:.1f}s, elapsed={elapsed:.0f}s",
                flush=True,
            )
    print(
        f"[INFO] Warm-up collect done: {collected} steps, {ep_total} episodes, "
        f"{time.time() - t0:.0f}s total.",
        flush=True,
    )

    # Training collect: still waypoint only (no actor). Keep middle so critic
    # sees collisions; random left|right|middle + randomized bin y.
    train_envs.set_env_attr("randomize_obstacle", True)
    train_envs.set_env_attr("include_middle_pass", True)
    train_envs.set_env_attr("wandb_video_every", video_every_train)
    train_envs.set_env_attr("log_rollout_to_wandb", True)
    train_envs.set_env_attr("debug_step_timing", False)
    if video_every_train > 0:
        print(
            f"[INFO] Restored wandb_video_every={video_every_train} after buffer "
            "(expensive on RTX; use --wandb_video_every 0 to keep off).",
            flush=True,
        )
    policy.forward = _waypoint_forward_train  # type: ignore[method-assign]
    policy.exploration_noise = _stash_act  # type: ignore[method-assign]
    print(
        "[INFO] Buffer done; training collect=waypoint "
        f"(goal=random left|right|middle, freeze_yaw={freeze_yaw}, "
        f"noise={train_collect_noise}). "
        "Middle kept for critic collision demos (no actor). "
        "Obstacle randomization ON; y_bound disabled."
    )

    n_critic_wu = int(getattr(args, "critic_warmup_updates", 1000))
    if n_critic_wu > 0 and not args.resume_policy:
        print(f"[INFO] Critic-only warmup: {n_critic_wu} updates...")
        for i in range(1, n_critic_wu + 1):
            metrics = policy.update(args.batch_size_pyhj, buffer)
            log_state["update"] = int(log_state.get("update", 0)) + 1
            wandb.log(
                {
                    "trainer/env_step": float(log_state.get("env_step", 0)),
                    "trainer/update": float(log_state["update"]),
                    "loss/critic": float(metrics["loss/critic"]),
                    "train/critic_warmup": 1.0,
                },
                commit=True,
            )
            if i == 1 or i % 100 == 0 or i == n_critic_wu:
                print(
                    f"  critic_warmup {i}/{n_critic_wu}: "
                    f"loss/critic={metrics['loss/critic']:.4f}"
                )
        print("[INFO] Critic-only warmup done.")

    log_path = Path(f"runs/qp_hj_humanoid/{args.dino_encoder}-{timestamp}")
    if args.resume_policy:
        log_path = Path(
            f"runs/qp_hj_humanoid/{args.dino_encoder}-{timestamp}-resume{resume_epoch}"
        )
    print(
        f"[INFO] Training epochs {start_epoch}..{end_epoch}; ckpts -> {log_path}"
    )
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n=== Epoch {epoch}/{end_epoch} ===")
        stats = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,
            max_epoch=1,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size_pyhj,
            update_per_step=args.update_per_step,
            stop_fn=lambda r: False,
            train_fn=train_fn,
            save_best_fn=None,
            logger=logger,
        )

        numeric = {
            "train/epoch": int(epoch),
            "trainer/env_step": float(log_state.get("env_step", 0)),
            "trainer/update": float(log_state.get("update", 0)),
        }
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                numeric[f"train/{k}"] = v
            elif isinstance(v, np.generic):
                numeric[f"train/{k}"] = float(v)
        wandb.log(numeric)

        ckpt_dir_epoch = log_path / f"epoch_id_{epoch}"
        ckpt_dir_epoch.mkdir(exist_ok=True, parents=True)
        torch.save(policy.state_dict(), ckpt_dir_epoch / "policy.pth")

    print("[INFO] Training complete.")
    simulation_app.close()


if __name__ == "__main__":
    main()
