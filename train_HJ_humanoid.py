"""DDPG HJ safety-filter training on Isaac G1 latent space."""

import argparse
import os
import sys
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

parser = argparse.ArgumentParser("DDPG HJ on DINO latent Humanoid (Isaac G1)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/storage1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/cargoal",
    help="Where to find the DINO-WM checkpoints",
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
    default="dino",
    help="Encoder subfolder under dino_ckpt_dir",
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
    default=1,
    help="Upload a rollout video to wandb every N finished episodes (0 disables).",
)
parser.add_argument(
    "--y_bound",
    type=float,
    default=1.5,
    help="Soft |y| corridor (meters). |y|>bound truncates episode; does not change h_s.",
)
parser.add_argument(
    "--critic_warmup_updates",
    type=int,
    default=1000,
    help=(
        "After buffer warm-up collect, run this many critic-only updates "
        "(policy.warmup=True, actor frozen) before joint training."
    ),
)
parser.add_argument(
    "--actor_bc_warmup_updates",
    type=int,
    default=0,
    help=(
        "DEPRECATED: actor imitation of buffer actions. Default 0 — "
        "actions are chosen by maximizing the critic over sampled candidates."
    ),
)
parser.add_argument(
    "--critic_action_samples",
    type=int,
    default=64,
    help=(
        "Number of uniform actions in [-1,1] to score with the critic when "
        "selecting the safest action (train collect + Bellman target)."
    ),
)
parser.add_argument(
    "--action_reg_coef",
    type=float,
    default=0.0,
    help=(
        "DEPRECATED mid-point L2 on actor output in [-1,1]. "
        "Do NOT use: zero maps to vx=0.4 and causes collapse. Prefer "
        "--boundary_reg_coef. Kept for CLI compat; applied only if > 0."
    ),
)
parser.add_argument(
    "--boundary_reg_coef",
    type=float,
    default=0.5,
    help=(
        "Extra penalty when |act| > 0.8 (near tanh corners ±1). "
        "Set 0 to disable."
    ),
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
        "  python train_HJ_humanoid.py --headless --visual_mode rtx_rgb ..."
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

# Progress bar: only actor/critic. Reg / sat metrics still go to wandb via logger.
_PROGRESS_BAR_LOSS_KEYS = ("loss/actor", "loss/critic")


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
        project="ddpg-hj-latent-humanoid",
        name=f"ddpg-{args.dino_encoder}-{timestamp}",
        config=vars(args),
    )
    # All rollout + loss curves share env_step as x-axis.
    wandb.define_metric("trainer/env_step")
    wandb.define_metric("trainer/update")
    wandb.define_metric("safety/*", step_metric="trainer/env_step")
    wandb.define_metric("actor_action/*", step_metric="trainer/env_step")
    wandb.define_metric("rollout/*", step_metric="trainer/env_step")
    wandb.define_metric("loss/*", step_metric="trainer/env_step")
    wandb.define_metric("train/*", step_metric="trainer/env_step")
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}/logs")
    # update_interval=1: log every grad step (default 1000 would skip almost everything).
    wb_logger = WandbLogger(update_interval=1, train_interval=10**9, test_interval=10**9)
    wb_logger.load(writer)
    logger = wb_logger

    ckpt_dir = Path(args.dino_ckpt_dir)
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
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
    # PyHJ default new_expl=True replaces acts with random when Q(rand)>=0;
    # that biases the buffer. Use standard Gaussian noise on critic-greedy acts.
    policy.new_expl = False
    n_act_samples = int(getattr(args, "critic_action_samples", 64))
    act_dim = int(np.prod(action_shape))

    def _critic_greedy_forward(batch, state=None, model="actor", input="obs", **kwargs):
        """Action = argmax over sampled candidates scored by the online critic."""
        del state, model, kwargs
        obs = batch[input]
        best_act, _ = select_actions_max_q(
            policy.critic,
            obs,
            act_dim=act_dim,
            n_candidates=n_act_samples,
            device=args.device,
        )
        return Batch(act=best_act, state=None)

    def _target_q_critic_max(buffer, indices):
        """Bellman next-action: max_a Q_target(s', a) via the same sampler."""
        batch = buffer[indices]
        best_act, _ = select_actions_max_q(
            policy.critic_old,
            batch.obs_next,
            act_dim=act_dim,
            n_candidates=n_act_samples,
            device=args.device,
        )
        return policy.critic_old(batch.obs_next, best_act)

    def learn_critic_only(batch, **kwargs):
        """Train critic only. Actions at collect/eval come from critic sampling."""
        del kwargs
        td, critic_loss = policy._mse_optimizer(
            batch, policy.critic, policy.critic_optim
        )
        batch.weight = td
        policy.sync_weight()
        with torch.no_grad():
            # Log what the sampler would pick on this batch (for wandb curves).
            best_act, _ = select_actions_max_q(
                policy.critic,
                batch.obs,
                act_dim=act_dim,
                n_candidates=min(32, n_act_samples),
                device=args.device,
            )
            act_abs_mean = float(best_act.abs().mean().item())
            act_sat_frac = float((best_act.abs() > 0.95).float().mean().item())
        return {
            "loss/actor": 0.0,
            "loss/critic": float(critic_loss.item()),
            "loss/action_reg": 0.0,
            "loss/boundary_reg": 0.0,
            "train/actor_abs_mean": act_abs_mean,
            "train/actor_sat_frac": act_sat_frac,
        }

    policy.forward = _critic_greedy_forward  # type: ignore[method-assign]
    policy._target_q = _target_q_critic_max  # type: ignore[method-assign]
    policy.learn = learn_critic_only  # type: ignore[method-assign]
    policy._critic_action_samples = n_act_samples
    # Keep actor in eval; not used for control or Bellman backup.
    policy.actor.eval()
    for p in policy.actor.parameters():
        p.requires_grad = False
    print(
        f"[INFO] critic-only SF: gamma={args.gamma_pyhj}, "
        f"critic_warmup_updates={args.critic_warmup_updates}, "
        f"critic_action_samples={n_act_samples}, "
        f"exploration_noise_start={args.exploration_noise} "
        f"(anneal → 0.1; added on top of critic-greedy), "
        f"actor frozen (no imitation / no actor RL)"
    )
    # Force every-episode video uploads (CLI 0 still disables).
    if int(getattr(args, "wandb_video_every", 1)) != 0:
        args.wandb_video_every = 1
    log_state = {"env_step": 0, "update": 0}
    train_envs.set_env_attr("wandb_video_every", int(args.wandb_video_every))
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
        # Smoothed losses (same MovAvg as the progress bar), x-axis = env_step.
        log_state["update"] = int(step)
        env_step = int(log_state.get("env_step", 0))
        payload = {
            "trainer/env_step": float(env_step),
            "trainer/update": float(step),
            "loss/actor": float(update_result["loss/actor"]),
            "loss/critic": float(update_result["loss/critic"]),
        }
        for key in (
            "loss/action_reg",
            "loss/boundary_reg",
            "train/actor_abs_mean",
            "train/actor_sat_frac",
        ):
            if key in update_result:
                payload[key] = float(update_result[key])
        wandb.log(payload)
        return _orig_log_update(update_result, step)

    logger.log_update_data = log_update_data_synced  # type: ignore[method-assign]

    # Stash deterministic SF action (pre-noise) for env-side logging; do not wandb.log here.
    orig_exploration_noise = policy.exploration_noise

    def exploration_noise_stash_clean(act, batch):
        try:
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            row = np.asarray(act_env, dtype=np.float64).reshape(-1, 3)[0]
            policy.last_clean_act_env = row
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] noiseless actor stash failed: {exc}")
            policy.last_clean_act_env = None
        return orig_exploration_noise(act, batch)

    policy.exploration_noise = exploration_noise_stash_clean

    def train_fn(epoch: int, step_idx: int):
        # Linearly anneal exploration noise: start (e.g. 0.3) → 0.1 by last epoch.
        del step_idx
        start_sigma = float(args.exploration_noise)
        end_sigma = 0.1
        if end_epoch <= start_epoch:
            sigma = end_sigma
        else:
            frac = (epoch - start_epoch) / float(end_epoch - start_epoch)
            frac = min(1.0, max(0.0, frac))
            sigma = start_sigma + (end_sigma - start_sigma) * frac
        if getattr(policy, "_noise", None) is not None:
            policy._noise._sigma = float(sigma)
        wandb.log(
            {
                "trainer/env_step": float(log_state.get("env_step", 0)),
                "train/exploration_sigma": float(sigma),
            }
        )

    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    # Replay buffer is not restored; collect a fresh warm-up even when resuming weights.
    # Initial buffer: region waypoint nav (DataCollection style), not random actor.
    print(
        "[INFO] Collecting initial transitions with WaypointNavController "
        "(start fixed (0,0) -> left|right|middle(hit bin) -> back; "
        "bin FIXED at (2,0) during buffer)..."
    )
    train_envs.set_env_attr("randomize_obstacle", False)
    _actor_forward = policy.forward
    _expl_fn = policy.exploration_noise

    def _waypoint_forward(batch, state=None, **kwargs):
        del batch, state, kwargs
        fns = train_envs.get_env_attr("compute_waypoint_nav_action")
        acts_env = np.stack(
            [np.asarray(fn(), dtype=np.float32).reshape(-1) for fn in fns],
            axis=0,
        )
        acts = policy.map_action_inverse(acts_env)
        return Batch(act=acts, state=None)

    def _waypoint_expl(act, batch):
        # Stash env cmd for actor_action/* logs; do not add Gaussian noise.
        try:
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            row = np.asarray(act_env, dtype=np.float64).reshape(-1, 3)[0]
            policy.last_clean_act_env = row
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] waypoint act stash failed: {exc}")
            policy.last_clean_act_env = None
        return act

    policy.forward = _waypoint_forward  # type: ignore[method-assign]
    policy.exploration_noise = _waypoint_expl  # type: ignore[method-assign]
    print(
        f"[INFO] Warm-up collect n_step=1000; "
        f"wandb video every {int(args.wandb_video_every)} episode(s)."
    )
    train_collector.collect(1000)
    policy.forward = _actor_forward  # type: ignore[method-assign]
    policy.exploration_noise = _expl_fn  # type: ignore[method-assign]
    # Training: resample bin y on x=2 each episode reset.
    train_envs.set_env_attr("randomize_obstacle", True)
    print(
        "[INFO] Initial waypoint collection done; restoring critic-greedy actions. "
        "Obstacle randomization ON for training (x=2, y∈[-0.5,0.5])."
    )

    # Critic-only warm-up: fit Q before enabling actor (anti-collapse).
    n_critic_wu = int(getattr(args, "critic_warmup_updates", 1000))
    if n_critic_wu > 0 and not args.resume_policy:
        print(f"[INFO] Critic-only warmup: {n_critic_wu} updates (actor frozen)...")
        policy.warmup = True
        for i in range(1, n_critic_wu + 1):
            metrics = policy.update(args.batch_size_pyhj, buffer)
            env_step = int(log_state.get("env_step", 0))
            log_state["update"] = int(log_state.get("update", 0)) + 1
            wandb.log(
                {
                    "trainer/env_step": float(env_step),
                    "trainer/update": float(log_state["update"]),
                    "loss/actor": float(metrics.get("loss/actor", 0.0)),
                    "loss/critic": float(metrics["loss/critic"]),
                    "train/critic_warmup": 1.0,
                }
            )
            if i == 1 or i % 100 == 0 or i == n_critic_wu:
                print(
                    f"  critic_warmup {i}/{n_critic_wu}: "
                    f"loss/critic={metrics['loss/critic']:.4f}"
                )
        policy.warmup = False
        print("[INFO] Critic-only warmup done.")
    else:
        policy.warmup = False

    n_actor_bc = int(getattr(args, "actor_bc_warmup_updates", 0))
    if n_actor_bc > 0:
        print(
            f"[WARN] actor_bc_warmup_updates={n_actor_bc} ignored "
            "(critic-only control; no actor imitation)."
        )

    log_path = Path(f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}")
    if args.resume_policy:
        log_path = Path(
            f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}-resume{resume_epoch}"
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
