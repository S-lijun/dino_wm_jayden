"""Offline real-robot fine-tune of a sim-trained lidar path SAC safety filter.

Same update as ``finetune_HJ_humanoid_sac_path.py``. The state is not DINO.
Each frame is frozen Utonia on the filtered Mid-360 cloud, concatenated with
the frozen WM proprio encoder, then averaged over ``--z_fuse_frames``
(default 5, matching lidar sim training).

Episode files from ``scripts/pack_real_lidar_episodes.py``::

    lidar_xyz      float32 (P, 3)   sim-frame xyz, frames concatenated
    lidar_counts   int32 (T,)
    proprio        float32 (T, J)
    joint_names
    act            float32 (T, 3)
    dist           float32 (T,)     lidar360_min_distance in meters

``h_s = dist - --lidar_distance_threshold`` (default 1.0). No Isaac.
Actor and critics are both updated. ``--critics_only`` freezes the actor.

Example::

    python finetune_HJ_humanoid_sac_path_lidar.py \\
      --data_dir DataCollection/ft_episodes_lidar \\
      --resume_policy runs/sac_hj_humanoid_path/.../epoch_id_96/policy.pth \\
      --dino_ckpt_dir wm_ckpt_18-27-17 \\
      --critic_fusion concat --with_proprio \\
      --z_fuse_frames 5 --lidar_distance_threshold 1.0
"""

from __future__ import annotations

import argparse
import copy
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import finetune_HJ_humanoid_sac_path as ft
from env.isaac.utonia_encoder import UtoniaLidarEncoder

REPO_ROOT = ft.REPO_ROOT


def get_args():
    parser = argparse.ArgumentParser("Offline SAC HJ fine-tune on real G1 lidar")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of episode_*.npz")
    parser.add_argument(
        "--resume_policy",
        type=str,
        required=True,
        help="Sim-trained lidar .../epoch_id_N/policy.pth",
    )
    parser.add_argument(
        "--dino_ckpt_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "wm_ckpt_18-27-17"),
        help="WM run folder. Proprio encoder is checkpoints/model_latest.pth. "
        "Utonia defaults to <this>/utonia.pth.",
    )
    parser.add_argument(
        "--utonia_ckpt",
        type=str,
        default=None,
        help="utonia.pth. Default: <dino_ckpt_dir>/utonia.pth.",
    )
    parser.add_argument("--config", type=str, default="train_HJ_configs.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="Override yaml seed if set.")
    parser.add_argument(
        "--with_proprio",
        action="store_true",
        default=True,
        help="Concat proprio embedding after z_lidar (default on; match sim SF).",
    )
    parser.add_argument("--no_with_proprio", action="store_false", dest="with_proprio")
    parser.add_argument("--critic_fusion", type=str, default="concat", choices=["late", "concat"])
    parser.add_argument(
        "--z_fuse_frames",
        type=int,
        default=5,
        help="Average the last N full z vectors. Lidar sim default is 5.",
    )
    parser.add_argument("--yaw_limit", type=float, default=1.0)
    parser.add_argument("--freeze_yaw", action="store_true")
    parser.add_argument(
        "--freeze_vy",
        action="store_true",
        help="Match a sim run trained with --freeze_vy. vy stays 0 and is dropped from log π.",
    )
    parser.add_argument(
        "--lidar_distance_threshold",
        type=float,
        default=1.0,
        help="h_s = lidar360_min_distance - this. Default 1.0 m.",
    )
    parser.add_argument("--no_hit_hs", type=float, default=ft.NO_HIT_HS)
    parser.add_argument("--no_hit_dist", type=float, default=ft.NO_HIT_DIST_M)
    parser.add_argument(
        "--train_actor",
        action="store_true",
        default=True,
        help="Update the SF actor together with the critics (default).",
    )
    parser.add_argument(
        "--critics_only",
        action="store_false",
        dest="train_actor",
        help="Freeze the actor and update critics only.",
    )
    parser.add_argument(
        "--actor_reg_coef",
        type=float,
        default=0.0,
        help="λ for MSE(π(z), π_sim(z)). 0 = off. >0 requires --train_actor.",
    )
    parser.add_argument(
        "--actor_reg_when_unsafe",
        action="store_true",
        help="Only apply π_sim reg on transitions with h_s < 0.",
    )
    parser.add_argument("--auto_alpha", action="store_true", default=True)
    parser.add_argument("--no_auto_alpha", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--total_epochs", type=int, default=50)
    parser.add_argument(
        "--updates_per_epoch",
        type=int,
        default=None,
        help="Override gradient steps/epoch. Default matches sim trainer: "
        "step-per-epoch * update-per-step (2000*0.125=250).",
    )
    parser.add_argument(
        "--keep_ckpts",
        type=int,
        default=1,
        help="Unused. FT always overwrites a single policy.pth.",
    )
    parser.add_argument(
        "--wm_joint_names",
        type=str,
        default=None,
        help="Isaac/WM joint names, only if real proprio dim or order differs.",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sac-hj-utonia-humanoid")
    ft_args, remaining = parser.parse_known_args()

    with open(ft_args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = ft.args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    args = argparse.Namespace(**vars(cfg_args))
    for key, val in vars(ft_args).items():
        if key in ("seed", "updates_per_epoch") and val is None:
            continue
        setattr(args, key, val)
    if getattr(args, "seed", None) is None:
        args.seed = 0
    if not hasattr(args, "updates_per_epoch"):
        args.updates_per_epoch = None
    return args


def split_clouds(raw: dict[str, Any]) -> list[np.ndarray]:
    if "lidar_xyz" not in raw or "lidar_counts" not in raw:
        raise ValueError("episode missing 'lidar_xyz' or 'lidar_counts'")
    xyz = np.asarray(raw["lidar_xyz"], dtype=np.float32)
    counts = np.asarray(raw["lidar_counts"]).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"lidar_xyz must be (P, 3), got {xyz.shape}")
    if int(counts.sum()) != int(xyz.shape[0]):
        raise ValueError(
            f"lidar_counts sum {int(counts.sum())} != lidar_xyz rows {xyz.shape[0]}"
        )
    clouds: list[np.ndarray] = []
    start = 0
    for n in counts.tolist():
        n = int(n)
        clouds.append(np.ascontiguousarray(xyz[start : start + n]))
        start += n
    return clouds


def prepare_episode(
    raw: dict[str, Any],
    *,
    proprio_dim: int,
    dst_names: list[str] | None,
    threshold: float,
    no_hit_hs: float,
    no_hit_dist: float,
    min_frames: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
    if "act" not in raw:
        raise ValueError("episode missing 'act'")
    clouds = split_clouds(raw)
    act = np.asarray(raw["act"], dtype=np.float32).reshape(-1, 3)
    t = min(len(clouds), int(act.shape[0]))
    if "h_s" in raw:
        hs = np.asarray(raw["h_s"], dtype=np.float32).reshape(-1)[:t]
    elif "dist" in raw:
        hs = ft.dist_to_hs(
            np.asarray(raw["dist"]).reshape(-1)[:t], threshold, no_hit_hs, no_hit_dist
        )
    else:
        raise ValueError("episode needs 'dist' or 'h_s'")
    t = min(t, int(hs.shape[0]))
    if t < min_frames:
        return None
    clouds = clouds[:t]
    act = act[:t]
    hs = hs[:t].astype(np.float32)
    src_names = ft._as_name_list(raw.get("joint_names"))
    if "proprio" in raw:
        proprio = np.asarray(raw["proprio"], dtype=np.float32)
        if proprio.ndim == 1:
            proprio = np.repeat(proprio.reshape(1, -1), t, axis=0)
        proprio = ft.remap_proprio(proprio[:t], src_names, proprio_dim, dst_names)
    else:
        proprio = np.zeros((t, proprio_dim), dtype=np.float32)
    return clouds, proprio, act, hs


def fuse_episode(z: np.ndarray, n_frames: int) -> np.ndarray:
    """Mean of the last N full z vectors. First frame is repeated to fill the window."""
    n = max(1, int(n_frames))
    z = np.asarray(z, dtype=np.float32)
    if n <= 1:
        return z
    hist: deque[np.ndarray] = deque(maxlen=n)
    out = np.empty_like(z, dtype=np.float32)
    for i, zi in enumerate(z):
        zi = np.asarray(zi, dtype=np.float32).reshape(-1)
        if len(hist) == 0:
            for _ in range(n):
                hist.append(zi.copy())
        else:
            hist.append(zi.copy())
        out[i] = np.mean(np.stack(list(hist), axis=0), axis=0)
    return out


class OfflineLidarEncoder:
    """Match LatentHumanoidEnv._encode_lidar_utonia without Isaac."""

    def __init__(
        self,
        utonia: UtoniaLidarEncoder,
        proprio_encoder,
        with_proprio: bool,
        z_fuse_frames: int,
    ):
        self.utonia = utonia
        self.proprio_encoder = proprio_encoder
        self.with_proprio = bool(with_proprio)
        self.z_fuse_frames = max(1, int(z_fuse_frames))
        self.device = utonia.device

    @torch.no_grad()
    def encode_episode(self, clouds: list[np.ndarray], proprios: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        for i, pts in enumerate(clouds):
            z_lidar = self.utonia.encode(pts)
            if self.with_proprio:
                prop = np.asarray(proprios[i], dtype=np.float32).reshape(-1)
                expected = int(self.proprio_encoder.in_chans)
                if int(prop.shape[0]) != expected:
                    raise ValueError(
                        f"proprio dim {int(prop.shape[0])} != proprio encoder in_chans {expected}"
                    )
                prop_t = (
                    torch.from_numpy(np.ascontiguousarray(prop))
                    .view(1, 1, expected)
                    .to(self.device)
                )
                z_prop = self.proprio_encoder(prop_t).reshape(-1)
                z = torch.cat([z_lidar, z_prop], dim=0)
            else:
                z = z_lidar
            rows.append(z.detach().cpu().numpy().astype(np.float32))
        return fuse_episode(np.stack(rows, axis=0), self.z_fuse_frames)


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--resume_policy not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        policy.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint: match --critic_fusion / --with_proprio "
            "to the lidar sim run (z = Utonia 576 ‖ proprio). "
            f"Path: {ckpt_path}"
        ) from exc
    print(f"[INFO] Resumed policy from {ckpt_path}")


def main():
    args = get_args()
    actor_reg_coef = float(args.actor_reg_coef)
    if actor_reg_coef > 0.0 and not bool(args.train_actor):
        raise ValueError("--actor_reg_coef > 0 requires --train_actor")
    use_auto_alpha = bool(args.auto_alpha) and not bool(args.no_auto_alpha)
    if not bool(args.train_actor):
        use_auto_alpha = False

    args.critic_lr = float(args.critic_lr)
    args.actor_lr = float(args.actor_lr)
    args.tau = float(args.tau)
    args.gamma_pyhj = float(args.gamma_pyhj)
    args.batch_size_pyhj = int(args.batch_size_pyhj)
    args.buffer_size = int(args.buffer_size)
    args.dino_ckpt_dir = os.path.abspath(args.dino_ckpt_dir)
    args.dino_encoder = Path(args.dino_ckpt_dir).name
    step_per_epoch = int(args.step_per_epoch)
    update_per_step = float(args.update_per_step)
    if getattr(args, "updates_per_epoch", None) is None:
        updates_per_epoch = max(1, int(round(step_per_epoch * update_per_step)))
    else:
        updates_per_epoch = int(args.updates_per_epoch)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"--data_dir not found: {data_dir}")

    ckpt_dir = Path(args.dino_ckpt_dir)
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
    utonia_ckpt = args.utonia_ckpt or str(ckpt_dir / "utonia.pth")
    if not os.path.isfile(utonia_ckpt):
        raise FileNotFoundError(f"Utonia checkpoint not found: {utonia_ckpt}")
    args.utonia_ckpt = os.path.abspath(utonia_ckpt)
    if bool(args.with_proprio) and not snapshot.is_file():
        raise FileNotFoundError(f"WM checkpoint not found: {snapshot}")

    print(f"[INFO] Utonia: {args.utonia_ckpt}")
    utonia = UtoniaLidarEncoder(args.utonia_ckpt, args.device)
    proprio_encoder = None
    proprio_dim = 0
    if bool(args.with_proprio):
        payload = torch.load(snapshot, map_location=args.device, weights_only=False)
        proprio_encoder = payload["proprio_encoder"].to(args.device).eval()
        for p in proprio_encoder.parameters():
            p.requires_grad = False
        del payload
        proprio_dim = int(proprio_encoder.in_chans)
        print(
            f"[INFO] proprio encoder in_chans={proprio_dim} "
            f"emb_dim={int(proprio_encoder.emb_dim)} from {snapshot}"
        )
        print(
            f"[INFO] actor input = [z_lidar {utonia.out_dim}, z_proprio "
            f"{int(proprio_encoder.emb_dim)}]"
        )
    else:
        print(f"[INFO] actor input = z_lidar {utonia.out_dim} (no proprio)")

    dst_names = ft.load_name_file(args.wm_joint_names) if args.wm_joint_names else None
    encoder = OfflineLidarEncoder(
        utonia,
        proprio_encoder,
        bool(args.with_proprio),
        int(args.z_fuse_frames),
    )
    min_frames = max(2, int(args.z_fuse_frames) + 1)

    obs_list: list[np.ndarray] = []
    next_list: list[np.ndarray] = []
    act_env_list: list[np.ndarray] = []
    rew_list: list[np.ndarray] = []
    done_list: list[np.ndarray] = []
    n_skip = 0
    for path in tqdm(ft.list_episode_files(data_dir), desc="encode episodes"):
        prepared = prepare_episode(
            ft.load_episode_file(path),
            proprio_dim=proprio_dim,
            dst_names=dst_names,
            threshold=float(args.lidar_distance_threshold),
            no_hit_hs=float(args.no_hit_hs),
            no_hit_dist=float(args.no_hit_dist),
            min_frames=min_frames,
        )
        if prepared is None:
            n_skip += 1
            continue
        clouds, proprio, act_env, hs = prepared
        z = encoder.encode_episode(clouds, proprio)
        obs_list.append(z[:-1])
        next_list.append(z[1:])
        act_env_list.append(act_env[:-1])
        rew_list.append(hs[1:])
        done = np.zeros((z.shape[0] - 1,), dtype=np.bool_)
        done[-1] = True
        done_list.append(done)

    if not obs_list:
        raise RuntimeError(f"No usable episodes in {data_dir} (skipped {n_skip} short ones).")

    obs = np.concatenate(obs_list, axis=0)
    obs_next = np.concatenate(next_list, axis=0)
    act_env = np.concatenate(act_env_list, axis=0)
    rew = np.concatenate(rew_list, axis=0).astype(np.float32)
    done = np.concatenate(done_list, axis=0)
    n_trans = int(obs.shape[0])
    if n_trans < int(args.batch_size_pyhj):
        print(
            f"[WARN] only {n_trans} transitions; shrinking batch_size_pyhj "
            f"{args.batch_size_pyhj} -> {n_trans}"
        )
        args.batch_size_pyhj = n_trans
    z_dim = int(obs.shape[-1])
    unsafe_frac = float((rew < 0.0).mean())
    print(
        f"[INFO] transitions={n_trans} episodes_used={len(obs_list)} "
        f"skipped_short={n_skip} z_dim={z_dim} "
        f"h_s<0 frac={unsafe_frac:.3f} h_s mean={float(rew.mean()):.3f}"
    )

    try:
        import gym
    except ImportError:
        import gymnasium as gym

    action_low = np.array(
        [ft.ENV_VX_RANGE[0], ft.ENV_VY_RANGE[0], -float(args.yaw_limit)], dtype=np.float32
    )
    action_high = np.array(
        [ft.ENV_VX_RANGE[1], ft.ENV_VY_RANGE[1], float(args.yaw_limit)], dtype=np.float32
    )
    policy_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    state_shape = (z_dim,)
    action_shape = (3,)

    def _make_critic():
        critic = ft.make_q_critic(
            args.critic_fusion,
            state_shape,
            action_shape,
            hidden_sizes=args.critic_net,
            activation=getattr(torch.nn, args.critic_activation),
            device=args.device,
        )
        optim = torch.optim.AdamW(
            critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj
        )
        return critic, optim

    critic1, critic1_optim = _make_critic()
    critic2, critic2_optim = _make_critic()
    actor_net = ft.Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device,
    )
    actor1 = ft.ActorProb(
        actor_net,
        action_shape,
        max_action=1.0,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    actor1_optim = torch.optim.AdamW(actor1.parameters(), lr=args.actor_lr)
    if use_auto_alpha:
        n_ent = 2.0 if bool(args.freeze_vy) else float(np.prod(action_shape))
        target_entropy = -n_ent
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=float(args.alpha_lr))
        alpha_arg = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha_arg = float(args.alpha)

    policy = ft.avoid_SACPolicy_annealing(
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        alpha=alpha_arg,
        exploration_noise=None,
        deterministic_eval=True,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=policy_action_space,
        actor1=actor1,
        actor1_optim=actor1_optim,
    )
    policy.critic = policy.critic1
    policy.warmup = not bool(args.train_actor)
    load_policy_checkpoint(policy, args.resume_policy, args.device)

    freeze_vy = bool(args.freeze_vy)
    _orig_policy_forward = policy.forward

    def _sac_forward(batch, state=None, input="obs", **kwargs):
        out = _orig_policy_forward(batch, state=state, input=input, **kwargs)
        if freeze_vy:
            act = out.act
            if torch.is_tensor(act):
                out.act = ft._zero_vy_tensor(act)
                if getattr(out, "logits", None) is not None:
                    out.log_prob = ft._log_prob_without_vy(out)
            else:
                act_np = np.array(act, dtype=np.float32, copy=True)
                if act_np.shape[-1] >= 2:
                    act_np[..., 1] = 0.0
                out.act = act_np
        return out

    policy.forward = _sac_forward  # type: ignore[method-assign]

    act_env = np.clip(act_env, action_low, action_high)
    if freeze_vy and act_env.shape[-1] >= 2:
        act_env = np.array(act_env, dtype=np.float32, copy=True)
        act_env[:, 1] = 0.0
    act_pol = np.asarray(policy.map_action_inverse(act_env), dtype=np.float32)
    if bool(args.freeze_yaw):
        act_pol = act_pol.copy()
        act_pol[:, 2] = 0.0
    if freeze_vy and act_pol.shape[-1] >= 2:
        act_pol = np.array(act_pol, dtype=np.float32, copy=True)
        act_pol[:, 1] = 0.0

    buffer_size = max(int(args.buffer_size), n_trans)
    buffer = ft.VectorReplayBuffer(buffer_size, 1)
    ft.fill_replay_buffer(buffer, obs, obs_next, act_pol, rew, done)
    print(f"[INFO] replay buffer size={buffer_size} filled={len(buffer)}")

    pi_sim_policy = None
    if actor_reg_coef > 0.0:
        pi_sim_policy = copy.deepcopy(policy)
        pi_sim_policy.eval()
        for p in pi_sim_policy.parameters():
            p.requires_grad = False
        print(
            f"[INFO] π_sim frozen from resume ckpt; "
            f"λ={actor_reg_coef:g} unsafe_only={bool(args.actor_reg_when_unsafe)}"
        )

    freeze_yaw = bool(args.freeze_yaw)
    train_actor = bool(args.train_actor)

    def learn_offline(batch, **kwargs):
        del kwargs
        td1, critic1_loss = policy._mse_optimizer(batch, policy.critic1, policy.critic1_optim)
        td2, critic2_loss = policy._mse_optimizer(batch, policy.critic2, policy.critic2_optim)
        batch.weight = (td1 + td2) / 2.0

        actor_loss_v = 0.0
        pi_reg_v = 0.0
        alpha_loss_v = None
        if train_actor and not policy.warmup:
            obs_result = policy(batch)
            act = obs_result.act
            if not torch.is_tensor(act):
                act = torch.as_tensor(act, device=args.device, dtype=torch.float32)
            if freeze_yaw:
                act = ft._zero_yaw_tensor(act)
            if freeze_vy:
                act = ft._zero_vy_tensor(act)
            q = torch.min(
                policy.critic1(batch.obs, act).flatten(),
                policy.critic2(batch.obs, act).flatten(),
            )
            actor_loss = (policy._alpha * obs_result.log_prob.flatten() - q).mean()
            if pi_sim_policy is not None and actor_reg_coef > 0.0:
                act_sim = ft._frozen_pi_sim_act(pi_sim_policy, batch.obs, torch.device(args.device))
                if freeze_yaw:
                    act_sim = ft._zero_yaw_tensor(act_sim)
                if freeze_vy:
                    act_sim = ft._zero_vy_tensor(act_sim)
                sq = (act - act_sim).pow(2).mean(dim=-1)
                if bool(args.actor_reg_when_unsafe):
                    rew_t = torch.as_tensor(batch.rew, device=act.device, dtype=act.dtype).reshape(-1)
                    w = (rew_t < 0.0).to(act.dtype)
                    denom = w.sum().clamp(min=1.0)
                    pi_reg = (sq * w).sum() / denom
                    if float(w.sum().item()) > 0.0:
                        actor_loss = actor_loss + actor_reg_coef * pi_reg
                else:
                    pi_reg = sq.mean()
                    actor_loss = actor_loss + actor_reg_coef * pi_reg
                pi_reg_v = float(pi_reg.item())
            policy.actor1_optim.zero_grad()
            actor_loss.backward()
            policy.actor1_optim.step()
            if policy._is_auto_alpha:
                log_prob = obs_result.log_prob.detach() + policy._target_entropy
                alpha_loss = -(policy._log_alpha * log_prob).mean()
                policy._alpha_optim.zero_grad()
                alpha_loss.backward()
                policy._alpha_optim.step()
                policy._alpha = policy._log_alpha.detach().exp()
                alpha_loss_v = float(alpha_loss.item())
            actor_loss_v = float(actor_loss.item())

        policy.sync_weight()
        result = {
            "loss/actor": actor_loss_v,
            "loss/critic1": float(critic1_loss.item()),
            "loss/critic2": float(critic2_loss.item()),
            "loss/critic": float(0.5 * (critic1_loss.item() + critic2_loss.item())),
            "loss/pi_sim_reg": pi_reg_v,
        }
        if alpha_loss_v is not None:
            result["loss/alpha"] = alpha_loss_v
            result["train/alpha"] = float(policy._alpha.item())
        elif not policy._is_auto_alpha:
            result["train/alpha"] = float(policy._alpha)
        return result

    policy.learn = learn_offline  # type: ignore[method-assign]

    resume_epoch = ft._parse_resume_epoch_from_path(args.resume_policy)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    mode = "actor" if train_actor else "critics"
    if actor_reg_coef > 0.0:
        mode = f"{mode}-pisim"
    run_name = (
        f"ft-path-utonia-{args.critic_fusion}-{mode}"
        f"-{args.dino_encoder}-{timestamp}-resume{resume_epoch}"
    )
    print(
        f"[INFO] offline lidar FT: mode={mode} gamma={args.gamma_pyhj} "
        f"h_s=dist-{float(args.lidar_distance_threshold):g} "
        f"z_fuse={int(args.z_fuse_frames)} proprio={bool(args.with_proprio)} "
        f"freeze_vy={freeze_vy} "
        f"epochs={int(args.total_epochs)} "
        f"(sim match: {int(args.step_per_collect)} collect / "
        f"{int(args.step_per_epoch)} steps/epoch / "
        f"update_per_step={float(args.update_per_step):g} → "
        f"{updates_per_epoch} updates/epoch, batch={int(args.batch_size_pyhj)})"
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    log_root = Path("runs/sac_hj_humanoid_path_ft_lidar") / run_name
    tb_dir = log_root / "logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_root / "meta.json"
    policy_path = log_root / "policy.pth"
    run_meta: dict[str, Any] = {
        "run_name": run_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "policy_path": str(policy_path.resolve()),
        "latest_epoch": 0,
        "config": {k: ft._json_safe(v) for k, v in sorted(vars(args).items())},
        "policy": {
            "obs": "lidar_utonia",
            "critic_fusion": str(args.critic_fusion),
            "with_proprio": bool(args.with_proprio),
            "z_fuse_frames": int(args.z_fuse_frames),
            "z_dim": int(z_dim),
            "utonia_out_dim": int(utonia.out_dim),
            "yaw_limit": float(args.yaw_limit),
            "freeze_yaw": bool(args.freeze_yaw),
            "freeze_vy": bool(args.freeze_vy),
            "lidar_distance_threshold": float(args.lidar_distance_threshold),
            "train_actor": bool(args.train_actor),
            "actor_reg_coef": float(args.actor_reg_coef),
            "actor_reg_when_unsafe": bool(args.actor_reg_when_unsafe),
            "auto_alpha": bool(use_auto_alpha),
            "alpha": float(args.alpha),
            "gamma": float(args.gamma_pyhj),
        },
        "data": {
            "data_dir": str(data_dir.resolve()),
            "n_episodes": int(len(obs_list)),
            "n_transitions": int(n_trans),
            "n_skip_short": int(n_skip),
            "hs_unsafe_frac": float(unsafe_frac),
            "hs_mean": float(rew.mean()),
        },
        "wm": {
            "dino_ckpt_dir": str(ckpt_dir.resolve()),
            "dino_encoder": str(args.dino_encoder),
            "snapshot": str(snapshot.resolve()) if snapshot.is_file() else "",
            "utonia_ckpt": str(args.utonia_ckpt),
        },
        "resume": {
            "resume_policy": str(Path(args.resume_policy).resolve()),
            "resume_epoch": int(resume_epoch),
        },
        "trainer": {
            "total_epochs": int(args.total_epochs),
            "updates_per_epoch": int(updates_per_epoch),
            "batch_size": int(args.batch_size_pyhj),
            "step_per_collect": int(args.step_per_collect),
            "step_per_epoch": int(args.step_per_epoch),
            "update_per_step": float(args.update_per_step),
            "device": str(args.device),
            "seed": int(args.seed),
        },
        "last_metrics": {},
    }
    ft.write_run_meta(meta_path, run_meta)
    writer = SummaryWriter(log_dir=str(tb_dir))
    use_wandb = not bool(args.no_wandb)
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb.define_metric("trainer/update")
        wandb.define_metric("loss/*", step_metric="trainer/update")
    else:
        wandb = None  # type: ignore[assignment]

    global_update = 0
    policy.train()
    for epoch in range(1, int(args.total_epochs) + 1):
        losses_acc: dict[str, float] = {}
        n_up = 0
        pbar = tqdm(range(updates_per_epoch), desc=f"epoch {epoch}/{args.total_epochs}")
        for _ in pbar:
            result = policy.update(args.batch_size_pyhj, buffer)
            global_update += 1
            n_up += 1
            for k, v in result.items():
                losses_acc[k] = losses_acc.get(k, 0.0) + float(v)
            if n_up % 20 == 0:
                pbar.set_postfix(
                    c=f"{result.get('loss/critic', 0):.3f}",
                    a=f"{result.get('loss/actor', 0):.3f}",
                )
        mean_loss = {k: v / max(n_up, 1) for k, v in losses_acc.items()}
        payload = {"trainer/update": float(global_update), "epoch": float(epoch)}
        payload.update({k: float(v) for k, v in mean_loss.items()})
        print(
            f"[epoch {epoch}] critic={mean_loss.get('loss/critic', 0):.4f} "
            f"actor={mean_loss.get('loss/actor', 0):.4f} "
            f"pi_sim_reg={mean_loss.get('loss/pi_sim_reg', 0):.4f}"
        )
        writer.add_scalar("loss/critic", mean_loss.get("loss/critic", 0.0), epoch)
        writer.add_scalar("loss/actor", mean_loss.get("loss/actor", 0.0), epoch)
        if wandb is not None:
            wandb.log(payload, step=global_update)
        ft.save_latest_policy(policy, log_root)
        run_meta["latest_epoch"] = int(epoch)
        run_meta["updated"] = datetime.now().isoformat(timespec="seconds")
        run_meta["last_metrics"] = {k: float(v) for k, v in mean_loss.items()}
        ft.write_run_meta(meta_path, run_meta)

    writer.close()
    if wandb is not None:
        wandb.finish()
    print(f"[INFO] done. latest policy: {policy_path}")
    print(f"[INFO] meta: {meta_path}")


if __name__ == "__main__":
    main()
