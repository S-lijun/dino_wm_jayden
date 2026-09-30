"""Offline real-robot fine-tune of a sim-trained path SAC safety filter.

No Isaac / no collect. Load 15 Hz episodes, encode with the same frozen WM
as training, dump every transition into a replay buffer, resume the sim SF,
then update.

Default: critics only.
Optional: ``--train_actor`` and ``--actor_reg_coef 0.2`` (MSE to frozen π_sim(z)).

Episode files (see conversation spec)::

    real_data/meta.json
    real_data/episode_000.npz   # visual, proprio, joint_names, act, dist

Example::

    python finetune_HJ_humanoid_sac_path.py \\
      --data_dir C:\\path\\to\\real_data \\
      --resume_policy C:\\dino_wm_jayden\\runs\\...\\epoch_id_90\\policy.pth \\
      --dino_ckpt_dir C:\\dino_wm_jayden\\wm_ckpt_18-27-17 \\
      --visual_feature cls --critic_fusion concat --with_proprio \\
      --z_fuse_frames 10 --lidar_distance_threshold 1.0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, REPO_ROOT)

from PyHJ.data import Batch, VectorReplayBuffer
from PyHJ.policy import avoid_SACPolicy_annealing
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import ActorProb

from env.isaac.ckpt_utils import save_epoch_checkpoint
from env.isaac.late_fusion_critic import make_q_critic
from wm_load import load_model

ENV_VX_RANGE = (0.0, 1.0)
ENV_VY_RANGE = (-0.5, 0.5)
NO_HIT_HS = 2.0
NO_HIT_DIST_M = 10.0


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


def get_args():
    parser = argparse.ArgumentParser("Offline SAC HJ fine-tune on real G1 data")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of episode_*.npz/.pth")
    parser.add_argument(
        "--resume_policy",
        type=str,
        required=True,
        help="Sim-trained .../epoch_id_N/policy.pth",
    )
    parser.add_argument(
        "--dino_ckpt_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "wm_ckpt_18-27-17"),
        help="WM run folder (contains hydra.yaml and checkpoints/).",
    )
    parser.add_argument("--config", type=str, default="train_HJ_configs.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="Override yaml seed if set.")
    parser.add_argument(
        "--with_proprio",
        action="store_true",
        default=True,
        help="Concat proprio embedding into z (default on; match sim SF).",
    )
    parser.add_argument("--no_with_proprio", action="store_false", dest="with_proprio")
    parser.add_argument("--visual_feature", type=str, default="cls", choices=["patch", "cls"])
    parser.add_argument("--critic_fusion", type=str, default="concat", choices=["late", "concat"])
    parser.add_argument("--z_fuse_frames", type=int, default=10)
    parser.add_argument("--yaw_limit", type=float, default=1.0)
    parser.add_argument("--freeze_yaw", action="store_true")
    parser.add_argument(
        "--lidar_distance_threshold",
        type=float,
        default=1.0,
        help="h_s = dist - this. Real-robot default 1.0 m.",
    )
    parser.add_argument(
        "--no_hit_hs",
        type=float,
        default=NO_HIT_HS,
        help="h_s when dist is inf / >= --no_hit_dist.",
    )
    parser.add_argument("--no_hit_dist", type=float, default=NO_HIT_DIST_M)
    parser.add_argument(
        "--train_actor",
        action="store_true",
        help="Also update the SF actor. Default: critics only.",
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
    parser.add_argument("--keep_ckpts", type=int, default=5)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument(
        "--wm_joint_names",
        type=str,
        default=None,
        help="Text/json list of Isaac/WM joint names (one per proprio dim). "
        "Needed only if real proprio dim or order differs.",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sac-hj-latent-humanoid")
    ft_args, remaining = parser.parse_known_args()

    with open(ft_args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
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


def _parse_resume_epoch_from_path(policy_path: str | Path) -> int:
    for part in Path(policy_path).resolve().parts[::-1]:
        m = re.fullmatch(r"epoch_id_(\d+)", part)
        if m:
            return int(m.group(1))
    raise ValueError(
        f"Cannot parse epoch from --resume_policy (expected .../epoch_id_N/policy.pth): "
        f"{policy_path}"
    )


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--resume_policy not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        policy.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint: match --visual_feature / --critic_fusion "
            f"/ --with_proprio to sim training. Path: {ckpt_path}"
        ) from exc
    print(f"[INFO] Resumed policy from {ckpt_path}")


def _as_name_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (bytes, str)):
        text = value.decode() if isinstance(value, bytes) else value
        text = text.strip()
        if text.startswith("["):
            return [str(x) for x in json.loads(text)]
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    arr = np.asarray(value)
    if arr.dtype == object or arr.ndim >= 1:
        return [str(x) for x in arr.reshape(-1).tolist()]
    return [str(arr)]


def load_name_file(path: str | Path) -> list[str]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("joint_names") or data.get("names")
        return [str(x) for x in data]
    names = _as_name_list(raw)
    if not names:
        raise ValueError(f"No joint names in {path}")
    return names


def wm_proprio_in_dim(wm) -> int:
    enc = wm.proprio_encoder
    if hasattr(enc, "in_chans"):
        return int(enc.in_chans)
    pe = getattr(enc, "patch_embed", None)
    if pe is not None and hasattr(pe, "weight"):
        return int(pe.weight.shape[1])
    raise RuntimeError("Cannot infer WM proprio input dim from proprio_encoder.")


def remap_proprio(
    values: np.ndarray,
    src_names: list[str] | None,
    dst_dim: int,
    dst_names: list[str] | None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[-1] == dst_dim and dst_names is None:
        return values
    if values.shape[-1] == dst_dim and (
        src_names is None or dst_names is None or list(src_names) == list(dst_names)
    ):
        return values
    if src_names is None or dst_names is None:
        raise ValueError(
            f"proprio dim {values.shape[-1]} != WM dim {dst_dim}. "
            "Save joint_names in each episode and pass --wm_joint_names."
        )
    if len(dst_names) != dst_dim:
        raise ValueError(
            f"--wm_joint_names has {len(dst_names)} names but WM proprio dim is {dst_dim}."
        )
    index = {n: i for i, n in enumerate(src_names)}
    out = np.zeros((values.shape[0], dst_dim), dtype=np.float32)
    missing = [n for n in dst_names if n not in index]
    if missing:
        raise ValueError(f"Real data missing WM joints: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    extra = [n for n in src_names if n not in set(dst_names)]
    if extra:
        print(f"[INFO] Ignoring extra real joints: {extra[:8]}{'...' if len(extra) > 8 else ''}")
    for j, name in enumerate(dst_names):
        out[:, j] = values[:, index[name]]
    return out


def dist_to_hs(dist: np.ndarray, threshold: float, no_hit_hs: float, no_hit_dist: float) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float32).reshape(-1)
    hs = d - float(threshold)
    no_hit = ~np.isfinite(d) | (d >= float(no_hit_dist))
    hs[no_hit] = float(no_hit_hs)
    return hs.astype(np.float32)


def _visual_to_uint8_hwc(visual: Any) -> np.ndarray:
    if isinstance(visual, torch.Tensor):
        arr = visual.detach().cpu().numpy()
    else:
        arr = np.asarray(visual)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC or CxHxW visual, got {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def _visual_to_normalized_chw(visual: Any) -> np.ndarray:
    hwc = _visual_to_uint8_hwc(visual)
    chw = np.transpose(hwc, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray((chw - 0.5) / 0.5, dtype=np.float32)


class OfflineLatentEncoder:
    """Match LatentHumanoidEnv.encode without Isaac."""

    def __init__(self, wm, device: str, visual_feature: str, with_proprio: bool, z_fuse_frames: int):
        self.wm = wm
        self.device = torch.device(device)
        self.visual_feature = str(visual_feature).lower()
        self.with_proprio = bool(with_proprio)
        self.z_fuse_frames = max(1, int(z_fuse_frames))
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

    def _flatten(self, lat: dict[str, torch.Tensor]) -> np.ndarray:
        z_vis = lat["visual"].reshape(lat["visual"].shape[0], -1)
        if self.with_proprio:
            z_prop = lat["proprio"].reshape(z_vis.shape[0], -1)
            z = torch.cat([z_vis, z_prop], dim=-1)
        else:
            z = z_vis
        return z.detach().cpu().numpy().astype(np.float32)

    def _encode_obs_visual(self, vis_t: torch.Tensor, prop_t: torch.Tensor):
        if self.visual_feature != "cls":
            return self.wm.encode_obs({"visual": vis_t, "proprio": prop_t})
        enc = getattr(self.wm, "encoder", None)
        if enc is None or not hasattr(enc, "feature_key"):
            raise RuntimeError("visual_feature=cls needs DinoV2Encoder.feature_key")
        old_key = enc.feature_key
        old_ndim = enc.latent_ndim
        enc.feature_key = "x_norm_clstoken"
        enc.latent_ndim = 1
        try:
            return self.wm.encode_obs({"visual": vis_t, "proprio": prop_t})
        finally:
            enc.feature_key = old_key
            enc.latent_ndim = old_ndim

    @torch.no_grad()
    def encode_frames(self, visuals: np.ndarray, proprios: np.ndarray, batch_size: int) -> np.ndarray:
        t = int(visuals.shape[0])
        chunks: list[np.ndarray] = []
        for start in range(0, t, batch_size):
            sl = slice(start, min(start + batch_size, t))
            vis_np = np.stack([_visual_to_normalized_chw(v) for v in visuals[sl]], axis=0)
            vis_t = torch.from_numpy(vis_np).unsqueeze(1).to(self.device)
            prop = np.asarray(proprios[sl], dtype=np.float32)
            prop_t = torch.from_numpy(prop).unsqueeze(1).to(self.device)
            lat = self._encode_obs_visual(vis_t, prop_t)
            chunks.append(self._flatten(lat))
        z = np.concatenate(chunks, axis=0)
        return self._fuse_episode(z)

    def _fuse_episode(self, z: np.ndarray) -> np.ndarray:
        n = self.z_fuse_frames
        if n <= 1:
            return z.astype(np.float32, copy=False)
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


def load_episode_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a dict checkpoint")
    return payload


def list_episode_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("episode_*.npz")) + sorted(data_dir.glob("episode_*.pth"))
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No episode_*.npz / episode_*.pth in {data_dir}")
    return files


def prepare_episode(
    raw: dict[str, Any],
    *,
    proprio_dim: int,
    dst_names: list[str] | None,
    threshold: float,
    no_hit_hs: float,
    no_hit_dist: float,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if "visual" not in raw or "act" not in raw:
        raise ValueError("episode missing 'visual' or 'act'")
    visual = np.asarray(raw["visual"])
    if visual.ndim != 4:
        raise ValueError(f"visual must be (T,H,W,3) or (T,3,H,W), got {visual.shape}")
    act = np.asarray(raw["act"], dtype=np.float32).reshape(-1, 3)
    t = min(int(visual.shape[0]), int(act.shape[0]))
    if "h_s" in raw:
        hs = np.asarray(raw["h_s"], dtype=np.float32).reshape(-1)
        hs = hs[:t]
    elif "dist" in raw:
        hs = dist_to_hs(np.asarray(raw["dist"]).reshape(-1)[:t], threshold, no_hit_hs, no_hit_dist)
    else:
        raise ValueError("episode needs 'dist' or 'h_s'")
    t = min(t, int(hs.shape[0]))
    if t < min_frames:
        return None
    visual = visual[:t]
    act = act[:t]
    hs = hs[:t].astype(np.float32)
    src_names = _as_name_list(raw.get("joint_names"))
    if "proprio" in raw:
        proprio = np.asarray(raw["proprio"], dtype=np.float32)
        if proprio.ndim == 1:
            proprio = np.repeat(proprio.reshape(1, -1), t, axis=0)
        proprio = proprio[:t]
        proprio = remap_proprio(proprio, src_names, proprio_dim, dst_names)
    else:
        proprio = np.zeros((t, proprio_dim), dtype=np.float32)
    return visual, proprio, act, hs


def _zero_yaw_tensor(act: torch.Tensor) -> torch.Tensor:
    if act.shape[-1] < 3:
        return act
    out = act.clone()
    out[..., 2] = 0.0
    return out


def _frozen_pi_sim_act(pi_sim_policy, obs, device: torch.device) -> torch.Tensor:
    """Deterministic π_sim(z) in policy space (eval + tanh mean)."""
    was = pi_sim_policy.training
    pi_sim_policy.eval()
    with torch.no_grad():
        out = pi_sim_policy(Batch(obs=obs, info=Batch()))
        act = out.act
        if not torch.is_tensor(act):
            act = torch.as_tensor(act, device=device, dtype=torch.float32)
        else:
            act = act.to(device=device, dtype=torch.float32)
        if act.ndim == 1:
            act = act.unsqueeze(0)
    pi_sim_policy.train(was)
    return act.detach()


def fill_replay_buffer(
    buffer: VectorReplayBuffer,
    obs: np.ndarray,
    obs_next: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    done: np.ndarray,
) -> None:
    n = int(obs.shape[0])
    for i in tqdm(range(n), desc="buffer.add", leave=False):
        info = Batch(
            episode_step=np.array([i], dtype=np.int32),
            end_reason=np.array([1 if bool(done[i]) else 0], dtype=np.int32),
            stuck=np.array([0.0], dtype=np.float32),
            skip_update=np.array([0.0], dtype=np.float32),
        )
        batch = Batch(
            obs=obs[i : i + 1],
            obs_next=obs_next[i : i + 1],
            act=act[i : i + 1],
            rew=rew[i : i + 1],
            terminated=np.array([False]),
            truncated=np.array([bool(done[i])]),
            done=np.array([bool(done[i])]),
            info=info,
        )
        buffer.add(batch)
    if len(buffer) != n:
        print(f"[WARN] buffer len={len(buffer)} != transitions={n}")


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
    # Same as PyHJ offpolicy_trainer: collect `step_per_epoch` env steps in
    # chunks of `step_per_collect`, with `update_per_step` updates per env step.
    # Offline skips collect; one epoch = that many policy.update calls.
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
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
    if not hydra_cfg.is_file():
        raise FileNotFoundError(f"WM hydra.yaml not found: {hydra_cfg}")
    if not snapshot.is_file():
        raise FileNotFoundError(f"WM checkpoint not found: {snapshot}")

    print(f"[INFO] Loading WM from {ckpt_dir}")
    train_cfg = OmegaConf.load(str(hydra_cfg))
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=args.device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    proprio_dim = wm_proprio_in_dim(wm)
    dst_names = load_name_file(args.wm_joint_names) if args.wm_joint_names else None
    encoder = OfflineLatentEncoder(
        wm,
        args.device,
        args.visual_feature,
        args.with_proprio,
        args.z_fuse_frames,
    )
    min_frames = max(2, int(args.z_fuse_frames) + 1)

    obs_list: list[np.ndarray] = []
    next_list: list[np.ndarray] = []
    act_env_list: list[np.ndarray] = []
    rew_list: list[np.ndarray] = []
    done_list: list[np.ndarray] = []
    n_skip = 0
    for path in tqdm(list_episode_files(data_dir), desc="encode episodes"):
        prepared = prepare_episode(
            load_episode_file(path),
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
        visual, proprio, act_env, hs = prepared
        z = encoder.encode_frames(visual, proprio, int(args.encode_batch_size))
        # Post-step cost: (z_t, a_t, h_s_{t+1}, z_{t+1}), matching sim step-then-label.
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
        [ENV_VX_RANGE[0], ENV_VY_RANGE[0], -float(args.yaw_limit)], dtype=np.float32
    )
    action_high = np.array(
        [ENV_VX_RANGE[1], ENV_VY_RANGE[1], float(args.yaw_limit)], dtype=np.float32
    )
    policy_action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    state_shape = (z_dim,)
    action_shape = (3,)

    def _make_critic():
        critic = make_q_critic(
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
    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device,
    )
    actor1 = ActorProb(
        actor_net,
        action_shape,
        max_action=1.0,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    actor1_optim = torch.optim.AdamW(actor1.parameters(), lr=args.actor_lr)
    if use_auto_alpha:
        target_entropy = float(-np.prod(action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=float(args.alpha_lr))
        alpha_arg = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha_arg = float(args.alpha)

    policy = avoid_SACPolicy_annealing(
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

    act_env = np.clip(act_env, action_low, action_high)
    act_pol = np.asarray(policy.map_action_inverse(act_env), dtype=np.float32)
    if bool(args.freeze_yaw):
        act_pol = act_pol.copy()
        act_pol[:, 2] = 0.0

    buffer_size = max(int(args.buffer_size), n_trans)
    buffer = VectorReplayBuffer(buffer_size, 1)
    fill_replay_buffer(buffer, obs, obs_next, act_pol, rew, done)
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
                act = _zero_yaw_tensor(act)
            q = torch.min(
                policy.critic1(batch.obs, act).flatten(),
                policy.critic2(batch.obs, act).flatten(),
            )
            actor_loss = (policy._alpha * obs_result.log_prob.flatten() - q).mean()
            if pi_sim_policy is not None and actor_reg_coef > 0.0:
                act_sim = _frozen_pi_sim_act(pi_sim_policy, batch.obs, torch.device(args.device))
                if freeze_yaw:
                    act_sim = _zero_yaw_tensor(act_sim)
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

    resume_epoch = _parse_resume_epoch_from_path(args.resume_policy)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    mode = "actor" if train_actor else "critics"
    if actor_reg_coef > 0.0:
        mode = f"{mode}-pisim"
    run_name = (
        f"ft-path-{args.visual_feature}-{args.critic_fusion}-{mode}"
        f"-{args.dino_encoder}-{timestamp}-resume{resume_epoch}"
    )
    print(
        f"[INFO] offline FT: mode={mode} gamma={args.gamma_pyhj} "
        f"h_s=dist-{float(args.lidar_distance_threshold):g} "
        f"z_fuse={int(args.z_fuse_frames)} proprio={bool(args.with_proprio)} "
        f"epochs={int(args.total_epochs)} "
        f"(sim match: {int(args.step_per_collect)} collect / "
        f"{int(args.step_per_epoch)} steps/epoch / "
        f"update_per_step={float(args.update_per_step):g} → "
        f"{updates_per_epoch} updates/epoch, batch={int(args.batch_size_pyhj)})"
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    log_root = Path("runs/sac_hj_humanoid_path_ft") / run_name
    tb_dir = log_root / "logs"
    tb_dir.mkdir(parents=True, exist_ok=True)
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
        save_epoch_checkpoint(policy, log_root, epoch, keep=int(args.keep_ckpts))

    writer.close()
    if wandb is not None:
        wandb.finish()
    print(f"[INFO] done. checkpoints in {log_root}")


if __name__ == "__main__":
    main()
