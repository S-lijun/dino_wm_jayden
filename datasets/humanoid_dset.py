"""Dataset loader for Isaac G1 humanoid offline trajectories."""

import bisect
from collections import OrderedDict
from fractions import Fraction

import torch
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from datasets.img_transforms import get_transform_hw, letterbox_resize_and_normalize
from datasets.traj_dset import TrajDataset, get_train_val_sliced, split_traj_datasets

# State layout from IsaacG1Wrapper.get_full_state():
# [base_pos(3), base_quat(4), joint_pos(N), joint_vel(N), velocity_cmd(3), lidar_min(1)]
PROPRIO_START = 7


# Isaac G1 offline data: sim control ~200 Hz, camera obses_15fps @ 15 Hz.
SIM_HZ = 200
VISUAL_FPS = 15
STEPS_PER_VISUAL = Fraction(SIM_HZ, VISUAL_FPS)  # 200/15, not ~13


def visual_to_step_indices(
    step_len: int,
    visual_len: int,
    sim_hz: int = SIM_HZ,
    visual_fps: int = VISUAL_FPS,
) -> list[int]:
    """Map visual frame v -> sim step round(v * sim_hz / visual_fps), clipped."""
    if visual_len <= 0:
        return []
    step_per_visual = Fraction(sim_hz, visual_fps)
    last = step_len - 1
    return [
        min(int(round(float(Fraction(v) * step_per_visual))), last)
        for v in range(visual_len)
    ]


def step_to_visual_indices(step_len: int, visual_len: int) -> list[int]:
    """Hold each visual frame until the next one (upsample to sim-step timeline)."""
    if step_len <= 0:
        return []
    if visual_len <= 1:
        return [0] * step_len
    vis_starts = visual_to_step_indices(step_len, visual_len)
    mapping = []
    for t in range(step_len):
        vi = bisect.bisect_right(vis_starts, t) - 1
        mapping.append(max(0, min(vi, visual_len - 1)))
    return mapping


class HumanoidDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        normalize_states: bool = True,
        n_rollout: Optional[int] = None,
        with_costs: bool = True,
        only_cost: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.normalize_states = normalize_states
        self.with_costs = with_costs
        self.only_cost = only_cost

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.sim_seq_lengths = torch.load(self.data_path / "seq_lengths.pth").long()
        self.costs = torch.load(self.data_path / "costs.pth").float()
        self._visual_lengths: dict[int, int] = {}
        self._visual_to_step: dict[int, list[int]] = {}
        self._vis_aligned: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._vis_aligned_max = 32
        self.align_action_to_visual = True
        self._image_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._image_cache_max = 8
        self._img_h, self._img_w = get_transform_hw(transform)

        meta_path = self.data_path / "meta.pth"
        if meta_path.exists():
            meta = torch.load(meta_path)
            self.proprio_dim = int(meta["proprio_dim"])
        else:
            self.proprio_dim = int(self.states.shape[-1] - PROPRIO_START - self.actions.shape[-1] - 1)

        self.n_rollout = n_rollout
        n = min(n_rollout, len(self.sim_seq_lengths)) if n_rollout else len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.sim_seq_lengths = self.sim_seq_lengths[:n]
        self.costs = self.costs[:n]

        proprio_end = PROPRIO_START + self.proprio_dim
        self.proprios = self.states[..., PROPRIO_START:proprio_end].clone()

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.sim_seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.sim_seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.sim_seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def _resize_and_normalize(self, images: torch.Tensor) -> torch.Tensor:
        return letterbox_resize_and_normalize(images, self._img_h, self._img_w)

    def _load_visual_length(self, idx: int) -> int:
        if idx not in self._visual_lengths:
            obs_file = self.data_path / "obses" / f"episode_{idx}.pth"
            obs = torch.load(obs_file, map_location="cpu")
            self._visual_lengths[idx] = int(obs.shape[0])
        return self._visual_lengths[idx]

    def _get_visual_to_step(self, idx: int) -> list[int]:
        """Map each 15fps visual frame index to its aligned sim step."""
        if idx not in self._visual_to_step:
            step_len = int(self.sim_seq_lengths[idx].item())
            visual_len = self._load_visual_length(idx)
            self._visual_to_step[idx] = visual_to_step_indices(step_len, visual_len)
        return self._visual_to_step[idx]

    def _get_vis_aligned(self, idx: int) -> dict[str, torch.Tensor]:
        """State/action/proprio/cost downsampled to 15fps visual length (1:1 with obs)."""
        cached = self._vis_aligned.get(idx)
        if cached is not None:
            self._vis_aligned.move_to_end(idx)
            return cached

        steps = torch.tensor(self._get_visual_to_step(idx), dtype=torch.long)
        aligned = {
            "states": self.states[idx].index_select(0, steps),
            "actions": self.actions[idx].index_select(0, steps),
            "proprios": self.proprios[idx].index_select(0, steps),
            "costs": self.costs[idx].index_select(0, steps),
        }
        self._vis_aligned[idx] = aligned
        self._vis_aligned.move_to_end(idx)
        while len(self._vis_aligned) > self._vis_aligned_max:
            self._vis_aligned.popitem(last=False)
        return aligned

    def get_seq_length(self, idx):
        """Trajectory length = number of 15fps visual frames (all used, no skip)."""
        return self._load_visual_length(idx)

    def get_all_actions(self):
        result = []
        for i in range(len(self.sim_seq_lengths)):
            result.append(self._get_vis_aligned(i)["actions"])
        return torch.cat(result, dim=0)

    def _load_episode_images(self, idx: int) -> torch.Tensor:
        cached = self._image_cache.get(idx)
        if cached is not None:
            self._image_cache.move_to_end(idx)
            return cached

        obs_file = self.data_path / "obses" / f"episode_{idx}.pth"
        raw = torch.load(obs_file, map_location="cpu")
        self._visual_lengths[idx] = int(raw.shape[0])

        chunks = []
        chunk_size = 8
        for start in range(0, raw.shape[0], chunk_size):
            chunk = raw[start : start + chunk_size].float().div_(255.0)
            chunk = rearrange(chunk, "T H W C -> T C H W")
            chunks.append(self._resize_and_normalize(chunk))
        images = torch.cat(chunks, dim=0)
        del raw, chunks

        self._image_cache[idx] = images
        self._image_cache.move_to_end(idx)
        while len(self._image_cache) > self._image_cache_max:
            self._image_cache.popitem(last=False)
        return images

    def get_frames(self, idx, frames, act_frames=None):
        """Index visual-frame timeline: one image, state, and action per 15fps frame."""
        frames = list(frames)
        act_frames = list(act_frames) if act_frames is not None else frames
        aligned = self._get_vis_aligned(idx)

        if not self.only_cost:
            images = self._load_episode_images(idx)
            image = images[frames]
            actions = aligned["actions"][act_frames]
            full_states = aligned["states"][frames]
            proprio_states = aligned["proprios"][frames]
            costs = aligned["costs"][frames]

            obs = {"visual": image, "proprio": proprio_states}
            return obs, actions, full_states, {
                "cost": (costs > 0).long(),
                "h": -costs + 0.5,
            }

        costs = aligned["costs"][frames]
        return None, None, None, {
            "cost": (costs > 0).long(),
            "h": -costs + 0.5,
        }

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.sim_seq_lengths)

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            all_data.append(data[traj, :traj_len])
        all_data = torch.vstack(all_data)
        return torch.mean(all_data, dim=0), torch.std(all_data, dim=0)


def load_humanoid_slice_train_val(
    transform,
    data_path,
    n_rollout=50,
    normalize_action=True,
    normalize_states=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=1,
    window_stride=1,
    with_costs=True,
):
    if frameskip != 1:
        raise ValueError(
            "humanoid uses 15fps visual frames with 1:1 state/action alignment; frameskip must be 1"
        )
    dset = HumanoidDataset(
        data_path=data_path,
        transform=transform,
        normalize_action=normalize_action,
        normalize_states=normalize_states,
        n_rollout=n_rollout,
        with_costs=with_costs,
    )

    train_dset, val_dset = split_traj_datasets(dset, train_fraction=split_ratio)

    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
        window_stride=window_stride,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
