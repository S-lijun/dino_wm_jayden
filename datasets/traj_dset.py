import abc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange

# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

class TrajSubset(TrajDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def _resolve_episode(dataset: TrajDataset, subset_idx: int):
    if isinstance(dataset, TrajSubset):
        return dataset.dataset, dataset.indices[subset_idx]
    return dataset, subset_idx


def _fetch_traj_window(dataset: TrajDataset, subset_idx: int, start: int, end: int, frameskip: int):
    """Load only the requested window instead of an entire trajectory."""
    base, ep_idx = _resolve_episode(dataset, subset_idx)
    obs_frames = list(range(start, end, frameskip))
    if getattr(base, "align_action_to_visual", False):
        act_frames = obs_frames
    else:
        act_frames = list(range(start, end))

    if hasattr(base, "get_frames"):
        result = base.get_frames(ep_idx, obs_frames, act_frames=act_frames)
        if len(result) == 4:
            obs, act, state, info = result
        else:
            obs, act, state = result
            info = None
        return obs, act, state, info

    obs, act, state, info = base[ep_idx]
    for k, v in obs.items():
        obs[k] = v[start:end:frameskip]
    state = state[start:end:frameskip]
    act = act[start:end]
    return obs, act, state, info


class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
        window_stride: int = 1,
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.window_stride = max(1, int(window_stride))
        self.slices = []
        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            if T - num_frames * frameskip < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(0, T - num_frames * frameskip + 1, self.window_stride)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        self.slices = np.random.permutation(self.slices)

        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, _ = _fetch_traj_window(self.dataset, i, start, end, self.frameskip)
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])

class TrajSlicerWithCostDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
        only_cost: bool = False,
        window_stride: int = 1,
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.window_stride = max(1, int(window_stride))
        self.slices = []
        self.only_cost = only_cost

        for i in range(len(self.dataset)):
            T = self.dataset.get_seq_length(i)
            if T - num_frames * frameskip < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(0, T - num_frames * frameskip + 1, self.window_stride)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        # self.slices = np.random.permutation(self.slices)
        
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, info = _fetch_traj_window(self.dataset, i, start, end, self.frameskip)
        if not self.only_cost:
            act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)
            cost = info["cost"]
            return tuple([obs, act, state, cost, info["h"]])
        cost = info["cost"]
        return tuple([obs, act, state, cost, info["h"]])


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    print(
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    )
    return [
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def random_split_traj_two(
    dataset: TrajDataset,
    dataset_cost: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    print(
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    )
    return (
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ), (
        TrajSubset(dataset_cost, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    )

def split_traj_datasets_two(dataset, dataset_cost, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    generator = torch.Generator().manual_seed(random_seed)
    (train_set, val_set), (train_set_cost, val_set_cost) = random_split_traj_two(
        dataset, dataset_cost, lengths, generator=generator
    )
    return train_set, val_set, train_set_cost, val_set_cost

def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
    window_stride: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_slices = TrajSlicerDataset(train, num_frames, frameskip, window_stride=window_stride)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip, window_stride=window_stride)
    return train, val, train_slices, val_slices

def get_train_val_sliced_with_cost(
    traj_dataset: TrajDataset,
    traj_dataset_cost: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
):
    train, val, train_cost, val_cost = split_traj_datasets_two(
        traj_dataset,
        traj_dataset_cost,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_slices = TrajSlicerWithCostDataset(train, num_frames, frameskip, only_cost=False)
    val_slices = TrajSlicerWithCostDataset(val, num_frames, frameskip, only_cost=False)
    train_slices_cost = TrajSlicerWithCostDataset(train_cost, num_frames, frameskip, only_cost=True)
    val_slices_cost = TrajSlicerWithCostDataset(val_cost, num_frames, frameskip, only_cost=True)
    return train_slices, val_slices, train_slices_cost, val_slices_cost