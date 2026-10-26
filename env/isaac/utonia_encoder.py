"""Frozen Utonia encoder: sensor-frame xyz -> one 576-D vector.

Color and normals are zeros. Voxel sampling keeps the first point in each
1 cm cell so the same cloud always maps to the same feature.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_UTONIA_ROOT = os.path.join(_REPO_ROOT, "third_party", "Utonia")
if _UTONIA_ROOT not in sys.path:
    sys.path.insert(0, _UTONIA_ROOT)

from utonia.model import PointTransformerV3  # noqa: E402
from utonia.transform import (  # noqa: E402
    Collect,
    GridSample,
    NormalizeColor,
    ToTensor,
    index_operator,
)


class _FirstPointGridSample(GridSample):
    """One point per voxel, the first after the spatial hash. No RNG."""

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
        idx_unique = idx_sort[idx_select]
        data_dict = index_operator(data_dict, idx_unique)
        if self.return_grid_coord:
            data_dict["grid_coord"] = grid_coord[idx_unique]
            data_dict["index_valid_keys"].append("grid_coord")
        return data_dict


class UtoniaLidarEncoder:
    """Load ``utonia.pth`` and pool the deepest point features."""

    GRID_SIZE = 0.01

    def __init__(self, ckpt_path: str, device: str, grid_size: float | None = None) -> None:
        path = os.path.abspath(ckpt_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Utonia checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"])
        cfg["enable_flash"] = False
        cfg["shuffle_orders"] = False
        self.model = PointTransformerV3(**cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        # GridPooling defaults shuffle_orders=True and is not given the ctor flag.
        # That randperm changes the serialized order every forward.
        for module in self.model.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        channels = tuple(cfg["enc_channels"])
        self.out_dim = int(channels[-1])
        self.grid_size = float(self.GRID_SIZE if grid_size is None else grid_size)
        if self.grid_size <= 0.0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        self._grid = _FirstPointGridSample(
            grid_size=self.grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        )
        self._color_norm = NormalizeColor()
        self._to_tensor = ToTensor()
        self._collect = Collect(
            keys=("coord", "grid_coord", "color"),
            feat_keys=("coord", "color", "normal"),
        )
        print(
            f"[UtoniaLidarEncoder] {path} out_dim={self.out_dim} "
            f"grid_size={self.grid_size:g}m flash=off shuffle_orders=off "
            f"device={self.device}"
        )

    def encode(self, xyz: np.ndarray) -> torch.Tensor:
        """Return a (out_dim,) feature on ``self.device``. Empty input is zeros."""
        pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return torch.zeros(self.out_dim, device=self.device, dtype=torch.float32)
        n = int(pts.shape[0])
        data = {
            "coord": pts,
            "color": np.zeros((n, 3), dtype=np.float32),
            "normal": np.zeros((n, 3), dtype=np.float32),
            "index_valid_keys": ["coord", "color", "normal"],
        }
        data = self._grid(data)
        data = self._color_norm(data)
        data = self._to_tensor(data)
        data = self._collect(data)
        for key, value in list(data.items()):
            if torch.is_tensor(value):
                data[key] = value.to(self.device, non_blocking=True)
        with torch.no_grad():
            point = self.model(data)
            feat = point.feat
            if feat.ndim != 2 or feat.shape[-1] != self.out_dim:
                raise RuntimeError(
                    f"Utonia feat shape {tuple(feat.shape)}, expected (*, {self.out_dim})"
                )
            return feat.max(dim=0).values
