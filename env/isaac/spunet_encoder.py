"""Frozen ScanNet SpUNet: sensor-frame xyz -> one 96-D vector.

The 20-class head is loaded so the checkpoint matches, and it is not used.
Color and normals are zeros. Voxels are 2 cm, the ScanNet grid.
"""

from __future__ import annotations

import os

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from torch import nn


class _BN(nn.Module):
    """Checkpoint stores BatchNorm under ``.module``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.module = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.module(features)


class _ConvBN(nn.Module):
    def __init__(self, conv: nn.Module, channels: int) -> None:
        super().__init__()
        self.add_module("0", conv)
        self.add_module("1", _BN(channels))

    @property
    def conv(self) -> nn.Module:
        return self._modules["0"]

    @property
    def bn(self) -> _BN:
        return self._modules["1"]


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, indice_key: str) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            in_channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.norm1 = _BN(channels)
        self.conv2 = spconv.SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.norm2 = _BN(channels)
        self.proj = None
        if in_channels != channels:
            self.proj = _ConvBN(
                spconv.SubMConv3d(in_channels, channels, kernel_size=1, bias=False),
                channels,
            )
        self.relu = nn.ReLU()

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        residual = x
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.norm1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        if self.proj is not None:
            residual = self.proj.conv(residual)
            residual = residual.replace_feature(self.proj.bn(residual.features))
        out = out.replace_feature(self.relu(out.features + residual.features))
        return out


def _stage(blocks: list[_ResBlock]) -> nn.Module:
    stage = nn.Module()
    for i, block in enumerate(blocks):
        stage.add_module(f"block{i}", block)
    return stage


class _SpUNetV1M1(nn.Module):
    """ScanNet SpUNet-v1m1. ``head`` is the 20-class layer and is not called."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.decoder = nn.Module()
        self.encoder.conv_input = _ConvBN(
            spconv.SubMConv3d(6, 32, kernel_size=5, padding=1, bias=False, indice_key="stem"),
            32,
        )
        self.encoder.down = nn.ModuleList(
            [
                _ConvBN(spconv.SparseConv3d(32, 32, kernel_size=2, stride=2, bias=False, indice_key="spconv1"), 32),
                _ConvBN(spconv.SparseConv3d(32, 64, kernel_size=2, stride=2, bias=False, indice_key="spconv2"), 64),
                _ConvBN(spconv.SparseConv3d(64, 128, kernel_size=2, stride=2, bias=False, indice_key="spconv3"), 128),
                _ConvBN(spconv.SparseConv3d(128, 256, kernel_size=2, stride=2, bias=False, indice_key="spconv4"), 256),
            ]
        )
        self.encoder.enc = nn.ModuleList(
            [
                _stage([_ResBlock(32, 32, "subm1") for _ in range(2)]),
                _stage([_ResBlock(64, 64, "subm2") for _ in range(3)]),
                _stage([_ResBlock(128, 128, "subm3") for _ in range(4)]),
                _stage([_ResBlock(256, 256, "subm4") for _ in range(6)]),
            ]
        )
        self.decoder.up = nn.ModuleList(
            [
                _ConvBN(spconv.SparseInverseConv3d(96, 96, kernel_size=2, bias=False, indice_key="spconv1"), 96),
                _ConvBN(spconv.SparseInverseConv3d(128, 96, kernel_size=2, bias=False, indice_key="spconv2"), 96),
                _ConvBN(spconv.SparseInverseConv3d(256, 128, kernel_size=2, bias=False, indice_key="spconv3"), 128),
                _ConvBN(spconv.SparseInverseConv3d(256, 256, kernel_size=2, bias=False, indice_key="spconv4"), 256),
            ]
        )
        self.decoder.dec = nn.ModuleList(
            [
                _stage([_ResBlock(128, 96, "dec0"), _ResBlock(96, 96, "dec0")]),
                _stage([_ResBlock(128, 96, "dec1"), _ResBlock(96, 96, "dec1")]),
                _stage([_ResBlock(192, 128, "dec2"), _ResBlock(128, 128, "dec2")]),
                _stage([_ResBlock(384, 256, "dec3"), _ResBlock(256, 256, "dec3")]),
            ]
        )
        self.head = spconv.SubMConv3d(96, 20, kernel_size=1, bias=True, indice_key="head")
        self.relu = nn.ReLU()

    def _activated(self, layer: _ConvBN, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = layer.conv(x)
        return x.replace_feature(self.relu(layer.bn(x.features)))

    def forward_features(self, x: spconv.SparseConvTensor) -> torch.Tensor:
        x = self._activated(self.encoder.conv_input, x)
        skips = [x]
        for s in range(4):
            x = self._activated(self.encoder.down[s], x)
            for block in self.encoder.enc[s].children():
                x = block(x)
            skips.append(x)
        skips.pop(-1)
        for s in reversed(range(4)):
            x = self._activated(self.decoder.up[s], x)
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat([x.features, skip.features], dim=1))
            for block in self.decoder.dec[s].children():
                x = block(x)
        return x.features


class SpUNetLidarEncoder:
    """Load the ScanNet SpUNet and pool the pre-head voxel features."""

    OUT_DIM = 96
    GRID_SIZE = 0.02

    def __init__(self, ckpt_path: str, device: str) -> None:
        path = os.path.abspath(ckpt_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SpUNet checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model = _SpUNetV1M1()
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.out_dim = int(self.OUT_DIM)
        self.grid_size = float(self.GRID_SIZE)
        print(
            f"[SpUNetLidarEncoder] {path} out_dim={self.out_dim} "
            f"grid_size={self.grid_size:g}m head=off color=0 normal=0 "
            f"frozen device={self.device}"
        )

    def encode(self, xyz: np.ndarray) -> torch.Tensor:
        """Return a (out_dim,) feature on ``self.device``. Empty input is zeros."""
        pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return torch.zeros(self.out_dim, device=self.device, dtype=torch.float32)
        grid = np.floor(pts / self.grid_size).astype(np.int64)
        _, keep = np.unique(grid, axis=0, return_index=True)
        grid = grid[keep]
        grid -= grid.min(axis=0)
        spatial_shape = (grid.max(axis=0) + 96).astype(np.int32).tolist()
        indices = np.concatenate(
            [np.zeros((grid.shape[0], 1), dtype=np.int32), grid.astype(np.int32)],
            axis=1,
        )
        features = torch.zeros((grid.shape[0], 6), dtype=torch.float32, device=self.device)
        indices_t = torch.from_numpy(np.ascontiguousarray(indices)).to(self.device)
        sparse = spconv.SparseConvTensor(
            features=features,
            indices=indices_t,
            spatial_shape=spatial_shape,
            batch_size=1,
        )
        with torch.no_grad():
            feat = self.model.forward_features(sparse)
            pooled = feat.max(dim=0).values
        if int(pooled.numel()) != self.out_dim:
            raise RuntimeError(
                f"SpUNet feat numel {int(pooled.numel())}, expected {self.out_dim}"
            )
        return pooled
