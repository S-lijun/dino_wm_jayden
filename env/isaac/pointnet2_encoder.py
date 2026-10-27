"""Frozen PointNet++ encoder: sensor-frame xyz -> one 1024-D vector.

The ModelNet classification head (1024 -> 512 -> 256 -> 40) is not used.
Coordinates stay in meters. The first FPS seed is point 0 so the same
cloud always maps to the same feature.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PN2_MODELS = os.path.join(
    _REPO_ROOT, "third_party", "Pointnet_Pointnet2_pytorch", "models"
)
if _PN2_MODELS not in sys.path:
    sys.path.insert(0, _PN2_MODELS)

import pointnet2_utils  # noqa: E402
from pointnet2_cls_ssg import get_model  # noqa: E402


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Same FPS as PointNet++, starting at index 0 instead of a random point."""
    device = xyz.device
    batch, n_pts, _ = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch, n_pts), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.zeros(batch, dtype=torch.long, device=device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        closer = dist < distance
        distance[closer] = dist[closer]
        farthest = torch.max(distance, -1)[1]
    return centroids


pointnet2_utils.farthest_point_sample = _farthest_point_sample


def _query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query that still returns ``nsample`` neighbors when the cloud is smaller.

    The vendored query slices to ``nsample`` and then repeats the closest point
    into empty slots. A lidar scan can have fewer points than ``nsample``
    (sa1 uses 32), so the slice is shorter than the repeated row and the
    mask assign crashes. Pad with the closest index first.
    """
    device = xyz.device
    batch, n_pts, _ = xyz.shape
    _, n_query, _ = new_xyz.shape
    group_idx = torch.arange(n_pts, dtype=torch.long, device=device).view(1, 1, n_pts).repeat(
        batch, n_query, 1
    )
    sqrdists = pointnet2_utils.square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = n_pts
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if group_idx.shape[-1] < nsample:
        pad = group_idx[:, :, :1].expand(batch, n_query, nsample - group_idx.shape[-1])
        group_idx = torch.cat([group_idx, pad], dim=-1)
    group_first = group_idx[:, :, :1].expand(batch, n_query, nsample)
    mask = group_idx == n_pts
    group_idx = group_idx.clone()
    group_idx[mask] = group_first[mask]
    return group_idx


pointnet2_utils.query_ball_point = _query_ball_point


class PointNet2LidarEncoder:
    """Load ``pointnet2_cls_ssg.pth`` and return the global set-abstraction feature."""

    OUT_DIM = 1024

    def __init__(self, ckpt_path: str, device: str) -> None:
        path = os.path.abspath(ckpt_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PointNet++ checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        self.model = get_model(num_class=40, normal_channel=False)
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.out_dim = int(self.OUT_DIM)
        print(
            f"[PointNet2LidarEncoder] {path} out_dim={self.out_dim} "
            f"head=off normals=off frozen device={self.device}"
        )

    def encode(self, xyz: np.ndarray) -> torch.Tensor:
        """Return a (out_dim,) feature on ``self.device``. Empty input is zeros."""
        pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] == 0:
            return torch.zeros(self.out_dim, device=self.device, dtype=torch.float32)
        cloud = torch.from_numpy(np.ascontiguousarray(pts)).to(self.device)
        cloud = cloud.transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            l1_xyz, l1_points = self.model.sa1(cloud, None)
            l2_xyz, l2_points = self.model.sa2(l1_xyz, l1_points)
            _l3_xyz, l3_points = self.model.sa3(l2_xyz, l2_points)
            feat = l3_points.reshape(-1)
        if int(feat.numel()) != self.out_dim:
            raise RuntimeError(
                f"PointNet++ feat numel {int(feat.numel())}, expected {self.out_dim}"
            )
        return feat
