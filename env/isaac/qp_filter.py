"""QP-style safety filter: closest action to nominal with Q >= threshold.

Mirrors CarGoal ``mode == "QP"``, but humanoid actions are (vx, vy, yaw) in
policy space ``[-1, 1]``.

Default search is **stratified continuous**: partition each axis into ``n_grid``
bins and draw one uniform sample inside every cell (covers the continuous
action space better than a fixed linspace lattice). Use ``sample_mode="fixed"``
for the legacy discrete corner/center lattice.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch


def _candidate_actions(
    *,
    n_grid: int,
    freeze_yaw: bool,
    yaw: float,
    device: torch.device | str,
    sample_mode: Literal["stratified", "fixed"],
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return (N, 3) candidate actions in policy space."""
    n = int(n_grid)
    if n < 1:
        raise ValueError(f"n_grid must be >= 1, got {n_grid}")

    if sample_mode == "fixed":
        g = torch.linspace(-1.0, 1.0, n, device=device)
        if freeze_yaw:
            vx, vy = torch.meshgrid(g, g, indexing="ij")
            return torch.stack(
                [
                    vx.reshape(-1),
                    vy.reshape(-1),
                    torch.full((vx.numel(),), float(yaw), device=device),
                ],
                dim=-1,
            )
        vx, vy, yaw_g = torch.meshgrid(g, g, g, indexing="ij")
        return torch.stack(
            [vx.reshape(-1), vy.reshape(-1), yaw_g.reshape(-1)],
            dim=-1,
        )

    if sample_mode != "stratified":
        raise ValueError(
            f"sample_mode must be 'stratified' or 'fixed', got {sample_mode!r}"
        )

    # Partition [-1, 1] into n equal bins; sample U(lo, hi) once per cell.
    edges = torch.linspace(-1.0, 1.0, n + 1, device=device)
    idx = torch.arange(n, device=device)
    if freeze_yaw:
        ix, iy = torch.meshgrid(idx, idx, indexing="ij")
        ix = ix.reshape(-1)
        iy = iy.reshape(-1)
        u = torch.rand((ix.numel(), 2), device=device, generator=generator)
        vx = edges[ix] + u[:, 0] * (edges[ix + 1] - edges[ix])
        vy = edges[iy] + u[:, 1] * (edges[iy + 1] - edges[iy])
        yaw_t = torch.full((ix.numel(),), float(yaw), device=device)
        return torch.stack([vx, vy, yaw_t], dim=-1)

    ix, iy, iz = torch.meshgrid(idx, idx, idx, indexing="ij")
    ix = ix.reshape(-1)
    iy = iy.reshape(-1)
    iz = iz.reshape(-1)
    u = torch.rand((ix.numel(), 3), device=device, generator=generator)
    vx = edges[ix] + u[:, 0] * (edges[ix + 1] - edges[ix])
    vy = edges[iy] + u[:, 1] * (edges[iy + 1] - edges[iy])
    yaw_t = edges[iz] + u[:, 2] * (edges[iz + 1] - edges[iz])
    return torch.stack([vx, vy, yaw_t], dim=-1)


def _vy_yaw_same_sign_mask(actions: torch.Tensor) -> torch.Tensor:
    """True where sign(vy) and sign(yaw) agree (either zero counts as compatible)."""
    vy = actions[:, 1]
    yaw = actions[:, 2]
    return vy * yaw >= 0.0


@torch.no_grad()
def qp_filter_action_policy(
    critic: torch.nn.Module,
    obs: Any,
    a_nom_policy: np.ndarray | torch.Tensor,
    *,
    safety_threshold: float = 0.0,
    n_grid: int = 21,
    freeze_yaw: bool = False,
    yaw: float = 0.0,
    device: str | torch.device | None = None,
    chunk_size: int = 256,
    sample_mode: Literal["stratified", "fixed"] = "stratified",
    generator: torch.Generator | None = None,
    require_vy_yaw_same_sign: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return policy-space action (3,) and debug info.

    If ``Q(z, a_nom) >= threshold``, return ``a_nom``.
    Else search candidates in ``[-1, 1]``:
      - ``sample_mode="stratified"`` (default): one continuous uniform draw per
        axis-aligned cell (``n_grid`` bins/axis → ``n_grid³`` or ``n_grid²``).
      - ``sample_mode="fixed"``: legacy linspace lattice.
      - ``freeze_yaw=True``: yaw fixed to ``yaw``.
      - ``require_vy_yaw_same_sign=True`` (default): drop candidates where
        ``sign(vy) != sign(yaw)`` (zeros allowed with either sign).
    Among candidates with Q >= threshold pick closest to ``a_nom`` (L2).
    If none are safe, fall back to argmax_Q on the candidate set.
    """
    if device is None:
        try:
            device = next(critic.parameters()).device
        except StopIteration:
            device = "cpu"

    if not torch.is_tensor(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    else:
        obs_t = obs.to(device=device, dtype=torch.float32)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    if obs_t.shape[0] != 1:
        raise ValueError(f"qp_filter expects a single state, got batch={obs_t.shape[0]}")

    a_nom = np.asarray(a_nom_policy, dtype=np.float32).reshape(-1)
    if a_nom.size < 2:
        raise ValueError(f"a_nom must have at least vx,vy, got shape {a_nom.shape}")
    a_nom_3 = np.zeros(3, dtype=np.float32)
    a_nom_3[0] = float(np.clip(a_nom[0], -1.0, 1.0))
    a_nom_3[1] = float(np.clip(a_nom[1], -1.0, 1.0))
    if freeze_yaw:
        a_nom_3[2] = float(yaw)
    else:
        a_nom_3[2] = float(np.clip(a_nom[2], -1.0, 1.0)) if a_nom.size >= 3 else 0.0

    a_nom_t = torch.as_tensor(a_nom_3, dtype=torch.float32, device=device).unsqueeze(0)
    q_nom = float(critic(obs_t, a_nom_t).reshape(-1)[0].item())
    info: dict[str, Any] = {
        "q_nom": q_nom,
        "intervened": False,
        "n_safe": 0,
        "fallback_maxq": False,
        "freeze_yaw": bool(freeze_yaw),
        "sample_mode": sample_mode,
    }
    if q_nom >= safety_threshold:
        return a_nom_3, info

    info["intervened"] = True
    info["require_vy_yaw_same_sign"] = bool(require_vy_yaw_same_sign)
    grid = _candidate_actions(
        n_grid=n_grid,
        freeze_yaw=freeze_yaw,
        yaw=yaw,
        device=device,
        sample_mode=sample_mode,
        generator=generator,
    )
    n_raw = int(grid.shape[0])
    info["n_raw_candidates"] = n_raw
    if require_vy_yaw_same_sign:
        keep = _vy_yaw_same_sign_mask(grid)
        grid = grid[keep]
    n_cand = int(grid.shape[0])
    info["n_candidates"] = n_cand
    if n_cand == 0:
        # Should be rare; fall back to a_nom rather than crash.
        info["fallback_maxq"] = True
        info["n_safe"] = 0
        info["q_chosen"] = q_nom
        info["q_min"] = q_nom
        info["q_max"] = q_nom
        info["q_mean"] = q_nom
        info["q_std"] = 0.0
        return a_nom_3, info

    q_all = torch.empty(n_cand, device=device, dtype=torch.float32)
    obs_rep_base = obs_t.expand(chunk_size, -1)
    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        n = end - start
        q_all[start:end] = critic(obs_rep_base[:n], grid[start:end]).reshape(-1)

    safe_mask = q_all >= safety_threshold
    n_safe = int(safe_mask.sum().item())
    info["n_safe"] = n_safe
    info["q_min"] = float(q_all.min().item())
    info["q_max"] = float(q_all.max().item())
    info["q_mean"] = float(q_all.mean().item())
    info["q_std"] = float(q_all.std(unbiased=False).item())
    a_nom_t3 = a_nom_t.reshape(1, 3)
    if n_safe > 0:
        safe_acts = grid[safe_mask]
        safe_q = q_all[safe_mask]
        dist = torch.norm(safe_acts - a_nom_t3, dim=-1)
        best_i = int(dist.argmin().item())
        best = safe_acts[best_i]
        info["q_chosen"] = float(safe_q[best_i].item())
    else:
        info["fallback_maxq"] = True
        best_i = int(q_all.argmax().item())
        best = grid[best_i]
        info["q_chosen"] = float(q_all[best_i].item())  # == q_max

    out = best.detach().cpu().numpy().astype(np.float32).reshape(3)
    if freeze_yaw:
        out[2] = float(yaw)
    return out, info
