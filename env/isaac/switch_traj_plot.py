"""Top-down switch-episode trajectory PNG (path-pipeline formal train)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

# SF when HJ/Q(a_nom)<0; otherwise waypoint/nominal.
_COLOR_SF = "#2ca02c"
_COLOR_NOM = "#d62728"


def _align_use_sf(n_pts: int, use_sf: np.ndarray | Sequence | None) -> np.ndarray:
    """Flag per point: True means the action that arrived here used SF."""
    flags = np.zeros(int(n_pts), dtype=bool)
    if use_sf is None or n_pts <= 0:
        return flags
    raw = np.asarray(use_sf, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        return flags
    if raw.size == n_pts - 1:
        flags[1:] = raw.astype(bool)
        return flags
    m = min(int(n_pts), int(raw.size))
    flags[:m] = raw[:m].astype(bool)
    return flags


def _plot_colored_traj(ax, traj: np.ndarray, use_sf: np.ndarray | None) -> None:
    """Green segments while SF (HJ<0), red otherwise."""
    if traj.shape[0] == 1:
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            c=_COLOR_NOM,
            s=18,
            zorder=4,
            label="nominal (HJ≥0)",
        )
        return
    if traj.shape[0] < 2:
        return
    flags = _align_use_sf(traj.shape[0], use_sf)
    labeled_sf = False
    labeled_nom = False
    i = 1
    n = traj.shape[0]
    while i < n:
        is_sf = bool(flags[i])
        j = i + 1
        while j < n and bool(flags[j]) == is_sf:
            j += 1
        color = _COLOR_SF if is_sf else _COLOR_NOM
        if is_sf and not labeled_sf:
            label = "SF (HJ<0)"
            labeled_sf = True
        elif (not is_sf) and not labeled_nom:
            label = "nominal (HJ≥0)"
            labeled_nom = True
        else:
            label = None
        ax.plot(
            traj[i - 1 : j, 0],
            traj[i - 1 : j, 1],
            "-",
            color=color,
            linewidth=1.8,
            zorder=4,
            label=label,
        )
        i = j


def save_switch_traj_png(
    out_path: str | Path,
    traj_xy: np.ndarray,
    waypoints: np.ndarray,
    obstacle_xys: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    failure_radius: float = 1.5,
    arena: tuple[float, float, float, float] | None = None,
    end_reason: str | None = None,
    episode_id: int | None = None,
    use_sf: np.ndarray | Sequence | None = None,
) -> Path:
    """2D top-down: switch path, start/vias/goal, blue obstacles + dashed r=failure_radius."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traj = np.asarray(traj_xy, dtype=np.float64).reshape(-1, 2)
    if waypoints is None or np.asarray(waypoints).size == 0:
        if traj.shape[0] >= 1:
            wps = np.stack([traj[0], traj[-1]], axis=0)
        else:
            wps = np.zeros((2, 2), dtype=np.float64)
    else:
        wps = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    start_xy = wps[0]
    goal_xy = wps[-1]
    vias = wps[1:-1] if wps.shape[0] > 2 else np.zeros((0, 2))
    obs_xy = []
    if obstacle_xys is not None:
        raw = np.asarray(obstacle_xys, dtype=np.float64)
        if raw.size:
            obs_xy = raw.reshape(-1, 2)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    if arena is not None:
        x0, x1, y0, y1 = (float(v) for v in arena)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="0.55",
                linewidth=1.0,
                linestyle="-",
                zorder=1,
                label="arena",
            )
        )
        ax.set_xlim(x0 - 0.4, x1 + 0.4)
        ax.set_ylim(y0 - 0.4, y1 + 0.4)

    _plot_colored_traj(ax, traj, use_sf)

    ax.scatter(
        float(start_xy[0]),
        float(start_xy[1]),
        facecolors="limegreen",
        edgecolors="black",
        linewidths=0.8,
        s=55,
        zorder=6,
        label="start",
    )
    if vias.size:
        ax.scatter(
            vias[:, 0],
            vias[:, 1],
            c="cyan",
            marker="x",
            s=60,
            zorder=6,
            label="waypoint",
        )
    ax.scatter(
        float(goal_xy[0]),
        float(goal_xy[1]),
        c="red",
        s=90,
        marker="*",
        zorder=7,
        label="goal",
    )

    r = float(failure_radius)
    for i, xy in enumerate(obs_xy):
        ax.scatter(
            float(xy[0]),
            float(xy[1]),
            c="blue",
            s=70,
            zorder=6,
            label="obstacle" if i == 0 else None,
        )
        ax.add_patch(
            Circle(
                (float(xy[0]), float(xy[1])),
                r,
                fill=False,
                edgecolor="blue",
                linestyle="--",
                linewidth=1.3,
                zorder=5,
                label=f"soft margin / failure set r={r:g}" if i == 0 else None,
            )
        )

    title_bits = []
    if episode_id is not None:
        title_bits.append(f"ep {int(episode_id)}")
    title_bits.append("switch")
    if end_reason:
        title_bits.append(str(end_reason))
    n_obs = len(obs_xy)
    title_bits.append("no obstacle" if n_obs == 0 else f"{n_obs} obstacle(s)")
    flags = _align_use_sf(traj.shape[0], use_sf)
    n_seg = max(0, int(traj.shape[0]) - 1)
    n_sf_seg = int(flags[1:].sum()) if n_seg > 0 else 0
    if n_seg > 0:
        title_bits.append(f"SF {n_sf_seg}/{n_seg}")
    ax.set_title(" | ".join(title_bits), fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
