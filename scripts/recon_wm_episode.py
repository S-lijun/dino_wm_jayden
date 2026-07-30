"""Encode->decode an offline episode; save original/recon comparison grids.

Each saved image has 2 rows x N frames:
  top    = original
  bottom = reconstructed

Usage:
  python scripts/recon_wm_episode.py \
    --ckpt_dir wm_ckpt_18-27-17 \
    --session Trajectories/session2 \
    --episode 0 \
    --frames_per_grid 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.humanoid_dset import PROPRIO_START, visual_to_step_indices
from wm_load import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=Path, default=REPO_ROOT / "wm_ckpt_18-27-17")
    p.add_argument("--session", type=Path, default=REPO_ROOT / "Trajectories" / "session2")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Default: <repo>/wm_recon_vis/episode_<N>",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--frames_per_grid",
        type=int,
        default=6,
        help="Number of frames per comparison image (top original / bottom recon).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--ckpt_name",
        type=str,
        default="model_best.pth",
        help="Checkpoint filename under checkpoints/",
    )
    return p.parse_args()


def letterbox_normalize(images_thwc_uint8: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """(T,H,W,C) uint8 [0,255] -> (T,C,H',W') float in [-1,1]."""
    x = images_thwc_uint8.float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()  # T C H W
    if x.shape[-2] != img_h or x.shape[-1] != img_w:
        x = F.interpolate(x, size=(img_h, img_w), mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5


def save_comparison_grids(
    originals: torch.Tensor,
    recons: torch.Tensor,
    out_dir: Path,
    frames_per_grid: int,
):
    """Save sequential 2-row grids: top=original, bottom=reconstructed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = originals.shape[0]
    n_saved = 0
    for start in range(0, T, frames_per_grid):
        end = min(start + frames_per_grid, T)
        gt = originals[start:end]
        pred = recons[start:end]
        # pad last chunk so nrow stays consistent visually
        n = gt.shape[0]
        if n < frames_per_grid:
            pad = torch.full(
                (frames_per_grid - n, *gt.shape[1:]),
                -1.0,
                dtype=gt.dtype,
            )
            gt = torch.cat([gt, pad], dim=0)
            pred = torch.cat([pred, pad], dim=0)
        # order for make_grid(nrow=k): first k = top row, next k = bottom row
        grid = torch.cat([gt, pred], dim=0)
        name = f"recon_f{start:05d}_{end - 1:05d}.png"
        save_image(
            grid,
            out_dir / name,
            nrow=frames_per_grid,
            normalize=True,
            value_range=(-1, 1),
        )
        n_saved += 1
    return n_saved


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_dir = args.ckpt_dir.resolve()
    session = args.session.resolve()
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / args.ckpt_name
    if not snapshot.exists():
        alt = ckpt_dir / "checkpoints" / "model_latest.pth"
        if alt.exists():
            print(f"[WARN] {snapshot.name} missing, falling back to {alt.name}")
            snapshot = alt
        else:
            raise FileNotFoundError(f"No checkpoint at {snapshot}")

    train_cfg = OmegaConf.load(str(hydra_cfg))
    img_h, img_w = int(train_cfg.img_h), int(train_cfg.img_w)

    print(f"Loading WM from {snapshot} on {device} ...")
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=str(device))
    wm.eval()
    if wm.decoder is None:
        raise RuntimeError("Decoder is None — cannot reconstruct.")

    obs_path = session / "obses_15fps" / f"episode_{args.episode}.pth"
    if not obs_path.exists():
        obs_path = session / "obses" / f"episode_{args.episode}.pth"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing episode visuals: {obs_path}")

    raw = torch.load(obs_path, map_location="cpu", weights_only=False)
    # (T, H, W, C) uint8
    T = int(raw.shape[0])
    print(f"Episode {args.episode}: {T} frames, raw shape={tuple(raw.shape)}")

    meta = torch.load(session / "meta.pth", map_location="cpu", weights_only=False)
    proprio_dim = int(meta["proprio_dim"])
    states = torch.load(session / "states.pth", map_location="cpu", weights_only=False)
    seq_lengths = torch.load(session / "seq_lengths.pth", map_location="cpu", weights_only=False)
    step_len = int(seq_lengths[args.episode].item())
    step_idx = visual_to_step_indices(step_len, T)
    proprio_end = PROPRIO_START + proprio_dim
    proprios = states[args.episode, step_idx, PROPRIO_START:proprio_end].float()

    visuals = letterbox_normalize(raw, img_h, img_w)  # (T,C,H,W) in [-1,1]
    del raw

    out_dir = args.out_dir or (REPO_ROOT / "wm_recon_vis" / f"episode_{args.episode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    recon_chunks = []
    mse_sum = 0.0
    n_pix = 0
    for start in range(0, T, args.batch_size):
        end = min(start + args.batch_size, T)
        vis = visuals[start:end].unsqueeze(0).to(device)  # (1,t,C,H,W)
        prop = proprios[start:end].unsqueeze(0).to(device)  # (1,t,D)
        z_obs = wm.encode_obs({"visual": vis, "proprio": prop})
        recon_obs, _ = wm.decode_obs(z_obs)
        recon = recon_obs["visual"][0].detach().cpu()  # (t,C,H,W)
        gt = visuals[start:end]

        mse_sum += float(F.mse_loss(recon, gt, reduction="sum").item())
        n_pix += recon.numel()
        recon_chunks.append(recon)
        print(f"  [{end}/{T}] encoded")

    recons = torch.cat(recon_chunks, dim=0)
    n_grids = save_comparison_grids(
        visuals, recons, out_dir, frames_per_grid=args.frames_per_grid
    )

    mse = mse_sum / max(n_pix, 1)
    print(f"Done. MSE ([-1,1] space): {mse:.6f}")
    print(f"Saved {n_grids} comparison grids (top=original, bottom=recon) -> {out_dir}")


if __name__ == "__main__":
    main()
