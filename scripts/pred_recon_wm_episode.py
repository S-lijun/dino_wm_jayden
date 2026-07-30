"""Predict next-frame latent with WM, decode it; save GT vs pred-recon grids.

Each saved image has 2 rows x N frames:
  top    = original next frame (GT)
  bottom = decoder(predicted z_{t+1})

Uses one-step open-loop prediction:
  encode frames [t - num_hist + 1 .. t] + actions -> predict -> take last block
  as z_{t+1}, then decode visual tokens.

Usage:
  python scripts/pred_recon_wm_episode.py \
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
        help="Default: <repo>/wm_recon_vis/episode_<N>_pred_grids",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of sliding prediction windows per forward pass.",
    )
    p.add_argument(
        "--frames_per_grid",
        type=int,
        default=6,
        help="Number of frames per comparison image (top original / bottom pred recon).",
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
    preds: torch.Tensor,
    out_dir: Path,
    frames_per_grid: int,
    frame_offset: int = 0,
):
    """Save sequential 2-row grids: top=original, bottom=predicted recon.

    frame_offset: GT/pred index of the first frame (e.g. num_hist), used in filenames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    T = originals.shape[0]
    n_saved = 0
    for start in range(0, T, frames_per_grid):
        end = min(start + frames_per_grid, T)
        gt = originals[start:end]
        pred = preds[start:end]
        n = gt.shape[0]
        if n < frames_per_grid:
            pad = torch.full(
                (frames_per_grid - n, *gt.shape[1:]),
                -1.0,
                dtype=gt.dtype,
            )
            gt = torch.cat([gt, pad], dim=0)
            pred = torch.cat([pred, pad], dim=0)
        grid = torch.cat([gt, pred], dim=0)
        f0 = frame_offset + start
        f1 = frame_offset + end - 1
        name = f"pred_recon_f{f0:05d}_{f1:05d}.png"
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
def predict_next_visuals(
    wm,
    visuals: torch.Tensor,
    proprios: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """One-step predict+decode for each target frame num_hist .. T-1.

    Returns (T - num_hist, C, H, W) predicted reconstructed images aligned to
    GT frames [num_hist, T).
    """
    num_hist = int(wm.num_hist)
    T = visuals.shape[0]
    if T <= num_hist:
        raise ValueError(f"Need T > num_hist ({num_hist}), got T={T}")

    # Sliding windows ending at t = num_hist-1 .. T-2 predict frames num_hist .. T-1
    ends = list(range(num_hist - 1, T - 1))  # last hist index for each window
    pred_chunks = []

    for i in range(0, len(ends), batch_size):
        batch_ends = ends[i : i + batch_size]
        b = len(batch_ends)
        vis_win = []
        prop_win = []
        act_win = []
        for e in batch_ends:
            s = e - num_hist + 1
            vis_win.append(visuals[s : e + 1])
            prop_win.append(proprios[s : e + 1])
            act_win.append(actions[s : e + 1])
        vis_b = torch.stack(vis_win, dim=0).to(device)  # (b, num_hist, C, H, W)
        prop_b = torch.stack(prop_win, dim=0).to(device)
        act_b = torch.stack(act_win, dim=0).to(device)

        z = wm.encode({"visual": vis_b, "proprio": prop_b}, act_b)
        z_pred = wm.predict(z)
        z_next = z_pred[:, -1:, ...]  # (b, 1, P, D) = predicted next latent
        z_obs, _ = wm.separate_emb(z_next)
        recon_obs, _ = wm.decode_obs(z_obs)
        pred = recon_obs["visual"][:, 0].detach().cpu()  # (b, C, H, W)
        pred_chunks.append(pred)
        done = i + b
        print(f"  predicted [{done}/{len(ends)}] (targets {num_hist}..{num_hist + done - 1})")

    return torch.cat(pred_chunks, dim=0)


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
    if wm.predictor is None:
        raise RuntimeError("Predictor is None — cannot predict next latent.")

    num_hist = int(wm.num_hist)
    print(f"num_hist={num_hist}, num_pred={wm.num_pred}")

    obs_path = session / "obses_15fps" / f"episode_{args.episode}.pth"
    if not obs_path.exists():
        obs_path = session / "obses" / f"episode_{args.episode}.pth"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing episode visuals: {obs_path}")

    raw = torch.load(obs_path, map_location="cpu", weights_only=False)
    T = int(raw.shape[0])
    print(f"Episode {args.episode}: {T} frames, raw shape={tuple(raw.shape)}")

    meta = torch.load(session / "meta.pth", map_location="cpu", weights_only=False)
    proprio_dim = int(meta["proprio_dim"])
    states = torch.load(session / "states.pth", map_location="cpu", weights_only=False)
    actions_full = torch.load(session / "actions.pth", map_location="cpu", weights_only=False)
    seq_lengths = torch.load(session / "seq_lengths.pth", map_location="cpu", weights_only=False)
    step_len = int(seq_lengths[args.episode].item())
    step_idx = visual_to_step_indices(step_len, T)
    steps = torch.tensor(step_idx, dtype=torch.long)

    proprio_end = PROPRIO_START + proprio_dim
    proprios = states[args.episode].index_select(0, steps)[:, PROPRIO_START:proprio_end].float()
    actions = actions_full[args.episode].index_select(0, steps).float()

    visuals = letterbox_normalize(raw, img_h, img_w)  # (T,C,H,W) in [-1,1]
    del raw

    out_dir = args.out_dir or (
        REPO_ROOT / "wm_recon_vis" / f"episode_{args.episode}_pred_grids"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running one-step predict + decode ...")
    preds = predict_next_visuals(
        wm, visuals, proprios, actions, args.batch_size, device
    )
    # Align GT to predicted targets: frames num_hist .. T-1
    gts = visuals[num_hist:]
    assert preds.shape[0] == gts.shape[0], (preds.shape, gts.shape)

    mse = float(F.mse_loss(preds, gts).item())
    n_grids = save_comparison_grids(
        gts,
        preds,
        out_dir,
        frames_per_grid=args.frames_per_grid,
        frame_offset=num_hist,
    )

    print(f"Done. Pred-recon MSE ([-1,1] space): {mse:.6f}")
    print(
        f"Saved {n_grids} comparison grids "
        f"(top=GT next frame, bottom=pred-latent recon) -> {out_dir}"
    )
    print(f"Covered GT frames {num_hist}..{T - 1} ({preds.shape[0]} frames)")


if __name__ == "__main__":
    main()
