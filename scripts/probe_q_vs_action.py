"""Fixed-z Q probe: Q(z, a_nom) vs Q(z, a_sf) vs Q(z, a_rand).

Loads a SAC path policy.pth (no Isaac). Confirms late-fusion critic actually
depends on the 3-D action.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import ActorProb
from env.isaac.late_fusion_critic import LateFusionCritic


def _min_q(c1, c2, z, a):
    return torch.min(c1(z, a), c2(z, a))


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt().item())


def resolve_policy_path(p: str | Path) -> Path:
    path = Path(str(p).strip().strip('"').strip("'")).expanduser()
    if path.is_file():
        return path
    cand = path / "policy.pth"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"policy not found: {p}")


def load_nets(ckpt: Path, device: str):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    obs_dim = int(sd["critic1.z_mlp.model.0.weight"].shape[1])
    act_dim = int(sd["critic1.a_mlp.model.0.weight"].shape[1])
    hidden = (512, 512, 512)

    def make_critic():
        return LateFusionCritic(
            obs_dim, act_dim, hidden_sizes=hidden, activation=torch.nn.ReLU, device=device
        ).to(device)

    c1, c2 = make_critic(), make_critic()
    c1.load_state_dict({k[len("critic1.") :]: v for k, v in sd.items() if k.startswith("critic1.") and not k.startswith("critic1_old.")})
    c2.load_state_dict({k[len("critic2.") :]: v for k, v in sd.items() if k.startswith("critic2.") and not k.startswith("critic2_old.")})
    c1.eval()
    c2.eval()

    actor_net = Net(obs_dim, hidden_sizes=hidden, activation=torch.nn.ReLU, device=device)
    actor = ActorProb(actor_net, act_dim, max_action=1.0, unbounded=True, device=device).to(device)
    actor.load_state_dict({k[len("actor1.") :]: v for k, v in sd.items() if k.startswith("actor1.") and not k.startswith("actor1_old.")})
    actor.eval()
    return c1, c2, actor, obs_dim, act_dim, sd


@torch.no_grad()
def a_sf_of(actor, z: torch.Tensor) -> torch.Tensor:
    (mu, _), _ = actor(z)
    return torch.tanh(mu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_z", type=int, default=8)
    parser.add_argument("--n_rand", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = resolve_policy_path(args.policy_path)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[INFO] ckpt={ckpt}")
    print(f"[INFO] device={device}")
    c1, c2, actor, obs_dim, act_dim, sd = load_nets(ckpt, device)
    print(f"[INFO] obs_dim={obs_dim} act_dim={act_dim}")
    print("[INFO] critic is LateFusion: z_mlp + a_mlp (3->128->512) + q_head(1024->512->1)")

    w_z = sd["critic1.q_head.model.0.weight"]  # (512, 1024)
    rms_z_half = _rms(w_z[:, :512])
    rms_a_half = _rms(w_z[:, 512:])
    print(
        f"[INFO] q_head first Linear RMS: z-half={rms_z_half:.4f} a-half={rms_a_half:.4f} "
        f"ratio a/z={rms_a_half / max(rms_z_half, 1e-12):.3f}"
    )
    print(
        f"[INFO] a_mlp[0] RMS={_rms(sd['critic1.a_mlp.model.0.weight']):.4f} "
        f"z_mlp[0] RMS={_rms(sd['critic1.z_mlp.model.0.weight']):.6f}"
    )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # z=0 plus Gaussian latents at two scales (off-manifold, but z is frozen).
    z_list = [("z=0", torch.zeros(1, obs_dim, device=device))]
    for scale in (0.05, 0.5):
        for i in range(args.n_z // 2):
            z = torch.as_tensor(
                rng.normal(0.0, scale, size=(1, obs_dim)),
                dtype=torch.float32,
                device=device,
            )
            z_list.append((f"z~N(0,{scale})#{i}", z))

    a_nom = torch.tensor([[1.0, 0.0, 0.0]], device=device)  # policy-space forward
    lines = []
    all_ranges = []
    all_dqa = []

    print("\n=== fixed z: Q(a_nom) vs Q(a_sf) vs Q(a_rand mean +/- std) ===")
    print(
        f"{'z':<18} {'Qnom':>8} {'Qsf':>8} {'Qrand':>8} {'std_rand':>8} "
        f"{'range64':>8} {'dQ_sf_nom':>10} {'dQda':>10} {'a_sf'}"
    )

    for name, z in z_list:
        with torch.no_grad():
            a_sf = a_sf_of(actor, z)
            a_rand = torch.as_tensor(
                rng.uniform(-1.0, 1.0, size=(args.n_rand, act_dim)),
                dtype=torch.float32,
                device=device,
            )
            z_b = z.expand(args.n_rand, -1)
            q_nom = float(_min_q(c1, c2, z, a_nom).item())
            q_sf = float(_min_q(c1, c2, z, a_sf).item())
            q_rand = _min_q(c1, c2, z_b, a_rand).reshape(-1).detach().cpu().numpy()
        q_range = float(q_rand.max() - q_rand.min())
        all_ranges.append(q_range)

        z_g = z.detach().clone().requires_grad_(True)
        a_g = a_sf.detach().clone().requires_grad_(True)
        q = _min_q(c1, c2, z_g, a_g)
        grads = torch.autograd.grad(q.sum(), a_g, retain_graph=False)[0]
        dqa = float(grads.norm().item())
        all_dqa.append(dqa)

        row = (
            f"{name:<18} {q_nom:8.3f} {q_sf:8.3f} {float(q_rand.mean()):8.3f} "
            f"{float(q_rand.std()):8.3f} {q_range:8.3f} {abs(q_sf - q_nom):10.3f} "
            f"{dqa:10.3f} {np.array2string(a_sf.detach().cpu().numpy().reshape(-1), precision=3)}"
        )
        print(row)
        lines.append(row)

    # 1-D sweep on z=0: a = [t, 0, 0]
    ts = np.linspace(-1.0, 1.0, 21)
    z0 = torch.zeros(1, obs_dim, device=device)
    qs = []
    with torch.no_grad():
        for t in ts:
            a = torch.tensor([[float(t), 0.0, 0.0]], device=device)
            qs.append(float(_min_q(c1, c2, z0, a).item()))
    qs = np.asarray(qs)
    print("\n=== z=0 sweep a=[t,0,0] ===")
    print("t   ", " ".join(f"{t:6.2f}" for t in ts[::4]))
    print("Q   ", " ".join(f"{q:6.3f}" for q in qs[::4]))
    print(f"sweep range={qs.max() - qs.min():.4f}  std={qs.std():.4f}")

    mean_range = float(np.mean(all_ranges))
    mean_dqa = float(np.mean(all_dqa))
    print("\n=== verdict ===")
    print(f"mean Q-range over {args.n_rand} random a (z frozen): {mean_range:.4f}")
    print(f"mean ||dQ/da|| at a_sf: {mean_dqa:.4f}")
    if mean_range < 1e-3 and mean_dqa < 1e-3:
        print("FAIL: Q is still ~Q(z); changing a does not move Q.")
    elif mean_range < 0.05:
        print("WEAK: Q moves with a, but only a little.")
    else:
        print("OK: Q depends on a (late fusion is not ignoring the action).")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
