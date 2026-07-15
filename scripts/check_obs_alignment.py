#!/usr/bin/env python3
import torch
from pathlib import Path

p = Path("/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino/humanoid_g1/session_dir")
sl = torch.load(p / "seq_lengths.pth", weights_only=False)
ratios = []
for i in range(min(20, len(sl))):
    obs = torch.load(p / f"obses/episode_{i}.pth", weights_only=False)
    ratios.append(sl[i].item() / obs.shape[0])
    print(f"ep{i}: seq_len={int(sl[i])}, obs_T={obs.shape[0]}, ratio={ratios[-1]:.3f}")

print(f"mean ratio (steps per visual frame): {sum(ratios)/len(ratios):.3f}")
