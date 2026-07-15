#!/usr/bin/env bash
# Quick node check before rtx_rgb. Safe to run on compute2 login or gpu node.
set -euo pipefail
echo "=== nvidia-smi ==="
nvidia-smi || true
echo ""
echo "=== env ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo ""
echo "=== torch ==="
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("device count:", torch.cuda.device_count())
PY
echo ""
echo "=== rtx_rgb note ==="
echo "If cuda available but rtx_rgb still crashes → H100 headless Vulkan RTX limit."
echo "Use depth_rgb on compute2; GS RGB on local RTX workstation."
