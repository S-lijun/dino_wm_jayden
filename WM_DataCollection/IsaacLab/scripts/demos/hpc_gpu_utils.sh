#!/usr/bin/env bash
# GPU selection for rtx_rgb on shared HPC nodes.
#
# Rules:
# - If Slurm already set CUDA_VISIBLE_DEVICES to ONE logical GPU → keep it (do NOT remap).
# - If unset or multiple GPUs listed → pick physical index with least used memory.

pick_free_gpu_index() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n \
    | head -1 \
    | awk -F, '{gsub(/ /,"",$1); print $1}'
}

export_single_gpu() {
  local cvd="${CUDA_VISIBLE_DEVICES:-}"

  # Slurm / srun --gpus=1 typically exposes exactly one *logical* GPU as index 0.
  if [[ -n "${cvd}" && "${cvd}" != *","* ]]; then
    echo "[INFO] Slurm already pinned GPU → keeping CUDA_VISIBLE_DEVICES=${cvd}"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
    return
  fi

  local idx="${GS_GPU:-}"
  if [[ -z "${idx}" ]]; then
    idx="$(pick_free_gpu_index)"
  fi
  export CUDA_VISIBLE_DEVICES="${idx}"
  # Do NOT set NVIDIA_VISIBLE_DEVICES here — breaks Vulkan GPU enumeration in Isaac containers.
  echo "[INFO] rtx_rgb pinned to physical GPU ${idx} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv 2>/dev/null || true
}

preflight_rtx_gpu() {
  python "${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/preflight_rtx_gpu.py"
}

preflight_cuda() {
  python - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("[WARN] torch not importable for preflight")
    sys.exit(0)
if not torch.cuda.is_available():
    print("[ERROR] torch.cuda.is_available()=False — rtx_rgb will not work on this node.")
    print("        Check: nvidia-smi, Slurm GPU allocation, CUDA_VISIBLE_DEVICES")
    sys.exit(1)
print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
PY
}
