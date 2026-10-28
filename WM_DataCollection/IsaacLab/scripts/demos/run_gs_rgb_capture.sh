#!/usr/bin/env bash
# True RGB + 3DGS lab (rtx_rgb). depth_rgb stays the compute2 default.
#
#   bash scripts/demos/run_gs_rgb_capture.sh --smoke   # test one PNG
#   bash scripts/demos/run_gs_rgb_capture.sh           # full collection
#
# If Slurm already allocated one GPU, do NOT set GS_GPU — the script keeps Slurm's pin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ISAACLAB_ROOT}"

source "${SCRIPT_DIR}/hpc_gpu_utils.sh"

SMOKE=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --smoke) SMOKE=1 ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done

export_single_gpu
preflight_cuda || {
  echo "[FATAL] CUDA preflight failed. rtx_rgb cannot run. Use depth_rgb on this node:"
  echo "  bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless"
  exit 1
}

preflight_rtx_gpu || {
  echo ""
  echo "[HINT] On H100 use offline GS RGB instead:"
  echo "  bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless --visual_mode depth_rgb"
  echo "  bash scripts/demos/run_gs_rgb_offline.sh --dataset_dir data/<run_id>"
  exit 1
}

export GIT_PYTHON_REFRESH=quiet
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

if [[ "${SMOKE}" -eq 1 ]]; then
  echo "[INFO] Smoke test rtx_rgb → /tmp/gs_rtx_smoke.png"
  exec bash isaaclab.sh -p scripts/demos/test_gs_rtx_smoke_impl.py \
    --headless --enable_cameras --rendering_mode performance \
    --out /tmp/gs_rtx_smoke.png "${EXTRA_ARGS[@]}"
fi

echo "[INFO] Full DataCollection_test.py --visual_mode rtx_rgb"
exec bash isaaclab.sh -p scripts/demos/DataCollection_test.py \
  --headless --visual_mode rtx_rgb --rendering_mode performance \
  "${EXTRA_ARGS[@]}"
