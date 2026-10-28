#!/usr/bin/env bash
# True GS lab RGB on H100 — bypass Isaac RTX (Vulkan ray_tracing).
#
# Step 1 — collect sim + camera poses (cluster-safe, no RTX):
#   bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless --visual_mode depth_rgb --no_collect false
#
# Step 2 — rasterize 3DGS PLY with gsplat (this script):
#   bash scripts/demos/run_gs_rgb_offline.sh --dataset_dir data/<run_id>
#
# Requires a SEPARATE venv (do NOT pip install gsplat into isaaclab5_pip — breaks numpy<2):
#   python3 -m venv /workspace/venvs/gsplat_render
#   source /workspace/venvs/gsplat_render/bin/activate
#   pip install gsplat plyfile imageio numpy
# PLY: scene_new/3dgs_lab.ply (standard 3DGS export of basement lab)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ISAACLAB_ROOT}"

DATASET_DIR=""
GS_PLY=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --gs_ply) GS_PLY="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: bash scripts/demos/run_gs_rgb_offline.sh --dataset_dir data/<timestamp>"
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]] || [[ "${VIRTUAL_ENV}" == *isaaclab5_pip* ]]; then
  echo "[ERROR] Do not run gsplat in isaaclab5_pip (breaks Isaac numpy<2)."
  echo "  python3 -m venv /workspace/venvs/gsplat_render"
  echo "  source /workspace/venvs/gsplat_render/bin/activate && pip install gsplat plyfile imageio"
  exit 1
fi

python - <<'PY' || exit 1
import sys
try:
    import gsplat  # noqa: F401
    import plyfile  # noqa: F401
except ImportError:
    print("[ERROR] Missing deps in THIS venv. pip install gsplat plyfile imageio")
    sys.exit(1)
print("[OK] gsplat + plyfile available")
PY

CMD=(python scripts/demos/render_gs_rgb_offline.py --dataset_dir "${DATASET_DIR}")
[[ -n "${GS_PLY}" ]] && CMD+=(--gs_ply "${GS_PLY}")
CMD+=("${EXTRA[@]}")
exec "${CMD[@]}"
