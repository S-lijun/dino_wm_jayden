#!/usr/bin/env bash
# Undo accidental `pip install gsplat plyfile` inside isaaclab5_pip.
# Run INSIDE the Isaac Sim container on compute2:
#   bash scripts/demos/revert_gsplat_from_isaac_venv.sh

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -x /workspace/venvs/isaaclab5_pip/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /workspace/venvs/isaaclab5_pip/bin/activate
  else
    echo "[ERROR] Activate isaaclab5_pip first: source /workspace/venvs/isaaclab5_pip/bin/activate"
    exit 1
  fi
fi

echo "[INFO] Reverting gsplat/plyfile install in: ${VIRTUAL_ENV}"

pip uninstall -y gsplat plyfile jaxtyping wadler-lindig ninja 2>/dev/null || true

# Isaac Sim 5.1 / Isaac Lab pins (from isaacsim-kernel)
pip install "numpy==1.26.0" "click==8.1.7" "psutil==5.9.8"

echo "[INFO] Versions after revert:"
python - <<'PY'
import click, numpy, psutil
print("numpy", numpy.__version__)
print("click", click.__version__)
print("psutil", psutil.__version__)
PY

if pip check; then
  echo "[OK] pip check passed — Isaac venv restored."
else
  echo "[WARN] pip check still reports issues; paste output above if Isaac fails to start."
  exit 1
fi
