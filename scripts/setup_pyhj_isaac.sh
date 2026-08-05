#!/usr/bin/env bash
# Clone upstream PyHJ (PytorchReachability), apply Isaac Lab / gymnasium 1.x patch,
# and install editable with --no-deps (do not touch Isaac's torch/gymnasium).
#
# Usage (from anywhere):
#   bash /workspace/dino_wm_jayden/scripts/setup_pyhj_isaac.sh
#
# Optional env vars:
#   PYHJ_ROOT   install location (default: /workspace/PytorchReachability)
#   PYTHON      python binary (default: current `python`)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/patches/pyhj_isaaclab_gymnasium.patch"
UPSTREAM_URL="https://github.com/CMU-IntentLab/PytorchReachability.git"
PYHJ_ROOT="${PYHJ_ROOT:-/workspace/PytorchReachability}"
PYTHON="${PYTHON:-python}"

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "[ERROR] Missing patch: ${PATCH_FILE}"
  exit 1
fi

if [[ ! -d "${PYHJ_ROOT}/.git" ]]; then
  echo "[INFO] git clone ${UPSTREAM_URL} -> ${PYHJ_ROOT}"
  rm -rf "${PYHJ_ROOT}"
  git clone --depth 1 "${UPSTREAM_URL}" "${PYHJ_ROOT}"
else
  echo "[INFO] Reusing existing clone at ${PYHJ_ROOT}"
fi

cd "${PYHJ_ROOT}"

is_fully_patched() {
  # All markers must be present; a partial apply must NOT count as done.
  grep -q "LatentHumanoidEnv" PyHJ/__init__.py 2>/dev/null \
    && grep -q "def __getattr__" PyHJ/data/__init__.py 2>/dev/null \
    && grep -q "from PyHJ.policy.base import BasePolicy" PyHJ/data/collector.py 2>/dev/null \
    && grep -q "from PyHJ.data.batch import Batch" PyHJ/policy/base.py 2>/dev/null \
    && grep -q "load_plugin_envs.*removed" PyHJ/reach_rl_gym_envs/__init__.py 2>/dev/null
}

if is_fully_patched; then
  echo "[INFO] Patch already present; skipping apply."
else
  # Recover from a previous partial apply (e.g. malformed patch left __init__ only).
  if grep -q "LatentHumanoidEnv" PyHJ/__init__.py 2>/dev/null; then
    echo "[WARN] Partial patch detected; resetting tracked PyHJ files to HEAD."
    git checkout -- PyHJ
  fi

  echo "[INFO] Applying ${PATCH_FILE}"
  if ! git apply --check "${PATCH_FILE}"; then
    echo "[ERROR] Patch does not apply cleanly to ${PYHJ_ROOT}."
    echo "        Re-clone or update the patch, then re-run."
    exit 1
  fi
  git apply "${PATCH_FILE}"
fi

echo "[INFO] pip install -e . --no-deps"
"${PYTHON}" -m pip install -e "${PYHJ_ROOT}" --no-deps
"${PYTHON}" -m pip install -q tqdm numba tensorboard packaging

echo "[INFO] Verifying import..."
"${PYTHON}" -c "from PyHJ.policy import avoid_DDPGPolicy_annealing; print('PyHJ OK')"

echo "[INFO] Done. PYHJ_ROOT=${PYHJ_ROOT}"
