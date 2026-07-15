#!/usr/bin/env bash
# Run INSIDE the C1 custom Isaac Sim container to create a C1-specific venv.
#
# Usage (inside container):
#   bash /workspace/dino_wm_jayden/IsaacLab/docker/isaac-sim-c1/setup_venv_c1.sh
#
# If a previous run failed halfway:
#   rm -rf /workspace/venvs/isaaclab5_pip_c1
#   bash .../setup_venv_c1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/.env.c1"

VENV_PATH="/workspace/venvs/${VENV_NAME}"
ISAACLAB_PATH="/workspace/dino_wm_jayden/IsaacLab"
# Use container /tmp for pip cache (Storage1 writes can fail during bootstrap)
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/pip-cache-${USER}"
mkdir -p "${PIP_CACHE_DIR}"

echo "[INFO] Checking /isaac-sim access..."
ls -la /isaac-sim/python.sh
/isaac-sim/python.sh -c "import sys; print('kit python', sys.version)"

if [[ -d "${VENV_PATH}" ]]; then
  echo "[WARN] Removing incomplete/old venv at ${VENV_PATH}"
  rm -rf "${VENV_PATH}"
fi

mkdir -p /workspace/venvs
echo "[INFO] Creating venv (--without-pip avoids ensurepip failures on Storage1)..."
/isaac-sim/python.sh -m venv --without-pip "${VENV_PATH}"

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
export PYTHONNOUSERSITE=1

echo "[INFO] Bootstrapping pip via get-pip.py (download to /tmp; never touch /isaac-sim)..."
GET_PIP="${TMPDIR:-/tmp}/get-pip-${USER}.py"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${GET_PIP}"
python3 "${GET_PIP}"
# setuptools 82+ removed pkg_resources; flatdict/isaaclab setup.py still need it
python3 -m pip install --upgrade pip
python3 -m pip install 'setuptools==80.10.2' wheel
python3 -c "import pkg_resources; print('pkg_resources ok')"
python3 -c "import pip; assert pip.__file__.startswith('${VENV_PATH}'), pip.__file__"
python3 -m pip --version

echo "[INFO] Installing Isaac Sim pip packages (same as C2 venv, no sudo)..."
python3 -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

echo "[INFO] Installing PyTorch..."
python3 -m pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "[INFO] Restoring build deps (isaacsim pins packaging==23.0; flatdict needs pkg_resources)..."
python3 -m pip install --force-reinstall 'setuptools==80.10.2' wheel packaging==23.0
python3 -c "import pkg_resources; print('pkg_resources ok')"
python3 -m pip install --no-build-isolation flatdict==4.0.1

echo "[INFO] Installing Isaac Lab extensions from source (editable)..."
cd "${ISAACLAB_PATH}/source"
for ext in */; do
  if [[ -f "${ext}setup.py" ]] || [[ -f "${ext}pyproject.toml" ]]; then
    echo "[INFO] pip install -e ${ext}"
    python3 -m pip install -e "${ext}"
  fi
done

cd "${ISAACLAB_PATH}"

echo "[INFO] Skipping isaaclab_rl / isaaclab_mimic by default (pulls duplicate CUDA wheels; needs git)."
echo "[INFO] create_empty.py and demos only need isaaclab core (installed above)."
echo "[INFO] To add RL later (after freeing quota):"
echo "  pip install -e source/isaaclab_rl --no-deps"
echo "  pip install hydra-core h5py tensorboard moviepy stable-baselines3 skrl tqdm rich gym"
echo "  pip install https://github.com/isaac-sim/rl_games/archive/refs/heads/python3.11.zip"

echo "[INFO] Verifying imports..."
python3 -c "import isaacsim; print('isaacsim ok')"
python3 -c "import isaaclab; print('isaaclab ok')"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "[INFO] Done. Activate with:"
echo "  source ${VENV_PATH}/bin/activate"
