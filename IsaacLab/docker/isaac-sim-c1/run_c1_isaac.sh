#!/usr/bin/env bash
# Interactive Isaac Sim session on Compute1 using the C1-compatible custom image.
#
# Usage:
#   cd .../IsaacLab/docker/isaac-sim-c1
#   ./run_c1_isaac.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/.env.c1"

FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
CACHE_ROOT="${STORAGE_ROOT}/docker/isaac-sim-c1/cache"

mkdir -p \
  "${CACHE_ROOT}/main" \
  "${CACHE_ROOT}/computecache" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/config" \
  "${CACHE_ROOT}/data" \
  "${CACHE_ROOT}/pkg"

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
# LSF sets HOME=/home/$USER (often not writable in container). Omniverse logs go to
# $HOME/.nvidia-omniverse — point HOME at /isaac-sim where cache volumes are mounted.
export HOME=/isaac-sim
export LSF_DOCKER_ENTRYPOINT=/bin/bash
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true
export LSF_DOCKER_SHM_SIZE=64g
export LSF_DOCKER_VOLUMES="${STORAGE_ROOT}:/workspace \
${CACHE_ROOT}/main:/isaac-sim/.cache \
${CACHE_ROOT}/computecache:/isaac-sim/.nv/ComputeCache \
${CACHE_ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs \
${CACHE_ROOT}/config:/isaac-sim/.nvidia-omniverse/config \
${CACHE_ROOT}/data:/isaac-sim/.local/share/ov/data \
${CACHE_ROOT}/pkg:/isaac-sim/.local/share/ov/pkg"

echo "[INFO] Starting interactive job with image: ${FULL_IMAGE}"
echo "[INFO] After shell starts, run:"
echo "       source /workspace/venvs/${VENV_NAME}/bin/activate   # after setup_venv_c1.sh"
echo "       cd /workspace/dino_wm_jayden/IsaacLab"
echo "       python3 -c \"import isaacsim, isaaclab; print('ok')\""

bsub -G "${LSF_GROUP}" -q "${LSF_QUEUE}" -Is \
  -n 8 -R 'rusage[mem=64GB]' -M 60GB \
  -R 'gpuhost' -gpu "num=1:gmem=31G" \
  -a "docker(${FULL_IMAGE})" /bin/bash
