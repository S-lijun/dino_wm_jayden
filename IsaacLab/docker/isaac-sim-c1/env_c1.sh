# Source inside the C1 Isaac Sim container before running Isaac Lab.
#   source /workspace/dino_wm_jayden/IsaacLab/docker/isaac-sim-c1/env_c1.sh

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export GIT_PYTHON_REFRESH=quiet
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/pip-cache-${USER}"

# Avoid: failed to create '/home/$USER/.nvidia-omniverse' (Permission denied)
export HOME=/isaac-sim

# shellcheck source=/dev/null
source /workspace/dino_wm_jayden/IsaacLab/docker/isaac-sim-c1/.env.c1
# shellcheck source=/dev/null
source "/workspace/venvs/${VENV_NAME}/bin/activate"
