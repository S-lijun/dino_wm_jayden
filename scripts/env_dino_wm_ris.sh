#!/bin/bash
# Activate dino_wm_ris using the shared lab env on Storage1.
# Works inside anaconda docker OR on exec nodes — do NOT rely on Isaac Sim python.

export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export DATASET_DIR=/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export DINO_WM_ROOT="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden"
export PYTHONPATH="${DINO_WM_ROOT}:${PYTHONPATH:-}"

# Disable wandb login prompt by default. Set to "online" if you want cloud logging.
export WANDB_MODE="${WANDB_MODE:-disabled}"

DINO_WM_PYTHON="/storage1/fs1/sibai/Active/ihab/conda/envs/dino_wm_ris/bin/python"

if [[ ! -x "${DINO_WM_PYTHON}" ]]; then
  echo "[ERROR] Missing ${DINO_WM_PYTHON}"
  exit 1
fi

# Prefer the shared env binary explicitly (conda activate alone can point at wrong python).
export PATH="/storage1/fs1/sibai/Active/ihab/conda/envs/dino_wm_ris/bin:${PATH}"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # Inside continuumio/anaconda3 container.
  export PATH="/opt/conda/bin:${PATH}"
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda activate dino_wm_ris
fi

if ! "${DINO_WM_PYTHON}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  echo "[ERROR] dino_wm_ris python cannot import torch."
  echo "which python: $(which python)"
  echo "expected: ${DINO_WM_PYTHON}"
  exit 1
fi

export PYTHON="${DINO_WM_PYTHON}"
