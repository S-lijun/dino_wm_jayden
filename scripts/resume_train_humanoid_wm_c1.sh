#!/bin/bash
# Resume DINO-WM training from an existing Hydra run directory.
#
# Usage:
#   bash scripts/resume_train_humanoid_wm_c1.sh /path/to/checkpt_dino/outputs/YYYY-MM-DD/HH-MM-SS
#
# The run dir must contain checkpoints/model_latest.pth (saved after each epoch).

set -euo pipefail

RUN_DIR="${1:?Usage: $0 /path/to/hydra/run/dir}"

if [[ ! -f "${RUN_DIR}/checkpoints/model_latest.pth" ]]; then
  echo "ERROR: no checkpoint at ${RUN_DIR}/checkpoints/model_latest.pth"
  echo "Wait until epoch 1 finishes and 'Saved model to ...' appears in the log."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env_dino_wm_ris.sh"

cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden

echo "Resuming from: ${RUN_DIR}"

"${PYTHON}" train.py --config-name train.yaml \
  hydra.run.dir="${RUN_DIR}" \
  env=humanoid \
  frameskip=1 \
  num_hist=3 \
  env.dataset.window_stride=1 \
  img_h=144 \
  img_w=192 \
  ckpt_base_path=/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino \
  training.batch_size=16 \
  training.epochs=50 \
  env.num_workers=24
