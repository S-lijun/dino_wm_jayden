#!/bin/bash
# Resume 96x128 humanoid WM training.

set -euo pipefail

RUN_DIR="${1:?Usage: $0 /path/to/hydra/run/dir}"

if [[ ! -f "${RUN_DIR}/checkpoints/model_latest.pth" ]]; then
  echo "ERROR: no checkpoint at ${RUN_DIR}/checkpoints/model_latest.pth"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env_dino_wm_ris.sh"

cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden

echo "Resuming 96x128 from: ${RUN_DIR}"

"${PYTHON}" train.py --config-name train.yaml \
  hydra.run.dir="${RUN_DIR}" \
  env=humanoid \
  frameskip=1 \
  num_hist=3 \
  env.dataset.window_stride=4 \
  img_h=96 \
  img_w=128 \
  ckpt_base_path=/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino \
  training.batch_size=16 \
  training.epochs=50 \
  env.num_workers=4
