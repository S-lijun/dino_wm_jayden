#!/bin/bash
# Train DINO-WM on G1 humanoid data (Compute1, anaconda docker + dino_wm_ris).
#
# Interactive (debug):
#   Option A — anaconda container (recommended):
#     export LSF_DOCKER_VOLUMES='/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active'
#     bsub ... -a 'docker(continuumio/anaconda3:2021.11)' /bin/bash
#   Option B — reuse current GPU exec shell (Isaac or not), then:
#     source scripts/env_dino_wm_ris.sh
#     bash scripts/run_train_humanoid_wm_c1.sh
#
# Batch:
#   bash scripts/submit_train_humanoid_wm_c1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env_dino_wm_ris.sh"

cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden

"${PYTHON}" scripts/smoke_test_humanoid_dset.py

CKPT_ROOT=/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs
echo ""
echo ">>> New run. After start, note Hydra output dir (or latest):"
echo "    ls -lt ${CKPT_ROOT}/\$(date +%Y-%m-%d) | head"
echo ">>> Resume after epoch 1:"
echo "    bash scripts/resume_train_humanoid_wm_c1.sh <RUN_DIR>"
echo ""

"${PYTHON}" train.py --config-name train.yaml \
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
