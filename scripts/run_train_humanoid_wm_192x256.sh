#!/bin/bash
# Train DINO-WM @ 192x256 (480:640 @ 0.4x). Higher quality, slower than 144x192.
#
#   source scripts/env_dino_wm_ris.sh
#   bash scripts/run_train_humanoid_wm_192x256.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env_dino_wm_ris.sh"

cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden

"${PYTHON}" scripts/smoke_test_humanoid_dset.py 192 256

CKPT_ROOT=/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs
echo ""
echo ">>> 192x256 new run. Note output dir:"
echo "    ls -lt ${CKPT_ROOT}/\$(date +%Y-%m-%d) | head"
echo ">>> Resume:"
echo "    bash scripts/resume_train_humanoid_wm_192x256.sh <RUN_DIR>"
echo ""

"${PYTHON}" train.py --config-name train.yaml \
  env=humanoid \
  frameskip=1 \
  num_hist=3 \
  env.dataset.window_stride=1 \
  img_h=192 \
  img_w=256 \
  ckpt_base_path=/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino \
  training.batch_size=16 \
  training.epochs=50 \
  env.num_workers=24
