#!/bin/bash
set -euo pipefail

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export LSF_DOCKER_VOLUMES='/storage1/fs1/sibai/Active/ihab/research_new:/workspace'
export LSF_DOCKER_SHM_SIZE='16g'
export LSF_DOCKER_ENTRYPOINT=/bin/bash
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

LOG_DIR="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/logs"
mkdir -p "$LOG_DIR"

bsub -G compute-sibai -q general \
  -n 4 \
  -R 'rusage[mem=32GB]' -M 32GB \
  -W 180 \
  -o "$LOG_DIR/merge_humanoid_%J.out" \
  -e "$LOG_DIR/merge_humanoid_%J.err" \
  -a 'docker(nvcr.io/nvidia/isaac-sim:5.1.0)' \
  /bin/bash -lc '
    export HOME=/isaac-sim
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y
    /isaac-sim/python.sh /workspace/dino_wm_jayden/scripts/merge_humanoid_sessions.py \
      --base-dir /workspace/datasets_dino/humanoid_g1
  '
