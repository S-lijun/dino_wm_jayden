#!/bin/bash
# Merge humanoid sessions on Compute1 using a plain PyTorch container (no Isaac Sim).
set -euo pipefail

export LSF_DOCKER_VOLUMES='/storage1/fs1/sibai/Active/ihab/research_new:/workspace'
export LSF_DOCKER_SHM_SIZE='16g'
export LSF_DOCKER_PRESERVE_ENVIRONMENT=false

LOG_DIR="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/logs"
mkdir -p "$LOG_DIR"

bsub -G compute-sibai -q general \
  -n 4 \
  -R 'rusage[mem=32GB]' -M 32GB \
  -W 240 \
  -o "$LOG_DIR/merge_humanoid_c1_%J.out" \
  -e "$LOG_DIR/merge_humanoid_c1_%J.err" \
  -a 'docker(pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime)' \
  /bin/bash -lc '
    pip install -q --no-cache-dir einops
    python /workspace/dino_wm_jayden/scripts/merge_humanoid_sessions.py \
      --base-dir /workspace/datasets_dino/humanoid_g1
  '

echo "Submitted. Check: bjobs -u \$USER"
echo "Logs: $LOG_DIR/merge_humanoid_c1_<JOBID>.out"
