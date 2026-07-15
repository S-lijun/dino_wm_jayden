#!/bin/bash
set -euo pipefail

export LSF_DOCKER_VOLUMES='/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active'
export LSF_DOCKER_SHM_SIZE='64g'

LOG_DIR="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/logs"
mkdir -p "$LOG_DIR"

bsub -G compute-sibai -q general \
  -n 8 \
  -R 'rusage[mem=64GB]' -M 60GB \
  -R 'gpuhost' \
  -gpu "num=1:gmem=31G" \
  -W 600:00 \
  -o "$LOG_DIR/train_humanoid_wm_%J.out" \
  -e "$LOG_DIR/train_humanoid_wm_%J.err" \
  -a 'docker(continuumio/anaconda3:2021.11)' \
  /bin/bash -lc 'bash /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/scripts/run_train_humanoid_wm_c1.sh'

echo "Submitted. Check: bjobs -u \$USER"
echo "Logs: $LOG_DIR/train_humanoid_wm_<JOBID>.{out,err}"
