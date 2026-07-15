#!/bin/bash
# Smoke test: Isaac Sim container on Compute 2
# Usage (on c2-login):
#   cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab
#   ml load ris && ml load slurm
#   sbatch verify_c2_isaac.sh
# Or interactive:
#   srun -A compute2-sibai -p general-gpu --gpus=1 \
#     --container-image='nvcr.io#nvidia/isaac-sim:5.1.0' \
#     --container-mounts='/storage1/fs1/sibai/Active/ihab/research_new:/workspace' \
#     --pty bash verify_c2_isaac.sh

set -euo pipefail

LOG_DIR="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab/logs"
mkdir -p "$LOG_DIR"

echo "=== Isaac Sim C2 smoke test ==="
echo "host: $(hostname)"
echo "date: $(date)"
id

source /workspace/venvs/isaaclab5_pip/bin/activate
cd /workspace/dino_wm_jayden/IsaacLab
rm -f _isaac_sim

echo "--- python ---"
which python3
python3 -c "import sys; print('python', sys.version)"
python3 -c "import isaacsim, isaaclab, torch; print('imports ok'); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export GIT_PYTHON_REFRESH=quiet
echo "--- isaaclab tutorial ---"
bash isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

echo "=== ALL CHECKS PASSED ==="
