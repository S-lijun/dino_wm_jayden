#!/bin/bash

### bsub -n 16 -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB" -R "rusage[mem=100]" -q gpu-compute-debug -Is /bin/bash 
### bsub -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAGeForceGTX1080Ti" -q interactive -Is /bin/bash

##run below using source /path.sh
source /storage1/sibai/Active/ihab/miniconda3/bin/activate
conda activate dino_wm
#cd /storage1/sibai/Active/ihab/research_new/dino_wm
export DATASET_DIR=/storage1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/sibai/Active/ihab/tmp/torch
# export HOME=/storage1/sibai/Active/ihab/tmp/

python test_dubins_latent_aci_sf.py 2>&1 | tee sacha_logs/log11.txt
#python plot_dubins_latent_aci_sf.py 2>&1 | tee sacha_logs/logplot.txt