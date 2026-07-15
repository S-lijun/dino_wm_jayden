#!/bin/bash

### Setup
# export LSF_DOCKER_VOLUMES="/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active"
# export LSF_DOCKER_SHM_SIZE='64g'
# bsub -n 8 -Is -q general-interactive -R 'rusage[mem=64GB]' -M 64GB -R 'gpuhost' -gpu "num=1:gmem=31G"  -a 'docker(continuumio/anaconda3:2021.11)'  /bin/bash
# bsub -n 4 -Is -q general-interactive -R 'rusage[mem=16GB]' -M 16GB -R 'gpuhost' -gpu "num=1:gmem=7G"  -a 'docker(continuumio/anaconda3:2021.11)'  /bin/bash
# source /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/script_aci_ris.sh

export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export DATASET_DIR=/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
source /opt/conda/etc/profile.d/conda.sh
conda activate dino_wm_ris
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm

### CarGoal
#python test_cargoal_latent_specific_ckpt_aci.py -f # f for critic (+finetune), d for dynamics

### Dubins
mkdir dubins_test/ACI/ris/test8
python test_dubins_latent_aci_sf.py 2>&1 | tee dubins_test/ACI/ris/test8/log.txt
#python test_dubins_latent_aci_sf.py > dubins_test/ACI/ris/test3/log.txt

### Plots
#python plot_sacha.py