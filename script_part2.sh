#!/bin/bash
#BSUB -n 24
#BSUB -q general
#BSUB -R 'rusage[mem=102GB]'
#BSUB -G compute-sibai
#BSUB -M 100GB
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1:gmem=30G"
#BSUB -a 'docker(maniskill/base)'
#BSUB -W 600:00
#BSUB -J dino_wm_job
#BSUB -oo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/newnew/output%J.log
#BSUB -eo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/newnew/error%J.log
#BSUB -N
#BSUB -u y.yuxuan@wustl.edu
#BSUB -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g"

export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export WANDB_API_KEY=7893bf6676aaa0213e6da2edbc8f4b42fa816084

source /opt/conda/etc/profile.d/conda.sh
# conda activate mani
conda activate dino_wm_ris
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
wandb login


python test_dubins_latent_specific_checkpoint.py --backbone r3m --pid_only --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone dino_cls --finetune --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone scratch --finetune --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone vc1 --finetune --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone dino --finetune --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone resnet --finetune --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone r3m --finetune --eps 0.2

python test_dubins_latent_specific_checkpoint.py --backbone dino_cls --only_dynamics --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone scratch --only_dynamics --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone vc1 --only_dynamics --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone dino --only_dynamics --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone resnet --only_dynamics --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone r3m --only_dynamics --eps 0.2

python test_dubins_latent_specific_checkpoint.py --backbone full_scratch --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone dino_cls --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone scratch --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone vc1 --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone dino --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone resnet --eps 0.2
python test_dubins_latent_specific_checkpoint.py --backbone r3m --eps 0.2