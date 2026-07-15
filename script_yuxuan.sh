

./run4.sh
# ./run4.sh
# python test_dubins_latent_specific_checkpoint.py --backbone dino_cls  --finetune
# python "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent_withfinetune_ddqn.py" --dino_ckpt_dir "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/output3_frameskip1/dubins"  --config train_HJ_configs.yaml --dino_encoder full_scratch --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 256 --gamma-pyhj 0.98 --critic-net 512 512 512 --with_finetune --encoder_lr 1e-6

# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task maniskill3000classif --epochs 2 # TeslaV100_SXM2_32GB NVIDIAA40 NVIDIAA100_SXM4_40GB 
#python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_withcost --without_proprio --epochs 2
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task cargoalnewshort --epochs 5

# ./run2.sh

##ran this <529343>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder r3m --with_finetune
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder resnet --with_finetune

##ran this <529345>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder vc1 --with_finetune
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder dino_cls --with_finetune

# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder dino
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder dino --with_finetune

# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder dino



# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder scratch --with_finetune
# directly submit this
# <565540>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder scratch --with_finetune

# <565681>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_visual_ft.py --dino_encoder dino --with_finetune

# # run it two times, one comment line 444, one uncomment line 444 (such that it's like we run this line with two jobs)
# # or if possible, split the backbones into multiple jobs to make it even faster (e.g., 5 jobs for 5 different backbones)

# <565691>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task carla_2k_v --epochs 10

# # same as above
# <565723>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task carla_2k_v --epochs 10 --finetune

# # only run it for resnet
# <595096>
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task maniskill3000classif --epochs 2

# python train_HJ_mani_ft.py --dino_encoder resnet --use_latent_h --with_finetune
# python train_HJ_mani_ft.py --dino_encoder r3m --use_latent_h --with_finetune
# python train_HJ_mani_ft.py --dino_encoder scratch --use_latent_h --with_finetune
# python train_HJ_mani_ft.py --dino_encoder dino --use_latent_h --with_finetune
# python train_HJ_mani_ft.py --dino_encoder vc1 --use_latent_h --with_finetune
# python train_HJ_mani_ft.py --dino_encoder dino_cls --use_latent_h --with_finetune

# python /storae1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_continuous_cost  --without_proprio #!/bin/bash
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
conda activate mani
# conda activate dino_wm_ris
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
wandb login