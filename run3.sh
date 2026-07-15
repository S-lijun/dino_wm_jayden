python test_maniskill_latent_specific_checkpoint.py --backbone r3m --pid_only --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone dino_cls --finetune --eps 0.2 
python test_maniskill_latent_specific_checkpoint.py --backbone scratch --finetune  --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone vc1 --finetune  --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone dino --finetune  --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone resnet --finetune  --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone r3m --finetune  --eps 0.2

python test_maniskill_latent_specific_checkpoint.py --backbone dino_cls --only_dynamics  --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone scratch --only_dynamics  --eps 0.2
python test_maniskill_latent_specific_checkpoint.py --backbone vc1 --only_dynamics  --eps 0.2

