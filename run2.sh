python test_maniskill_latent_specific_checkpoint.py --backbone r3m --pid_only
python test_maniskill_latent_specific_checkpoint.py --backbone dino_cls --finetune 
python test_maniskill_latent_specific_checkpoint.py --backbone scratch --finetune 
python test_maniskill_latent_specific_checkpoint.py --backbone vc1 --finetune 
python test_maniskill_latent_specific_checkpoint.py --backbone dino --finetune 
python test_maniskill_latent_specific_checkpoint.py --backbone resnet --finetune 
python test_maniskill_latent_specific_checkpoint.py --backbone r3m --finetune 

python test_maniskill_latent_specific_checkpoint.py --backbone dino_cls --only_dynamics 
python test_maniskill_latent_specific_checkpoint.py --backbone scratch --only_dynamics 
python test_maniskill_latent_specific_checkpoint.py --backbone vc1 --only_dynamics 

