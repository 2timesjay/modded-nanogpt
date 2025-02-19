# Current best: --cooldown_frac 0.01 --adam_head_lr 0.1 --adam_embed_lr 1.0 --adam_scalar_lr 0.01 --muon_lr 0.031 --muon_momentum 0.9 --iter_frac 0.17
torchrun --standalone --nproc_per_node=1 train_gpt.py "$@"
