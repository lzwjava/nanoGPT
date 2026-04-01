# FineWeb 1.5B config — for H200 143GB
# Based on GPT-2 XL architecture (1.5B params)
# Run: torchrun --standalone --nproc_per_node=1 train.py config/train_fineweb1_5b.py

out_dir = 'out-fineweb1_5b'
eval_interval = 250
eval_iters = 200
log_interval = 50
always_save_checkpoint = True

wandb_log = False
wandb_project = 'fineweb'
wandb_run_name = 'fineweb1_5b'

dataset = 'fineweb'
gradient_accumulation_steps = 64
batch_size = 16          # ~1M tokens effective batch (16 * 1024 * 64)
block_size = 1024

# Model — GPT-2 XL: 1.5B params
n_layer = 48
n_head = 25
n_embd = 1600
dropout = 0.0
learning_rate = 3e-4
max_iters = 10000       # ~6.7B tokens with this config — good for 10B FineWeb
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 3e-5
beta2 = 0.99

# Extras
compile = True
bias = False
weight_decay = 0.1
