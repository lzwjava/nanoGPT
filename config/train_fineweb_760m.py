# GPT-2 760M config — for single MI300X 192GB, 14.6B tokens
# Architecture: n_layer=24, n_head=24, n_embd=1536 → ~760M params
# Run: python3 train.py config/train_fineweb_760m.py

out_dir = 'out-fineweb-760m'
shard_dir = '/root/nanoGPT/data/fineweb/edu_fineweb100B'
dataset = 'fineweb'

eval_interval = 1000
eval_iters = 200
log_interval = 50
always_save_checkpoint = True

wandb_log = False
wandb_project = 'fineweb-760m'
wandb_run_name = 'gpt2-760m-14B'

# Batch: 32 × 1024 = 32,768 tokens/step
# Total: 445,000 steps × 32,768 = 14.6B tokens
# grad_accum=1 → effective batch = 32,768 (single GPU, no DDP)
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 1

# Model — GPT-2 760M
# ~760M params: 12*24*1536² + vocab*embd = 722M + 77M
n_layer = 24
n_head = 24
n_embd = 1536
dropout = 0.0
bias = False

# Optimizer — GPT-3 style
learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 2000
max_iters = 445000
lr_decay_iters = 445000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Speed
compile = True
