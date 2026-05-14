# GPT-2 ~219M params, Chinchilla-optimal at ~4B tokens on a single RTX 4070 (12GB).
# Architecture: n_layer=18, n_head=16, n_embd=896  → 219M total / 218.5M non-embedding.
# Training budget: 4B tokens = 524,288 tokens/step * ~7,630 steps (rounded to 8000).

out_dir = 'out-gpt2-200m'
shard_dir = '/mnt/data/nanoGPT/data/fineweb/edu_fineweb100B'
dataset = 'fineweb'

eval_interval = 250
eval_iters = 100
log_interval = 20
always_save_checkpoint = True

wandb_log = False
wandb_project = 'gpt2-200m'
wandb_run_name = 'gpt2-200m-4B'

# 524,288 tokens/step = 4 * 1024 * 128.
# VRAM probe: bs=4 peaks at ~8.0 GB on a 12GB 4070 in bfloat16 — ~4 GB headroom for compile.
# bs=6 also fits (~10.4 GB) but is tighter and only marginally faster per token.
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 128

# Model: GPT-2 ~200M (custom, between small and medium)
n_layer = 18
n_head = 16
n_embd = 896
dropout = 0.0
bias = False

# Optimizer — LR sits between nanoGPT 124M (6e-4) and GPT-3 350M (3e-4).
learning_rate = 4e-4
min_lr = 4e-5
warmup_iters = 715        # ~0.375B tokens, GPT-3 style
max_iters = 8000          # ~4.19B tokens at 524k tokens/step
lr_decay_iters = 8000
weight_decay = 0.1
beta2 = 0.95
grad_clip = 1.0

compile = True
