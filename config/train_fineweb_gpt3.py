out_dir = 'out-fineweb-gpt3'
shard_dir = '/mnt/data/nanoGPT/data/fineweb/edu_fineweb100B'
dataset = 'fineweb'

eval_interval = 500
eval_iters = 100
log_interval = 20
always_save_checkpoint = True

wandb_log = False
wandb_project = 'fineweb-gpt3'
wandb_run_name = 'gpt2-124M-100B'

# 524,288 tokens / step (4 * 1024 * 128) — micro-batch 4 fits a 12GB 4070
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 128

# GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 715
max_iters = 19073        # ~10B tokens; bump toward ~190k for full 100B
lr_decay_iters = 19073
weight_decay = 0.1
beta2 = 0.95
grad_clip = 1.0

compile = True
