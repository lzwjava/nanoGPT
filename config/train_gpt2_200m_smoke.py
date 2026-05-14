# Smoke test for the 200M config: verify VRAM fits and measure ms/iter on the 4070.
# Same architecture as train_gpt2_200m.py, but: no compile (faster startup),
# tiny grad-accum, ~25 iters, log every step.

out_dir = 'out-gpt2-200m-smoke'
shard_dir = '/mnt/data/nanoGPT/data/fineweb/edu_fineweb100B'
dataset = 'fineweb'

eval_interval = 1000     # don't eval during smoke
eval_iters = 5
log_interval = 1
always_save_checkpoint = False

wandb_log = False

batch_size = 2
block_size = 1024
gradient_accumulation_steps = 4   # tiny — we just want timing per micro-batch

# Same architecture as the full run
n_layer = 18
n_head = 16
n_embd = 896
dropout = 0.0
bias = False

learning_rate = 4e-4
min_lr = 4e-5
warmup_iters = 5
max_iters = 25
lr_decay_iters = 25
weight_decay = 0.1
beta2 = 0.95
grad_clip = 1.0

compile = False
