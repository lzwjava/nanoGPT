# SEC-EDGAR 124M — smoke test
# Verify VRAM fits and measure tok/s on RTX 4070

out_dir = 'out-sec-edgar-124m-smoke'
shard_dir = '/mnt/data/zz/datasets/sec-edgar-tok'
dataset = 'sec-edgar'

eval_interval = 1000
eval_iters = 5
log_interval = 1
always_save_checkpoint = False

wandb_log = False

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 8

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
min_lr = 6e-5
warmup_iters = 5
max_iters = 25
lr_decay_iters = 25
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

compile = False
