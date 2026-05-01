out_dir = 'out-fineweb-test'
eval_interval = 10
eval_iters = 5
log_interval = 1
always_save_checkpoint = False

wandb_log = False

dataset = 'fineweb'
gradient_accumulation_steps = 1
batch_size = 4
block_size = 128

# Tiny model for CPU test
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
learning_rate = 3e-4
max_iters = 100
warmup_iters = 5
lr_decay_iters = 100
min_lr = 3e-5
beta2 = 0.99

compile = False
bias = False
weight_decay = 0.1