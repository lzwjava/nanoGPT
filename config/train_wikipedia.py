# train on openwebtext dataset
# uses data prepared by data/openwebtext_local/prepare.py

out_dir = 'out-wikipedia'
eval_interval = 500 # evaluate less frequently on larger dataset
eval_iters = 200
log_interval = 100 # don't print too often

# save checkpoint when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'wikipedia'
wandb_run_name = 'nano-gpt'

dataset = 'openwebtext_local'
gradient_accumulation_steps = 4 # increase for effective batch size while reducing per-iteration memory
batch_size = 16 # reduced for memory constraints
block_size = 512 # reduced context length for memory

# GPT model for openwebtext (smaller for limited GPU memory)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0

learning_rate = 3e-4 # standard for GPT models
max_iters = 20000
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 # standard for GPT models

warmup_iters = 2000 # more important for larger models

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
