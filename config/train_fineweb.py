out_dir = 'out-fineweb'
eval_interval = 500       # Evaluate more often on small data
eval_iters = 200
log_interval = 200         # Log more frequently
always_save_checkpoint = True

wandb_log = False          # Optional
wandb_project = 'fineweb'
wandb_run_name = 'fineweb'

dataset = 'fineweb'       # Assumes you adapted prepare.py for your single file
gradient_accumulation_steps = 64     # Effective batch size: 16 * 32 = 512 sequences
batch_size = 8
block_size = 1024                    # Matches FineWeb's processing

# Model (~125M parameters) – perfect for 12 GB VRAM
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0                        # Add 0.1 if overfitting
learning_rate = 3e-4                # Slightly lower for smaller data
max_iters = 6000                     # ~3B tokens seen (adjust up to 10000 if loss keeps dropping)
warmup_iters = 500                   # Shorter warmup
lr_decay_iters = 6000
min_lr = 3e-5
beta2 = 0.99

# Extras for speed/stability
compile = True            # PyTorch compile for 20–30% faster training
bias = False              # Like LLaMA/Mistral
weight_decay = 0.1