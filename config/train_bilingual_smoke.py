# Smoke test for the bilingual (EN/ZH) 65k-vocab corpus produced by
# /mnt/data/bilingual-gpt (mixed_sample_2gb.txt -> data/{train,val}.bin).
#
# Goal: verify the pipeline end-to-end on the RTX 4070 with limited free
# VRAM (~4 GiB) — vocab loads, loss decreases, checkpoint saves. NOT a
# convergence run. Bump block_size / model size for a real run.
#
# Run from /mnt/data/nanoGPT:
#   python train.py config/train_bilingual_smoke.py

out_dir = 'out-bilingual-smoke'

# data: absolute path -> train.py does os.path.join('data', dataset) which
# keeps the absolute path when it starts with '/'. meta.pkl inside gives
# vocab_size=65536.
dataset = '/mnt/data/bilingual-gpt/data'

eval_interval = 50
eval_iters = 10
log_interval = 1
always_save_checkpoint = False

wandb_log = False
wandb_project = 'bilingual-gpt'
wandb_run_name = 'smoke'

# tiny model that fits ~4 GiB free VRAM; vocab=65536 dominates the embed+lm_head
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

block_size = 256
batch_size = 8
gradient_accumulation_steps = 4

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 10
max_iters = 50
lr_decay_iters = 50
weight_decay = 0.1
beta2 = 0.95
grad_clip = 1.0

compile = False
