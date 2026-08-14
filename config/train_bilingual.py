# Real training on the bilingual (EN/ZH) 65k-vocab corpus from
# /mnt/data/bilingual-gpt (mixed_sample_2gb.txt -> data/{train,val}.bin).
#
# Corpus: 434M train tokens, 22.9M val tokens (vocab=65536, uint16 ids).
# Model: ~232M params (L18/H16/C896, same body as the gpt2-200m run but
#   with a 65k-token embedding: 65.5M embed vs 50.3M for GPT-2's 50k vocab).
# Budget: ~524k tokens/step -> ~828 steps/epoch over 434M train tokens.
#   max_iters=5000  ≈ 2.62B tokens  ≈ 6.0 epochs (data is small, so we
#   retrain over it multiple times; watch val loss for overfitting).
# VRAM: bs=4 * block=1024 peaked ~8 GB at bs=4 on the 4070 for the 219M
#   model; the +13M embed params keep it well under 12 GB. compile=True.
#
# Run from /mnt/data/nanoGPT:
#   python train.py config/train_bilingual.py
# Resume:
#   python train.py config/train_bilingual.py --init_from=resume

out_dir = 'out-bilingual'

# Absolute path -> train.py uses it as-is for train.bin/val.bin/meta.pkl.
dataset = '/mnt/data/bilingual-gpt/data'

eval_interval = 250
eval_iters = 100
log_interval = 20
always_save_checkpoint = True

wandb_log = False
wandb_project = 'bilingual-gpt'
wandb_run_name = 'bilingual-232m-2.6B'

# 524,288 tokens/step = 4 * 1024 * 128.
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 128

# Model: same body as gpt2-200m; larger vocab handled by the 65k embedding.
n_layer = 18
n_head = 16
n_embd = 896
dropout = 0.0
bias = False

# Optimizer — LR matches the 200M run (between nanoGPT 124M 6e-4 and GPT-3 350M 3e-4).
learning_rate = 4e-4
min_lr = 4e-5
warmup_iters = 500          # ~0.26B tokens, GPT-3 style
max_iters = 5000            # ~2.62B tokens  (~6 epochs over 434M train)
lr_decay_iters = 5000
weight_decay = 0.1
beta2 = 0.95
grad_clip = 1.0

compile = True
