# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
CHUNK_SIZE = 10_000_000  # characters per chunk
VAL_RATIO = 0.0005

def process_file(input_path: str, train_bin: str, val_bin: str, val_ratio=VAL_RATIO):
    temp_train = train_bin + '.tmp'
    temp_val = val_bin + '.tmp'

    total_tokens = 0
    val_tokens_written = 0
    val_target = None  # we decide it after first pass or approximate

    with open(input_path, "r", encoding="utf-8", errors='ignore') as f, \
         open(temp_train, "wb") as train_f, \
         open(temp_val, "wb") as val_f:

        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            tokens = enc.encode_ordinary(chunk)
            tokens_u16 = np.array(tokens, dtype=np.uint16)

            total_tokens += len(tokens_u16)

            # Approximate validation split on-the-fly (good enough)
            if val_target is None and total_tokens > 10_000_000:
                val_target = int(total_tokens * val_ratio / (1 - val_ratio))

            if val_tokens_written < val_target:
                split_point = min(len(tokens_u16), val_target - val_tokens_written)
                val_f.write(tokens_u16[:split_point].tobytes())
                train_f.write(tokens_u16[split_point:].tobytes())
                val_tokens_written += split_point
            else:
                train_f.write(tokens_u16.tobytes())

            print(f"Processed {total_tokens/1e6:.1f}M tokens")

    # Rename temp files
    os.rename(temp_train, train_bin)
    os.rename(temp_val, val_bin)
    print(f"Done! Total ≈ {total_tokens/1e9:.2f}B tokens")
    print(f"train.bin and val.bin ready (no RAM explosion)")

if __name__ == '__main__':
    # Read the local fineweb.txt file
    txt_file = os.path.join(os.path.dirname(__file__), 'train_fineweb.txt')
    print(f"Reading from local file: {txt_file}")

    train_bin = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_bin = os.path.join(os.path.dirname(__file__), 'val.bin')

    process_file(txt_file, train_bin, val_bin)

    print("Processing complete!")

    # train.bin and val.bin are now ready for training
    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
