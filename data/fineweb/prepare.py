# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Read the local fineweb.txt file
    txt_file = os.path.join(os.path.dirname(__file__), 'train_fineweb.txt')
    print(f"Reading from local file: {txt_file}")

    # Use chunked reading for large file
    chunk_size = 10_000_000  # characters per chunk
    all_tokens = []

    with open(txt_file, "r", encoding="utf-8", errors='ignore') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            tokens = enc.encode_ordinary(chunk)
            all_tokens.extend(tokens)
            print(f"Tokenized so far: {len(all_tokens)/1e6:.1f}M tokens")

    # Convert to numpy array
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)/1e6:.1f}M")

    # Create a simple dataset structure for compatibility
    # Split into train/val
    val_size = int(len(all_tokens) * 0.0005)
    train_tokens = all_tokens[val_size:]
    val_tokens = all_tokens[:val_size]

    # Store as binary files directly
    for split_name, tokens in [('train', train_tokens), ('val', val_tokens)]:
        filename = os.path.join(os.path.dirname(__file__), f'{split_name}.bin')
        tokens.tofile(filename)
        print(f"Saved {split_name}: {len(tokens)/1e6:.1f}M tokens to {filename}")

    print("Processing complete!")

    # train.bin and val.bin are now ready for training
    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
