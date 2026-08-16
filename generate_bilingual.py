#!/usr/bin/env python3
"""
Generate bilingual (EN/ZH) text from the out-bilingual checkpoint with a
language knob: prepend <|lang_en|> / <|lang_zh|> (the same byte-BPE
fragments the model saw during training) to steer the language.

Usage:
    ./venv/bin/python generate_bilingual.py \
        --ckpt out-bilingual/ckpt.pt \
        --tok-prefix /mnt/data/bilingual-gpt/tok_mixed \
        --lang zh --prompt "机器学习是什么？" --max-new-tokens 200

    # same prompt, both languages, to show the knob:
    ./venv/bin/python generate_bilingual.py --lang en --prompt "The future of artificial intelligence"
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig  # noqa: E402

LANG_TOK = {"en": "<|lang_en|>", "zh": "<|lang_zh|>"}


def load_tokenizer(tok_prefix):
    from tokenizers import ByteLevelBPETokenizer
    vocab = glob.glob(os.path.join(tok_prefix, "*vocab.json"))
    merges = glob.glob(os.path.join(tok_prefix, "*merges.txt"))
    if not (vocab and merges):
        raise FileNotFoundError(f"tokenizer not found under {tok_prefix}")
    return ByteLevelBPETokenizer(vocab[0], merges[0])


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # strip torch.compile "_orig_mod." prefix if present
    unwanted = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted):
            state_dict[k[len(unwanted):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, checkpoint


def clean_output(text, lang):
    """Strip the control-token fragments / EOT that may appear in output."""
    for tag in ["<|lang_en|>", "<|lang_zh|>", "<|endoftext|>"]:
        text = text.replace(tag, "")
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out-bilingual/ckpt.pt")
    ap.add_argument("--tok-prefix", default="/mnt/data/bilingual-gpt/tok_mixed")
    ap.add_argument("--lang", choices=["en", "zh"], default="zh")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to cpu", file=sys.stderr)
        a.device = "cpu"

    device = a.device
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed(a.seed)

    tok = load_tokenizer(a.tok_prefix)
    model, ckpt = load_model(a.ckpt, device)
    print(f"[model] iter {ckpt['iter_num']}  best_val_loss {ckpt['best_val_loss']:.4f}  "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  device {device}")

    # build the prompt: language tag + newline + user prompt (matches training docs)
    prompt = f"{LANG_TOK[a.lang]}\n{a.prompt}"
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"[prompt ids {len(ids)}] {prompt!r}\n")

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float32
    ctx = torch.amp.autocast("cuda", dtype=dtype) if device.startswith("cuda") else torch.no_grad()

    with torch.no_grad(), ctx:
        for i in range(a.num_samples):
            y = model.generate(x, max_new_tokens=a.max_new_tokens,
                               temperature=a.temperature, top_k=a.top_k)
            text = tok.decode(y[0].tolist())
            print(f"===== sample {i+1}/{a.num_samples} (lang={a.lang}, T={a.temperature}, top_k={a.top_k}) =====")
            print(clean_output(text, a.lang))
            print()


if __name__ == "__main__":
    sys.exit(main())
