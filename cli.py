import argparse

import tiktoken
import torch

from transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Interact with a trained translation model.")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Path to the saved checkpoint.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="Override the auto-selected device.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum tokens to generate per translation.")
    return parser.parse_args()


args = parse_args()
device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# load checkpoint
# -----------------------
ckpt = torch.load(args.checkpoint, map_location=device)

vocab_size = ckpt["tokenizer_vocab_size"]
pad_id = ckpt["pad_id"]

# reconstruct tokenizer (same as training)
tokenizer_model = ckpt.get("tokenizer_model", "gpt-2")
enc = tiktoken.encoding_for_model(tokenizer_model)

# special tokens (must match training)
PAD_ID = pad_id
BOS_ID = enc.n_vocab + 1
EOS_ID = enc.n_vocab + 2


# -----------------------
# model hyperparams (loaded from checkpoint, with fallback for older checkpoints)
# -----------------------
model_config = ckpt.get("model_config", {})
block_size = model_config.get("block_size", 64)
n_embd = model_config.get("n_embd", 512)
n_layers = model_config.get("n_layers", 6)
n_heads = model_config.get("n_heads", 8)
dropout = model_config.get("dropout", 0.1)


# -----------------------
# load model
# -----------------------
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    pad_id=PAD_ID,
    block_size=block_size,
    n_embd=n_embd,
    n_layers=n_layers,
    n_heads=n_heads,
    dropout=dropout,
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded.")


# -----------------------
# helper: decode ids safely
# -----------------------
def decode(ids):
    filtered = [i for i in ids if i not in {PAD_ID, BOS_ID, EOS_ID}]
    return enc.decode(filtered)


# -----------------------
# generation (greedy)
# -----------------------
@torch.no_grad()
def generate(src_ids, max_new_tokens=50):
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    # start with BOS
    tgt = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(src, tgt)

        # take last timestep
        next_token_logits = logits[:, -1, :]  # (1, vocab)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1,1)

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == EOS_ID:
            break

    return tgt[0].tolist()


# -----------------------
# CLI loop
# -----------------------
print("\nType a German sentence (or 'quit'):\n")

while True:
    text = input(">> ")

    if text.lower() in {"quit", "exit"}:
        break

    # encode source
    src_ids = enc.encode(text)

    if len(src_ids) > block_size:
        print(f"Input too long (max {block_size} tokens)")
        continue

    # generate
    tgt_ids = generate(src_ids, max_new_tokens=args.max_new_tokens)

    # decode
    output = decode(tgt_ids)

    print(f"EN: {output}\n")