import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import TokenizerConfig, build_dataloader
from transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the encoder-decoder Transformer.")
    parser.add_argument("--train-data", default="data/wmt14_translate_de-en_train.csv")
    parser.add_argument("--val-data", default="data/wmt14_translate_de-en_validation.csv")
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")
    parser.add_argument("--loss-plot-path", default="figures/loss_curve.png")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="Override the auto-selected device.")
    parser.add_argument("--block-size", type=int, default=64, help="Maximum sequence length.")
    parser.add_argument("--n-embd", type=int, default=512, help="Embedding size.")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of encoder and decoder blocks.")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--max-iters", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--eval-interval", type=int, default=500, help="Steps between evaluations.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--eval-iters", type=int, default=100, help="Number of batches to average for evaluation.")
    return parser.parse_args()


args = parse_args()

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = Path(args.checkpoint_path)
loss_plot_path = Path(args.loss_plot_path)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
loss_plot_path.parent.mkdir(parents=True, exist_ok=True)

# build tokenizer + data
tokenizer_cfg = TokenizerConfig()

train_loader = build_dataloader(args.train_data, tokenizer_cfg, batch_size=args.batch_size)
val_loader = build_dataloader(args.val_data, tokenizer_cfg, batch_size=args.batch_size, shuffle=False)

# handy iterator wrapper
def get_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

# build model
model = Transformer(
    src_vocab_size=tokenizer_cfg.vocab_size,
    tgt_vocab_size=tokenizer_cfg.vocab_size,
    pad_id=tokenizer_cfg.PAD_ID,
    block_size=args.block_size,
    n_embd=args.n_embd,
    n_layers=args.n_layers,
    n_heads=args.n_heads,
    dropout=args.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    model.eval()
    out = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        loader_iter = iter(loader)

        for _ in range(eval_iters):
            try:
                src, tgt_in, tgt_out = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                src, tgt_in, tgt_out = next(loader_iter)

            src = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            _, loss = model(src, tgt_in, tgt_out)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out

# training loop
train_losses = []
val_losses = []
eval_steps = []

train_iter = iter(train_loader)

for step in range(args.max_iters):
    # periodic eval
    if step % args.eval_interval == 0:
        losses = estimate_loss(model, train_loader, val_loader, args.eval_iters)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        eval_steps.append(step)
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

    # get batch
    batch, train_iter = get_batch(train_iter, train_loader)
    src, tgt_in, tgt_out = batch

    src = src.to(device)
    tgt_in = tgt_in.to(device)
    tgt_out = tgt_out.to(device)

    # forward
    logits, loss = model(src, tgt_in, tgt_out)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "eval_steps": eval_steps,
        "tokenizer_vocab_size": tokenizer_cfg.vocab_size,
        "pad_id": tokenizer_cfg.PAD_ID,
        "tokenizer_model": "gpt-2",
        "model_config": {
            "block_size": args.block_size,
            "n_embd": args.n_embd,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
        },
        "train_config": {
            "train_data": args.train_data,
            "val_data": args.val_data,
            "checkpoint_path": args.checkpoint_path,
            "loss_plot_path": args.loss_plot_path,
            "device": device,
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "eval_interval": args.eval_interval,
            "learning_rate": args.learning_rate,
            "eval_iters": args.eval_iters,
        },
    },
    checkpoint_path,
)

print(f"Saved checkpoint to {checkpoint_path}")

# plot loss
plt.figure(figsize=(8, 5))
plt.plot(eval_steps, train_losses, label="train loss")
plt.plot(eval_steps, val_losses, label="val loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(loss_plot_path)
plt.show()