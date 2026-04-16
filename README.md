# Encoder-Decoder Transformer

This project is a simple PyTorch implementation of the original encoder-decoder Transformer for German-to-English translation from Attention is All You Need. It uses a GPT-2 BPE tokenizer, trains on the WMT 2014 English-German dataset, and saves a checkpoint that can be loaded from a small command-line interface for interactive translation.

## Project Structure

- `transformer.py` implements the encoder-decoder Transformer model.
- `data.py` handles tokenization, padding, and `DataLoader` creation.
- `train.py` trains the model and saves `checkpoint.pt`.
- `download_data.py` downloads and unpacks the Kaggle dataset.
- `cli.py` loads the trained checkpoint and lets you type German sentences for translation.

## Requirements

- Python 3.10+
- PyTorch
- `tiktoken`
- `matplotlib`
- Kaggle account to download the English-German dataset yourself

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

## Data

This project uses the WMT 2014 English-German dataset from Kaggle:

- https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german

To download it:

1. Create a Kaggle account and accept the dataset terms on the dataset page.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

3. Download and unzip the dataset with the helper script:

```bash
python download_data.py
```

If you prefer the Kaggle CLI directly, you can run:

```bash
mkdir -p data
kaggle datasets download -d mohamedlotfy50/wmt-2014-english-german -p data --unzip
```

Each CSV should contain `de` and `en` columns for the source and target text.

## Train

Run training with:

```bash
python train.py
```

You can also tune the main hyperparameters from the command line:

```bash
python train.py --block-size 128 --n-embd 256 --n-layers 4 --n-heads 4 --dropout 0.2 --batch-size 32
```

Useful flags include `--max-iters`, `--eval-interval`, `--learning-rate`, `--eval-iters`, `--checkpoint-path`, and `--loss-plot-path`.

This will:

- train the model on the dataset
- save the model and optimizer state to `checkpoint.pt`
- write a loss curve to `figures/loss_curve.png`

## Interactive CLI

After training, start the translation prompt with:

```bash
python cli.py
```

You can point the CLI at a different checkpoint or change generation length:

```bash
python cli.py --checkpoint runs/baseline.pt --max-new-tokens 80
```

The CLI loads the checkpoint path you pass in, or `checkpoint.pt` by default, then opens a simple prompt:

```text
Type a German sentence (or 'quit'):
>> Ich liebe maschinelles Lernen.
EN: I love machine learning.
```

Type `quit` or `exit` to leave the session.

## Notes

- The CLI assumes the checkpoint was created with the same tokenizer and model hyperparameters used in `train.py`.
- Input text longer than the model block size is rejected by the CLI.
- Generation is greedy, so results are deterministic but not necessarily the most fluent output.

## References

- Vaswani et al., 2017, [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- WMT 2014 English-German dataset on Kaggle: https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german
- OpenAI `tiktoken` tokenizer: https://github.com/openai/tiktoken
- PyTorch documentation: https://pytorch.org/docs/stable/index.html

