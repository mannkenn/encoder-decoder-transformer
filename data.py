import torch
import csv
import tiktoken
from torch.utils.data import Dataset, DataLoader

class TokenizerConfig:
    def __init__(self, model_name="gpt-2"):
        # BPE from GPT-2
        self.enc = tiktoken.encoding_for_model(model_name)
        self.base_vocab_size = self.enc.n_vocab

        # custom special tokens
        self.PAD_ID = self.base_vocab_size
        self.BOS_ID = self.base_vocab_size + 1
        self.EOS_ID = self.base_vocab_size + 2

        self.vocab_size = self.base_vocab_size + 3

def read_pairs(data_path, src_col="de", tgt_col="en"):
    pairs = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row[src_col].strip()
            tgt = row[tgt_col].strip()
            if src and tgt:
                pairs.append((src, tgt))
    return pairs

def bpe_encode(data_path, tokenizer_cfg, src_col="de", tgt_col="en"):
    pairs = read_pairs(data_path, src_col=src_col, tgt_col=tgt_col)

    encoded_data = []
    for src_text, tgt_text in pairs:
        src_ids = tokenizer_cfg.enc.encode(src_text)
        tgt_ids = tokenizer_cfg.enc.encode(tgt_text)

        tgt_input_ids = [tokenizer_cfg.BOS_ID] + tgt_ids
        tgt_output_ids = tgt_ids + [tokenizer_cfg.EOS_ID]

        encoded_data.append((src_ids, tgt_input_ids, tgt_output_ids))

    return encoded_data

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pad_sequence(seq, max_len, pad_id):
    return seq + [pad_id] * (max_len - len(seq))


def make_collate_fn(pad_id):
    def collate_fn(batch):
        src_batch, tgt_in_batch, tgt_out_batch = zip(*batch)

        max_src_len = max(len(x) for x in src_batch)
        max_tgt_len = max(len(x) for x in tgt_in_batch)

        src_padded = [pad_sequence(x, max_src_len, pad_id) for x in src_batch]
        tgt_in_padded = [pad_sequence(x, max_tgt_len, pad_id) for x in tgt_in_batch]
        tgt_out_padded = [pad_sequence(x, max_tgt_len, pad_id) for x in tgt_out_batch]

        return (
            torch.tensor(src_padded, dtype=torch.long),
            torch.tensor(tgt_in_padded, dtype=torch.long),
            torch.tensor(tgt_out_padded, dtype=torch.long),
        )

    return collate_fn

def build_dataloader(data_path, tokenizer_cfg, batch_size=32, shuffle=True, src_col="de", tgt_col="en"):
    encoded_data = bpe_encode(
        data_path=data_path,
        tokenizer_cfg=tokenizer_cfg,
        src_col=src_col,
        tgt_col=tgt_col,
    )

    dataset = TranslationDataset(encoded_data)
    collate_fn = make_collate_fn(tokenizer_cfg.PAD_ID)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return loader