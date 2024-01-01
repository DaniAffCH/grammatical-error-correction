import random
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models import Seq2Seq
from enum import Enum
from transformers import GPT2Tokenizer
import pandas as pd
import numpy as np


class SpecialToken(Enum):
    BEGIN = "[CLS]"
    END = "[SEP]"
    PADDING = "[PAD]"

    def __str__(self):
        return str(self.value)


def tokenizerSetup():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_tokens([str(t) for t in SpecialToken])
    tokenizer.padding_side = "right"

    tokenizer.pad_token = str(SpecialToken.PADDING)
    tokenizer.bos_token = str(SpecialToken.BEGIN)
    tokenizer.eos_token = str(SpecialToken.END)

    return tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_len=128) -> None:
        super().__init__()
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self._preprocess()
        self.max_len = max_len

    def _preprocess(self):
        self.data = self.data.groupby(
            'input')['target'].agg(np.array).reset_index()
        self.data["input"] = self.data["input"].str.replace(
            r'^grammar: ', '', regex=True)

    def _process_sequence(self, sequence):
        sequence = f"{self.tokenizer.bos_token} {sequence} {self.tokenizer.eos_token}"
        result = self.tokenizer(sequence, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=self.max_len)

        result = {key: value.squeeze() for key, value in result.items()}
        return result

    def __len__(self):
        return self.data.size//2

    def __getitem__(self, index):
        input = self.data.iloc[index]["input"]
        input = self._process_sequence(input)

        target_text_list = self.data.iloc[index]["target"]
        target_out = random.choice(target_text_list)
        target_out = self._process_sequence(target_out)

        return input, target_out

    def decode(self, embedding):
        return self.tokenizer.decode(embedding, skip_special_tokens=False)


def generate_batch(data_batch, pad_idx):
    src, trg = [], []
    for (src_item, trg_item) in data_batch:
        src.append(src_item)
        trg.append(trg_item)
    src = pad_sequence(src, padding_value=pad_idx, batch_first=True)
    trg = pad_sequence(trg, padding_value=pad_idx, batch_first=True)
    return src, trg


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--epochs", default=2000)

    parser.add_argument(
        "--base_path", default=str(Path(__file__).absolute().parents[1] / "output")
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    base_path = Path(args.base_path)
    base_path.mkdir(exist_ok=True)

    tokenizer = tokenizerSetup()

    train_data = Dataset(path="../src/dataset/train.csv", tokenizer=tokenizer)
    val_data = Dataset(path="../src/dataset/eval.csv", tokenizer=tokenizer)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,

    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,

    )

    model = Seq2Seq(
        tokenizer,
        out_vocab_size=tokenizer.vocab_size+len(SpecialToken),
        pad_idx=tokenizer("[PAD]")["input_ids"][0],
        lr=1e-6,
        dropout=0.1
    )

    trainer = pl.Trainer(
        max_epochs=epochs,

    )
    trainer.fit(model, train_loader, val_loader)
