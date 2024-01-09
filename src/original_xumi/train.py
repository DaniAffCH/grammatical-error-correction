import random
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models import Seq2Seq
from apply_perturbations import apply_perturbation_to_text
from text import Text
import pandas as pd
import numpy as np
from perturbations import PERTURBATIONS, WEIGHTS
from text import Text

torch.set_printoptions(threshold=10_000)

MAX_LEN = 256


class Dataset(torch.utils.data.Dataset):
    def __init__(self, syntheticPath, jflegPath, tokenizer, deterministic=False):
        syntheticData = pd.read_csv(syntheticPath, delimiter="\t", names=[
                                    "input", "target", "isSynthetic"])
        syntheticData["isSynthetic"] = True
        jflegData = pd.read_csv(jflegPath)
        jflegData = self._preprocessJfleg()
        jflegData["isSynthetic"] = False

        self.data = pd.concat([jflegData, syntheticData], ignore_index=True)

        self.n_samples = len(self.jflegData)//2 + self.syntheticData
        self.tokenizer = tokenizer

        if deterministic:
            np.random.seed(42)

    def _preprocessJfleg(self):
        self.jflegData = self.jflegData.groupby(
            'input')['target'].agg(np.array).reset_index()
        self.jflegData["input"] = self.jflegData["input"].str.replace(
            r'^grammar: ', '', regex=True)

    def __len__(self):
        return self.n_samples

    def apply_perturbation_to_text(text: Text, freq: int = 4, skip_probability: float = 0.5) -> None:
        if np.random.uniform(0, 1) < skip_probability:
            return
        perturbations = np.random.choice(
            PERTURBATIONS, p=WEIGHTS, size=freq)
        for perturbation in perturbations:
            if perturbation.is_applicable(text):
                perturbation.perturb(text)

    def _getJfleg(self, row):
        target = np.random.choice(row["target"])
        text = Text(original=row["input"], transformed=target)
        return text

    def _getSynthetic(self, row):
        text = Text(original=row["input"])
        self.apply_perturbation_to_text(text)
        return text

    def __getitem__(self, index):
        row = self.data.iloc[index]

        text = self._getSynthetic(
            row) if row["isSynthetic"] else self._getJfleg(row)

        x = self.hf_tokenizer.encode(text.transformed).ids
        y = self.hf_tokenizer.encode(text.original).ids

        x = x[:MAX_LEN]
        y = y[:MAX_LEN]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


'''

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, path, hf_tokenizer):
        self.samples = samples
        self.n_samples = len(self.samples)
        self.hf_tokenizer = hf_tokenizer

    def __len__(self):
        return self.n_samples // 100  # Smaller epochs

    def __getitem__(self, _):
        idx = random.randint(0, self.n_samples - 1)

        text_str = self.samples[idx]
        text = Text(original=text_str)

        apply_perturbation_to_text(text)

        x = self.hf_tokenizer.encode(text.transformed).ids
        y = self.hf_tokenizer.encode(text.original).ids

        x = x[:MAX_LEN]
        y = y[:MAX_LEN]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, phase) -> None:
        super().__init__()
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self._preprocess()
        self.phase = phase

    def _preprocess(self):
        self.data = self.data.groupby(
            'input')['target'].agg(np.array).reset_index()
        self.data["input"] = self.data["input"].str.replace(
            r'^grammar: ', '', regex=True)

    def _process_sequence(self, sequence):
        sequence = f"{self.tokenizer.bos_token} {sequence} {self.tokenizer.eos_token}"
        # result = self.tokenizer(sequence, return_tensors="pt",
        #                       padding="max_length", truncation=True, max_length=self.max_len)

        # result = {key: value.squeeze() for key, value in result.items()}
        return sequence

    def __len__(self):
        return self.data.size//2

    def __getitem__(self, index):
        input = self.data.iloc[index]["input"]
        target_text_list = self.data.iloc[index]["target"]

        target_out = random.choice(
            target_text_list) if self.phase == "train" else target_text_list[0]
        target_out = target_out
        
        x = self.tokenizer.encode(input).ids
        y = self.tokenizer.encode(target_out).ids

        x = x[:MAX_LEN]
        y = y[:MAX_LEN]

        # print("input x :", self.hf_tokenizer.decode(x))
        # print("output y :", self.hf_tokenizer.decode(y))

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
'''


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
    import nltk
    nltk.download('punkt')
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=160)
    parser.add_argument("--epochs", default=2000)
    parser.add_argument(
        "--init_model_path",
        default=str(
            Path(__file__).absolute().parents[0] / "output" / "checker-v6.ckpt"
        ),
    )
    parser.add_argument(
        "--data_path", default=str(Path(__file__).absolute().parents[0] / "wikisent2.txt")
    )
    parser.add_argument(
        "--tokenizer_path",
        default=str(
            Path(__file__).absolute().parents[0] /
            "resources" / "tokenizer.json"
        ),
    )
    parser.add_argument(
        "--base_path", default=str(Path(__file__).absolute().parents[0] / "output")
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    data_path = args.data_path
    init_model_path = args.init_model_path
    tokenizer_path = args.tokenizer_path
    base_path = Path(args.base_path)
    base_path.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    with open(data_path) as f:
        data = f.read().split("\n")

    train, val = train_test_split(data, test_size=0.05, random_state=1337)

    train_data = Dataset(syntheticPath="wikisent_train.csv",
                         jflegPath="../dataset/train.csv", tokenizer=tokenizer)
    val_data = Dataset(syntheticPath="wikisent_eval.csv",
                       jflegPath="../dataset/eval.csv", tokenizer=tokenizer, deterministic=True)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(
            generate_batch, pad_idx=tokenizer.token_to_id("[PAD]")),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(
            generate_batch, pad_idx=tokenizer.token_to_id("[PAD]")),
    )

    model = Seq2Seq(
        tokenizer,
        out_vocab_size=tokenizer.get_vocab_size(),
        pad_idx=tokenizer.token_to_id("[PAD]"),
        lr=1e-6,
        dropout=0.1
    )

    # model.load_state_dict(torch.load(init_model_path)["state_dict"])

    logger = TensorBoardLogger(
        save_dir=str(base_path),
        name="logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc", mode="max", dirpath=base_path, filename="checker"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
