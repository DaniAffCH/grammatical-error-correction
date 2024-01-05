from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import S2S

import pytorch_lightning as pl

from jflegDataset import JflegDataset
import argparse

from _utils import tokenizerSetup, SpecialToken


def train(args):
    tokenizer = tokenizerSetup()

    train_data = JflegDataset(args.train_dataset, tokenizer, "train")
    val_data = JflegDataset(args.eval_dataset, tokenizer, "val")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
    )

    model = S2S(
        tokenizer,
        out_vocab_size=tokenizer.vocab_size+len(SpecialToken),
        pad_idx=tokenizer("[PAD]")["input_ids"][0],
        lr=1e-6,
        dropout=0.1
    )

    earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=50)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.output_path,
        callbacks=[earlyStop],
        log_every_n_steps=5
    )
    if args.resume_checkpoint:
        trainer.fit(model, train_loader, val_loader,
                    ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)


def setParser(parser):
    abspath = Path(__file__).absolute().parents[0]

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--train_dataset", default=f"{abspath}/dataset/train.csv")
    parser.add_argument(
        "--eval_dataset", default=f"{abspath}/dataset/eval.csv")
    parser.add_argument(
        "--output_path", default=f"{abspath}/output")
    parser.add_argument(
        "--embedding_size", type=int, default=768)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setParser(parser)

    args = parser.parse_args()

    train(args)
