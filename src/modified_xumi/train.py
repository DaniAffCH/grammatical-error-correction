from functools import partial
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


from mergeDataset import MergeDataset

from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from mod_models import S2S

import pytorch_lightning as pl

import argparse

from functools import partial

import os

from _utils import collate

import wandb


def train(args):
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    # TODO: no hardcode
    train_data = MergeDataset(syntheticPath="dataset/wikisent_train.csv",
                              jflegPath="dataset/train.csv", tokenizer=tokenizer)
    val_data = MergeDataset(syntheticPath="dataset/wikisent_eval.csv",
                            jflegPath="dataset/eval.csv", tokenizer=tokenizer, deterministic=True)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(
            collate, tokenizer=tokenizer)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(
            collate, tokenizer=tokenizer)
    )

    model = S2S(tokenizer,
                tokenizer.get_vocab_size(),
                lr=1e-6,
                dropout=0.1)

    earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_path, "checkpoints"),
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    wandb_logger = pl.loggers.WandbLogger(
        project=args.wandb_name, log_model=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.output_path,
        callbacks=[earlyStop, checkpoint_callback],
        log_every_n_steps=5,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger
    )

    if args.resume_checkpoint:
        trainer.fit(model, train_loader, val_loader,
                    ckpt_path=args.resume_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)


def setParser(parser):
    abspath = Path(__file__).absolute().parents[0]

    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--train_jfleg_dataset", default=f"{abspath}/dataset/train.csv")
    parser.add_argument(
        "--eval_jfleg_dataset", default=f"{abspath}/dataset/eval.csv")
    parser.add_argument(
        "--train_synthetic_dataset", default=f"{abspath}/dataset/wikisent_train.csv")
    parser.add_argument(
        "--eval_synthetic_dataset", default=f"{abspath}/dataset/wikisent_eval.csv")
    parser.add_argument(
        "--output_path", default=f"{abspath}/output")
    parser.add_argument(
        "--embedding_size", type=int, default=768)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6)
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None)
    parser.add_argument(
        "--wandb_name", type=str, default="grammatical-error-correction")
    parser.add_argument(
        "--tokenizer_path", type=str, default=f"{abspath}/resources/tokenizer.json")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setParser(parser)

    args = parser.parse_args()

    train(args)
