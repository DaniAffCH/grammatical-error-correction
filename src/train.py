from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from models import S2S

import tqdm

from jflegDataset import JflegDataset
import argparse

from _utils import tokenizerSetup, SpecialToken

from oneStep import trainOneStep


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = tokenizerSetup()

    train_data = JflegDataset(args.train_dataset, tokenizer)
    val_data = JflegDataset(args.eval_dataset, tokenizer)

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
        out_vocab_size=tokenizer.vocab_size + len(SpecialToken),
        dropout=0.1
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = StepLR(optimizer, args.epochs//10)

    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in tqdm.tqdm(range(args.epochs)):
        loss, acc = trainOneStep(model, train_loader, device,
                                 optimizer, criterion, tokenizer, lr_scheduler)

        print(f"#{epoch} {loss=} {acc=}")


def setParser(parser):
    abspath = Path(__file__).absolute().parents[0]

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--train_dataset", default=f"{abspath}/dataset/train.csv")
    parser.add_argument(
        "--eval_dataset", default=f"{abspath}/dataset/eval.csv")
    parser.add_argument(
        "--output_path", default=f"{abspath}/output/mod_ck.pt")
    parser.add_argument(
        "--embedding_size", type=int, default=768)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setParser(parser)

    args = parser.parse_args()

    train(args)
