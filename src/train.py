import torch
import argparse
import tqdm
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, NLLLoss
from jflegDataset import JflegDataset
from _utils import tokenizerSetup, SpecialToken
from model import S2S
from trainOneStep import trainOneStep

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train(args):
    tokenizer = tokenizerSetup()
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_train = JflegDataset(args.train_dataset, tokenizer)
    ds_eval = JflegDataset(args.eval_dataset, tokenizer)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size)
    dl_eval = DataLoader(ds_eval, batch_size=args.batch_size)

    model = S2S(tokenizer.vocab_size + len(SpecialToken),
                args.embedding_size, device).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = StepLR(optimizer, args.epochs//10)

    criterion = NLLLoss(ignore_index=tokenizer.pad_token_id)

    logger.log(logging.INFO, f"Training summary:\n{'-'*50}\
               \nDevice: {device}\
               \nOptimizer: {optimizer.__class__.__name__}\
               \nCriterion: {criterion.__class__.__name__}\
               \nLearning Rate: {args.learning_rate}\
               \nBatch Size: {args.batch_size}\
               \nEpochs: {args.epochs}\
               \nEmbedding Size: {args.embedding_size}\
               \nPath trained model: {args.output_path}\
               \n{'-'*50}\n")

    for epoch in tqdm.tqdm(range(args.epochs)):
        l = trainOneStep(model, dl_train, optimizer,
                         lr_scheduler, criterion, device)

        logger.log(logging.INFO, f"Epoch:{epoch} - Avg loss:{l}")


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
        "--learning_rate", type=float, default=1e-3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setParser(parser)

    args = parser.parse_args()

    train(args)
