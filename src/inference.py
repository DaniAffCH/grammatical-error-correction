from models import S2S
import torch
from _utils import tokenizerSetup, SpecialToken

from jflegDataset import JflegDataset
from torch.utils.data import DataLoader


@torch.no_grad()
def inference():
    tokenizer = tokenizerSetup()

    model = S2S.load_from_checkpoint(
        'output/best_model.ckpt', tok=tokenizer, out_vocab_size=tokenizer.vocab_size + len(SpecialToken), pad_idx=tokenizer("[PAD]")[
            "input_ids"][0])
    model.eval()
    train_data = JflegDataset("dataset/train.csv", tokenizer, "train")

    train_loader = DataLoader(
        train_data,
        batch_size=1,
        num_workers=10,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    y_hat, _ = model.feedforward(batch, 0)
    y_hat = y_hat.squeeze()

    y_hat = y_hat.argmax(-1)
    print(tokenizer.decode(y_hat, skip_special_tokens=False))
    exit(0)

    max_length = 128

    sequence = batch[0]["input_ids"]
    print(sequence)
    sequence = f"{tokenizer.bos_token} {sequence} {tokenizer.eos_token}"
    src = tokenizer(sequence, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=max_length)

    trg_str = tokenizer.bos_token
    trg = tokenizer(trg_str, return_tensors="pt",
                    padding="max_length", truncation=True, max_length=max_length)

    for i in range(max_length):
        out = model(src, trg)
        out = out.argmax(2).squeeze()
        # idk maybe of 0?
        out_str = tokenizer.decode(out[i], skip_special_tokens=False)
        print("PREDICTED:")
        print(out_str)
        print()

        is_end = out == tokenizer.eos_token_id
        is_end, _ = is_end.max(-1)
        if is_end.sum() == out.shape[0]:
            break

        trg_str = f"{trg_str}{out_str}"
        trg = tokenizer(trg_str, return_tensors="pt",
                        padding="max_length", truncation=True, max_length=max_length)
        print("TARGET:")
        print(trg_str)
        print()


if __name__ == "__main__":
    inference()
