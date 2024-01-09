from models import S2S
import torch
from functools import partial
from _utils import tokenizerSetup, SpecialToken

from jflegDataset import JflegDataset, collate
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
        collate_fn=partial(
            collate, tokenizer=tokenizer)
    )

    batch = next(iter(train_loader))

    transformed_text = "For not use car ."

    src = tokenizer(transformed_text, return_tensors="pt", padding=True)

    trg = {
        "input_ids": torch.full((src["input_ids"].shape[0], 5), tokenizer.pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros((src["input_ids"].shape[0], 5))
    }

    trg["input_ids"][:, 0] = tokenizer.bos_token_id
    trg["attention_mask"][:, 0] = 1

    for i in range(1, 5):
        trg_in = dict()
        trg_in["input_ids"] = trg["input_ids"][:, :i]
        trg_in["attention_mask"] = trg["attention_mask"][:, :i]

        output = model(src, trg_in)
        output = output.argmax(2)

        is_end = output == tokenizer.eos_token_id
        is_end, _ = is_end.max(1)
        if is_end.sum() == output.shape[0]:
            break

        next_vals = output[:, -1]
        trg["input_ids"][:, i] = next_vals
        trg["attention_mask"][:, i] = 1

    trg = trg["input_ids"].squeeze().numpy()

    print(tokenizer.decode(trg))


if __name__ == "__main__":
    inference()
