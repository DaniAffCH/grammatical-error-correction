from enum import Enum
from transformers import GPT2Tokenizer
import torch


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


def masked_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, pad_idx):
    mask = y_true != pad_idx
    y_true = torch.masked_select(y_true, mask)
    y_pred = torch.masked_select(y_pred, mask)

    return (y_true == y_pred).double().mean()


def moveToDevice(val, device):
    val["input_ids"] = val["input_ids"].to(device)
    val["attention_mask"] = val["attention_mask"].to(device)
    return val
