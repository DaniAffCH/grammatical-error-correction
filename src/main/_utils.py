from enum import Enum
from transformers import GPT2Tokenizer

from torch.nn.utils.rnn import pad_sequence


class SpecialToken(Enum):
    BEGIN = "[CLS]"
    END = "[SEP]"
    PADDING = "[PAD]"

    def __str__(self):
        return str(self.value)


def collate(data_batch, tokenizer):
    pad_idx = tokenizer.token_to_id(str(SpecialToken.PADDING))
    src, trg = [], []
    for (src_item, trg_item) in data_batch:
        src.append(src_item)
        trg.append(trg_item)
    src = pad_sequence(src, padding_value=pad_idx, batch_first=True)
    trg = pad_sequence(trg, padding_value=pad_idx, batch_first=True)
    return src, trg
