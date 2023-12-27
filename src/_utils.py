from enum import Enum
from transformers import GPT2Tokenizer


class SpecialToken(Enum):
    BEGIN = "[CLS]"
    END = "[SEP]"
    PADDING = "[PAD]"

    def __str__(self):
        return str(self.value)


def tokenizerSetup():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_tokens([str(t) for t in SpecialToken])
    tokenizer.padding_side = "left"

    tokenizer.pad_token = str(SpecialToken.PADDING)
    tokenizer.bos_token = str(SpecialToken.BEGIN)
    tokenizer.eos_token = str(SpecialToken.END)

    return tokenizer
