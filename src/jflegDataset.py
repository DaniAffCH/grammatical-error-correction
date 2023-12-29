from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random


class JflegDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128) -> None:
        super().__init__()
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self._preprocess()
        self.max_len = max_len

    def _preprocess(self):
        self.data = self.data.groupby(
            'input')['target'].agg(np.array).reset_index()
        self.data["input"] = self.data["input"].str.replace(
            r'^grammar: ', '', regex=True)

    def _process_sequence(self, sequence):
        sequence = f"{self.tokenizer.bos_token} {sequence} {self.tokenizer.eos_token}"
        result = self.tokenizer(sequence, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=self.max_len)
        result = {key: value.squeeze() for key, value in result.items()}
        return result

    def _right_shift(self, original_tensor: torch.Tensor, shift, filling_value) -> torch.Tensor:
        head = torch.full((shift,), filling_value)
        return torch.cat((head, original_tensor[:-shift]))

    def __len__(self):
        return self.data.size//2

    def __getitem__(self, index):
        input = self.data.iloc[index]["input"]
        input = self._process_sequence(input)

        target_text_list = self.data.iloc[index]["target"]
        target_out = random.choice(target_text_list)
        target_out = self._process_sequence(target_out)

        bos_token_index = torch.where(
            target_out["input_ids"] == self.tokenizer.bos_token_id)[0]

        random_shift = random.randint(1, self.max_len - bos_token_index - 1)

        target_in = {
            "input_ids": self._right_shift(target_out["input_ids"].clone(), random_shift,
                                           self.tokenizer.pad_token_id),
            "attention_mask": self._right_shift(
                target_out["attention_mask"].clone(), random_shift, 0.)
        }

        return input, target_in, target_out

    def decode(self, embedding):
        return self.tokenizer.decode(embedding, skip_special_tokens=False)
