from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np


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

    def __len__(self):
        return self.data.size//2

    def __getitem__(self, index):
        input = self.data.iloc[index]["input"]
        input = self._process_sequence(input)

        target_text_list = self.data.iloc[index]["target"]
        target_out = random.choice(target_text_list)
        target_out = self._process_sequence(target_out)

        return input, target_out

    def decode(self, embedding):
        return self.tokenizer.decode(embedding, skip_special_tokens=False)
