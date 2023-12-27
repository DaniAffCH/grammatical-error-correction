from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class JflegDataset(Dataset):
    def __init__(self, path, tokenizer) -> None:
        super().__init__()
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self._preprocess()

    def _preprocess(self):
        self.data = self.data.groupby('input')['target'].agg(np.array).reset_index()
        self.data["input"] = self.data["input"].str.replace(r'^grammar: ', '', regex=True) 

    def _process_sequence(self, sequence):
        result = self.tokenizer(sequence, return_tensors="pt", padding=True)
        result = {key: value.squeeze() for key, value in result.items()}
        return result

    def __len__(self):
        return self.data.size
    
    def __getitem__(self, index):
        input = self.data.iloc[index]["input"]
        input = self._process_sequence(input)

        target_text_list = self.data.iloc[index]["target"]
        target = [self._process_sequence(s) for s in target_text_list]

        return input, target
    
    def decode(self, embedding):
        return self.tokenizer.decode(embedding)