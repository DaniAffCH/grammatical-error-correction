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

    def __len__(self):
        return self.data.size
    
    def __getitem__(self, index):
        input = self.tokenizer(self.data.iloc[index]["input"], return_tensors="pt")
        input["input_ids"] = input["input_ids"].squeeze()
        target = [self.tokenizer(s, return_tensors="pt", ) for s in self.data.iloc[index]["target"]]
        return input, target
    
    def decode(self, embedding):
        return self.tokenizer.decode(embedding)