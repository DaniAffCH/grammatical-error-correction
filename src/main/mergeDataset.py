

import torch
from text import Text
import pandas as pd
import numpy as np
from perturbations import PERTURBATIONS, WEIGHTS
from text import Text
from _utils import SpecialToken


class MergeDataset(torch.utils.data.Dataset):
    def __init__(self, syntheticPath, jflegPath, tokenizer, deterministic=False, max_len=256):
        syntheticData = pd.read_csv(syntheticPath, delimiter="\t", names=[
                                    "input", "target", "isSynthetic"])
        syntheticData["isSynthetic"] = True
        jflegData = pd.read_csv(jflegPath)
        jflegData = self._preprocessJfleg(jflegData)
        jflegData["isSynthetic"] = False

        self.data = pd.concat([jflegData, syntheticData], ignore_index=True)

        self.n_samples = len(jflegData)//2 + len(syntheticData)
        self.tokenizer = tokenizer
        self.max_len = max_len
        if deterministic:
            np.random.seed(42)

    def _preprocessJfleg(self, data):
        data = data.groupby(
            'input')['target'].agg(np.array).reset_index()
        data["input"] = data["input"].str.replace(
            r'^grammar: ', '', regex=True)
        return data

    def __len__(self):
        return self.n_samples

    def apply_perturbation_to_text(self, text: Text, freq: int = 4, skip_probability: float = 0.5) -> None:
        if np.random.uniform(0, 1) < skip_probability:
            return
        perturbations = np.random.choice(
            PERTURBATIONS, p=WEIGHTS, size=freq)
        for perturbation in perturbations:
            if perturbation.is_applicable(text):
                perturbation.perturb(text)

    def _getJfleg(self, row):
        target = np.random.choice(row["target"])
        text = Text(original=row["input"], transformed=target)
        return text

    def _getSynthetic(self, row):
        text = Text(original=row["input"])
        self.apply_perturbation_to_text(text)
        return text

    def _processSample(self, sample):
        sample = self.tokenizer.encode(sample).ids
        return torch.tensor(sample[:self.max_len], dtype=torch.long)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        text = self._getSynthetic(
            row) if row["isSynthetic"] else self._getJfleg(row)

        x = self._processSample(text.transformed)
        y = self._processSample(text.original)

        return x, y
