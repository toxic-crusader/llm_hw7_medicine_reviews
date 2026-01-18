# File: scripts/text_dataset.py

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset


class ReviewRegressionDataset(Dataset):
    """
    PyTorch dataset for text regression using transformer tokenization.
    """

    def __init__(self, texts, targets, tokenizer, max_length: int = 256):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item
