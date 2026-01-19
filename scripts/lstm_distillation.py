# File: scripts/lstm_distillation.py

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DistillationDataset(Dataset):
    """
    Dataset for knowledge distillation.

    Stores tokenized text sequences and teacher predictions.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_targets: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.teacher_targets = teacher_targets

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "target": self.teacher_targets[index],
        }


class LSTMRegressor(nn.Module):
    """
    LSTM based regressor used as student model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)

        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        output = hidden[-1]
        prediction = self.regressor(output).squeeze(1)
        return prediction
