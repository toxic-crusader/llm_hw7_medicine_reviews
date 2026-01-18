# File: scripts/model.py

from __future__ import annotations

from torch import nn
from transformers import AutoModel


class DistilBertRegressor(nn.Module):
    """
    DistilBERT based regressor for rating prediction.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.last_hidden_state[:, 0]
        value = self.regressor(pooled)
        return value.squeeze(-1)
