# File: scripts/train.py

from __future__ import annotations

import torch
from torch.nn import MSELoss
from tqdm.auto import tqdm


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = MSELoss()
    total_loss = 0.0

    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        preds = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = loss_fn(preds, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    loss_fn = MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(preds, batch["labels"])
            total_loss += loss.item()

    return total_loss / len(loader)
