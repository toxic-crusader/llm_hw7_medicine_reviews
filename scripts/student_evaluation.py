# File: scripts/student_evaluation.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_student(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a regression student model on a dataloader with real targets.

    The function supports different batch schemas.
    If target_key is provided, it will be used.
    Otherwise it will try common keys in order: "labels", then "target".

    Parameters
    ----------
    model : torch.nn.Module
        Trained student model.
    dataloader : DataLoader
        Dataloader that yields batches with "input_ids", "attention_mask" and a target field.
    device : torch.device
        Computation device.
    target_key : Optional[str]
        Explicit target key in batch dict. If None, common keys will be auto detected.

    Returns
    -------
    Dict[str, float]
        Dictionary with mse, rmse and mae.
    """
    model.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for batch in dataloader:
        preds = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        predictions.append(preds.detach().cpu())

        if target_key is not None:
            y = batch[target_key]
        else:
            if "labels" in batch:
                y = batch["labels"]
            elif "target" in batch:
                y = batch["target"]
            else:
                raise KeyError(
                    'Batch has no target key. Expected "labels" or "target", or pass target_key explicitly.'
                )

        targets.append(y.detach().cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = torch.cat(targets).numpy()

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    return {"mse": mse, "rmse": rmse, "mae": mae}
