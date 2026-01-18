# File: scripts/evaluation.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def collect_predictions(
    model,
    loader,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect model predictions and true targets from a dataloader.

    Returns arrays of predictions and targets on CPU.
    """
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            preds.append(outputs.cpu().numpy())
            targets.append(batch["labels"].cpu().numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
    )


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def absolute_error_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute summary statistics of absolute prediction error.
    """
    abs_err = np.abs(y_true - y_pred)

    return {
        "p50": float(np.percentile(abs_err, 50)),
        "p75": float(np.percentile(abs_err, 75)),
        "p90": float(np.percentile(abs_err, 90)),
        "p95": float(np.percentile(abs_err, 95)),
        "mean": float(abs_err.mean()),
        "max": float(abs_err.max()),
    }
