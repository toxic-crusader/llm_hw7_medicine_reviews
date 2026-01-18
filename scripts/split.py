# File: scripts/split.py

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_train_val(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training dataset into train and validation using grouped split.

    Grouping is performed by review_norm to prevent identical or nearly
    identical reviews from appearing in both splits.

    Parameters
    ----------
    df:
        Cleaned training dataframe with review_norm column.
    val_size:
        Fraction of data to use for validation.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and validation dataframes.
    """
    if "review_norm" not in df.columns:
        raise ValueError("Column 'review_norm' is required for grouped split")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )

    groups = df["review_norm"]
    indices = next(splitter.split(df, groups=groups))
    train_idx, val_idx = indices

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df
