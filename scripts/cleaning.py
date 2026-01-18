# File: scripts/cleaning.py

from __future__ import annotations

from typing import Dict

import pandas as pd


def clean_train_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean training dataset by removing conflicting and duplicate reviews.

    The function expects a column review_norm to be present.
    All rows where the same review_norm has multiple different ratings
    are fully removed.
    Then duplicate rows by review_norm and rating are dropped,
    keeping a single representative.

    Parameters
    ----------
    df:
        Training dataframe with review_norm column.

    Returns
    -------
    tuple[pd.DataFrame, Dict[str, int]]
        Cleaned dataframe and statistics about removed rows.
    """
    if "review_norm" not in df.columns:
        raise ValueError("Column 'review_norm' is required for cleaning")

    initial_rows = len(df)

    rating_per_review = df.groupby("review_norm", dropna=False)["rating"].nunique()

    conflicting_reviews = rating_per_review[rating_per_review > 1].index
    conflicts_removed = int(df["review_norm"].isin(conflicting_reviews).sum())

    df_clean = df[~df["review_norm"].isin(conflicting_reviews)].copy()

    rows_after_conflicts = len(df_clean)

    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["review_norm", "rating"])
    duplicates_removed = before_dedup - len(df_clean)

    final_rows = len(df_clean)

    stats = {
        "initial_rows": initial_rows,
        "conflicts_removed": conflicts_removed,
        "duplicates_removed": duplicates_removed,
        "final_rows": final_rows,
    }

    return df_clean, stats
