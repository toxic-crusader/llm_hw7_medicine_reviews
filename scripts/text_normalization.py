# File: scripts/text_normalization.py

from __future__ import annotations

import html
import re

import pandas as pd

_whitespace_re = re.compile(r"\s+")


def normalize_review(text: str) -> str:
    """
    Normalize review text for deduplication and grouping purposes.

    The normalization is intentionally minimal and reversible.
    It removes superficial differences without changing semantics.

    Steps:
    HTML entities are unescaped.
    Newlines and tabs are converted to spaces.
    Multiple whitespace characters are collapsed.
    Leading and trailing whitespace is stripped.

    Parameters
    ----------
    text:
        Raw review text.

    Returns
    -------
    str
        Normalized review text.
    """
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = _whitespace_re.sub(" ", text)
    text = text.strip()

    return text


def add_review_norm_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized review column to dataframe.

    The original dataframe is not modified in place.
    A shallow copy is returned with an extra column review_norm.

    Parameters
    ----------
    df:
        Input dataframe with a review column.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional review_norm column.
    """
    if "review" not in df.columns:
        raise ValueError("Column 'review' is required to build review_norm")

    out = df.copy()
    out["review_norm"] = out["review"].map(normalize_review)

    return out
