# File: scripts/inspection.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from IPython.display import Markdown, display


def md(text: str) -> None:
    """Render a markdown message in a Jupyter environment."""
    display(Markdown(text))


def _styled_head(df: pd.DataFrame, n: int = 5) -> None:
    """Display a wrapped head of the dataframe for long text columns."""
    head_df = df.head(n)
    try:
        styled = head_df.style.set_properties(**{"white space": "pre wrap"})
        display(styled)
    except Exception:
        display(head_df)


def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table for missing values and dtypes."""
    total_rows = len(df)
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": missing.astype(int),
            "missing_share": (missing / max(total_rows, 1)).astype(float),
            "nunique": df.nunique(dropna=True).astype(int),
        }
    ).sort_values(by="missing_count", ascending=False)


def _duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duplicate statistics for several relevant column subsets."""
    stats = []

    def add_row(name: str, subset: Optional[list[str]]) -> None:
        if subset is None:
            dup = df.duplicated().sum()
            subset_name = "all_columns"
        else:
            subset = [c for c in subset if c in df.columns]
            if not subset:
                return
            dup = df.duplicated(subset=subset).sum()
            subset_name = ", ".join(subset)

        stats.append(
            {
                "key": name,
                "subset": subset_name,
                "duplicate_rows": int(dup),
                "duplicate_share": float(dup / max(len(df), 1)),
            }
        )

    add_row("full_row_duplicates", None)
    add_row("review_duplicates", ["review"])
    add_row("review_rating_duplicates", ["review", "rating"])
    add_row("drug_condition_review_duplicates", ["drugName", "condition", "review"])

    return pd.DataFrame(stats)


def _conflicting_ratings_for_same_review(df: pd.DataFrame) -> pd.DataFrame:
    """Find reviews that appear with multiple distinct ratings."""
    if "review" not in df.columns or "rating" not in df.columns:
        return pd.DataFrame()

    tmp = df[["review", "rating"]].copy()
    tmp["rating"] = pd.to_numeric(tmp["rating"], errors="coerce")

    conflicts = (
        tmp.groupby("review", dropna=False)["rating"]
        .nunique(dropna=True)
        .reset_index(name="distinct_ratings")
    )

    return conflicts[conflicts["distinct_ratings"] > 1].sort_values(
        by="distinct_ratings", ascending=False
    )


def _date_report(df: pd.DataFrame) -> dict:
    """Parse date column and report invalid values and date range."""
    if "date" not in df.columns:
        return {"has_date": False}

    md("Parsing date column using non strict pandas parser")

    parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")

    invalid_mask = parsed.isna() & df["date"].notna()
    invalid_count = int(invalid_mask.sum())

    result = {
        "has_date": True,
        "invalid_count": invalid_count,
        "invalid_share": float(invalid_count / max(len(df), 1)),
        "min_date": None,
        "max_date": None,
        "invalid_examples": [],
    }

    valid = parsed.dropna()
    if not valid.empty:
        result["min_date"] = valid.min()
        result["max_date"] = valid.max()

    if invalid_count > 0:
        bad_values = df.loc[invalid_mask, "date"].astype(str).value_counts().head(10)
        result["invalid_examples"] = [
            (idx, int(val)) for idx, val in bad_values.items()
        ]

    return result


def _rating_report(df: pd.DataFrame) -> dict:
    """Report basic statistics for rating column."""
    if "rating" not in df.columns:
        return {"has_rating": False}

    rating = pd.to_numeric(df["rating"], errors="coerce")
    missing = int(rating.isna().sum())
    valid = rating.dropna()

    out = {
        "has_rating": True,
        "missing_count": missing,
        "missing_share": float(missing / max(len(df), 1)),
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "value_counts": None,
    }

    if not valid.empty:
        out["min"] = float(valid.min())
        out["max"] = float(valid.max())
        out["mean"] = float(valid.mean())
        out["median"] = float(valid.median())
        out["value_counts"] = valid.astype(int).value_counts().sort_index()

    return out


@dataclass
class TextLengthStats:
    has_review: bool
    rows: int
    empty_like_count: int
    empty_like_share: float
    char_quantiles: Optional[pd.Series]
    word_quantiles: Optional[pd.Series]
    long_words_512_count: int
    long_words_512_share: float
    long_words_1024_count: int
    long_words_1024_share: float


def _text_length_report(df: pd.DataFrame) -> TextLengthStats:
    """Compute text length statistics for review column."""
    if "review" not in df.columns:
        return TextLengthStats(
            has_review=False,
            rows=len(df),
            empty_like_count=0,
            empty_like_share=0.0,
            char_quantiles=None,
            word_quantiles=None,
            long_words_512_count=0,
            long_words_512_share=0.0,
            long_words_1024_count=0,
            long_words_1024_share=0.0,
        )

    review = df["review"].astype(str).str.strip()
    empty_like = review.eq("") | review.str.lower().eq("nan")

    char_len = review.str.len()
    word_len = review.apply(lambda s: len(s.split()))

    rows = len(df)
    empty_count = int(empty_like.sum())

    return TextLengthStats(
        has_review=True,
        rows=rows,
        empty_like_count=empty_count,
        empty_like_share=float(empty_count / max(rows, 1)),
        char_quantiles=char_len.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        word_quantiles=word_len.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]),
        long_words_512_count=int((word_len > 512).sum()),
        long_words_512_share=float((word_len > 512).mean()),
        long_words_1024_count=int((word_len > 1024).sum()),
        long_words_1024_share=float((word_len > 1024).mean()),
    )


def inspect_split(df: pd.DataFrame, name: str) -> None:
    """Run non destructive inspection for a dataset split."""
    md(f"## {name} inspection")
    md(f"Shape: {df.shape}")

    md("### Head")
    _styled_head(df)

    md("### Missing values and dtypes")
    display(_missing_summary(df))

    md("### Duplicates")
    display(_duplicate_summary(df))

    conflicts = _conflicting_ratings_for_same_review(df)
    md(f"Conflicting ratings for same review: {len(conflicts)}")
    if not conflicts.empty:
        display(conflicts.head())

    date_info = _date_report(df)
    if date_info.get("has_date"):
        md("### Dates")
        md(
            f"Invalid dates: {date_info['invalid_count']} "
            f"({date_info['invalid_share']:.6f})"
        )
        if date_info["min_date"] is not None:
            md(f"Date range: {date_info['min_date']} to {date_info['max_date']}")
        if date_info["invalid_examples"]:
            display(
                pd.DataFrame(
                    date_info["invalid_examples"],
                    columns=["value", "count"],
                )
            )

    rating_info = _rating_report(df)
    if rating_info.get("has_rating"):
        md("### Rating")
        md(
            f"min {rating_info['min']}, "
            f"max {rating_info['max']}, "
            f"mean {rating_info['mean']}, "
            f"median {rating_info['median']}, "
            f"missing {rating_info['missing_count']}"
        )
        if rating_info["value_counts"] is not None:
            display(rating_info["value_counts"].to_frame("count"))

    text_stats = _text_length_report(df)
    if text_stats.has_review:
        md("### Review text length")
        md(
            f"Empty like reviews: {text_stats.empty_like_count} "
            f"({text_stats.empty_like_share:.6f})"
        )
        display(text_stats.char_quantiles.to_frame("chars"))
        display(text_stats.word_quantiles.to_frame("words"))
        md(
            f"Over 512 words: {text_stats.long_words_512_count} "
            f"({text_stats.long_words_512_share:.6f}), "
            f"over 1024 words: {text_stats.long_words_1024_count} "
            f"({text_stats.long_words_1024_share:.6f})"
        )


def run_inspection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Run inspection for train and test splits."""
    inspect_split(train_df, "Train")
    inspect_split(test_df, "Test")
