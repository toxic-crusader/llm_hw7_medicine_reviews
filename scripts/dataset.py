# File: scripts/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import shutil
import pandas as pd
import kagglehub
from IPython.display import Markdown, display


REQUIRED_COLUMNS = {"review", "rating"}


def md(text: str) -> None:
    """
    Render a markdown message in a Jupyter environment.

    Parameters
    ----------
    text:
        Markdown formatted text to display.
    """
    display(Markdown(text))


def download_dataset(
    dataset_id: str,
    target_dir: Path,
    force: bool = False,
) -> None:
    """
    Download a Kaggle dataset and copy its contents into a target directory.

    The function checks whether the target directory already contains files.
    If it does and force is False, the download step is skipped.
    All dataset files are copied from the Kaggle cache into the target directory.

    Parameters
    ----------
    dataset_id:
        Kaggle dataset identifier.
    target_dir:
        Directory where raw dataset files will be stored.
    force:
        Whether to force re download even if target directory is not empty.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.iterdir()) and not force:
        md("Dataset already exists. Skipping download.")
        return

    md("Downloading dataset from Kaggle")

    downloaded_path = Path(kagglehub.dataset_download(dataset_id))

    for item in downloaded_path.iterdir():
        destination = target_dir / item.name

        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)

    md(f"Dataset saved to `{target_dir}`")


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load a raw CSV dataset into a pandas DataFrame without filtering columns.

    The function performs minimal validation and type normalization.
    No rows or columns are dropped at this stage to preserve full observability
    of potential data issues.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns preserved.
    """
    md(f"Loading dataset from `{csv_path}`")

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    md(f"Loaded dataframe with {df.shape[0]} rows and {df.shape[1]} columns")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["review"] = df["review"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    missing_ratings = int(df["rating"].isna().sum())
    md(f"Ratings converted to numeric. Missing values after conversion: {missing_ratings}")

    return df


def load_raw_splits(
    train_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets using identical rules.

    Both splits are loaded without dropping columns or rows.
    This ensures consistent inspection and downstream cleaning decisions.

    Parameters
    ----------
    train_path:
        Path to the training CSV file.
    test_path:
        Path to the test CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and test datasets as pandas DataFrames.
    """
    md("Loading training dataset")
    train_df = load_raw_dataset(train_path)

    md("Loading test dataset")
    test_df = load_raw_dataset(test_path)

    md("Finished loading train and test datasets")
    return train_df, test_df
