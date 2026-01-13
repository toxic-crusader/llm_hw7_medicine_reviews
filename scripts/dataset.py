# File: scripts/dataset.py
from pathlib import Path
import shutil
import kagglehub
from IPython.display import Markdown, display


def md(text: str) -> None:
    display(Markdown(text))


def download_dataset(
    dataset_id: str,
    target_dir: Path,
    force: bool = False,
) -> None:
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
