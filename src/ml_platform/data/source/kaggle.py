from pathlib import Path

import kaggle

from ml_platform.data.sources.base import DataSource


class KaggleSource(DataSource):
    """Dataset source backed by Kaggle."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset

    def fetch(self, destination: Path) -> Path:
        """Download and extract a Kaggle dataset."""

        destination.mkdir(parents=True, exist_ok=True)

        kaggle.api.dataset_download_files(
            self.dataset,
            path=str(destination),
            unzip=True,
        )

        return destination