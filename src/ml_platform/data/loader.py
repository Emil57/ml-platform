from pathlib import Path

import pandas as pd


class DataLoader:
    """Load datasets from supported file formats."""

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Dataset path is not a file: {file_path}")

        return pd.read_csv(file_path)