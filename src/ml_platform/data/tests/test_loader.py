from tkinter import _test

import pandas as pd
import pytest
from ml_platform.data import DataLoader


def test_load_csv(tmp_path):
    dataset = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [10, 20, 30],
        }
    )

    file_path = tmp_path / "data.csv"
    dataset.to_csv(file_path, index=False)

    loader = DataLoader()

    result = loader.load_csv(file_path)

    pd.testing.assert_frame_equal(result, dataset)

def test_load_csv_file_not_found():
    loader = DataLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_csv("does_not_exist.csv")