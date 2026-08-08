import pandas as pd

from ml_platform.data.splitter import DataSplitter


def test_split():
    data = pd.DataFrame(
        {
            "feature": range(100),
            "target": range(100),
        }
    )

    splitter = DataSplitter()

    train, validation, test = splitter.split(data)

    assert len(train) == 64
    assert len(validation) == 16
    assert len(test) == 20


def test_split_preserves_all_rows():
    data = pd.DataFrame(
        {
            "feature": range(100),
            "target": range(100),
        }
    )

    splitter = DataSplitter()

    train, validation, test = splitter.split(data)

    assert len(train) + len(validation) + len(test) == len(data)


def test_split_is_reproducible():
    data = pd.DataFrame(
        {
            "feature": range(100),
            "target": range(100),
        }
    )

    splitter = DataSplitter()

    train_1, validation_1, test_1 = splitter.split(data)
    train_2, validation_2, test_2 = splitter.split(data)

    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(validation_1, validation_2)
    pd.testing.assert_frame_equal(test_1, test_2)
