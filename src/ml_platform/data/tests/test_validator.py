import pandas as pd
import pytest

from ml_platform.data.validator import DataValidator


def test_validate_not_empty():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    validator = DataValidator()

    validator.validate_not_empty(data)


def test_validate_not_empty_fails_for_empty_dataset():
    data = pd.DataFrame()

    validator = DataValidator()

    with pytest.raises(ValueError, match="Dataset is empty"):
        validator.validate_not_empty(data)


def test_validate_columns():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [10, 20, 30],
        }
    )

    validator = DataValidator()

    validator.validate_columns(
        data,
        ["feature", "target"],
    )


def test_validate_columns_fails_when_column_is_missing():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    validator = DataValidator()

    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate_columns(
            data,
            ["feature", "target"],
        )


def test_validate_no_missing_values():
    data = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [10, 20, 30],
        }
    )

    validator = DataValidator()

    validator.validate_no_missing_values(data)


def test_validate_no_missing_values_fails_with_missing_values():
    data = pd.DataFrame(
        {
            "feature": [1, None, 3],
            "target": [10, 20, 30],
        }
    )

    validator = DataValidator()

    with pytest.raises(ValueError, match="Dataset contains missing values"):
        validator.validate_no_missing_values(data)