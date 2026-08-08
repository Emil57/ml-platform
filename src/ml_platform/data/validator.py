import pandas as pd


class DataValidator:
    """Validate datasets before they enter an ML pipeline."""

    def validate_not_empty(self, data: pd.DataFrame) -> None:
        """Ensure the dataset contains at least one row."""
        if data.empty:
            raise ValueError("Dataset is empty.")

    def validate_columns(
        self,
        data: pd.DataFrame,
        required_columns: list[str],
    ) -> None:
        """Ensure all required columns are present."""
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

    def validate_no_missing_values(self, data: pd.DataFrame) -> None:
        """Ensure the dataset contains no missing values."""
        if data.isnull().values.any():
            raise ValueError("Dataset contains missing values.")