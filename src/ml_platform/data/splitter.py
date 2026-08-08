import pandas as pd
from sklearn.model_selection import train_test_split

from ml_platform.config import settings


class DataSplitter:
    """Split datasets into training, validation, and test sets."""

    def split(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""

        if random_state is None:
            random_state = settings.random_seed

        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
        )

        validation_ratio = validation_size

        train_data, validation_data = train_test_split(
            train_data,
            test_size=validation_ratio,
            random_state=random_state,
        )

        return train_data, validation_data, test_data