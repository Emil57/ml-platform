from pathlib import Path

import joblib
import pandas as pd

from ml_platform.exceptions import TrainingError
from ml_platform.utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """Reusable framework for training and persisting ML models."""

    def __init__(self, model):
        self.model = model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """Train the configured model."""

        logger.info(
            "Starting model training: %s",
            type(self.model).__name__,
        )

        try:
            self.model.fit(X_train, y_train)
        except Exception as exc:
            logger.exception(
                "Model training failed: %s",
                type(self.model).__name__,
            )
            raise TrainingError(
                f"Training failed for {type(self.model).__name__}: {exc}"
            ) from exc

        logger.info(
            "Model training completed: %s",
            type(self.model).__name__,
        )

        return self.model

    def save(self, output_path: Path) -> None:
        """Persist the trained model to disk."""

        logger.info("Saving model to %s", output_path)

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            joblib.dump(self.model, output_path)

        except Exception as exc:
            logger.exception(
                "Failed to save model to %s",
                output_path,
            )
            raise TrainingError(
                f"Failed to save model to {output_path}: {exc}"
            ) from exc

        logger.info(
            "Model saved successfully to %s",
            output_path,
        )
