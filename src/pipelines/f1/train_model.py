from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import SGDClassifier

from ml_platform.artifacts import ArtifactManager
from ml_platform.config import settings
from ml_platform.training import Trainer
from pipelines.f1 import FEATURE_DATA_DIR


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load featured training and validation datasets."""

    feature_data_dir = Path(FEATURE_DATA_DIR)

    train_path = feature_data_dir / "train.csv"
    valid_path = feature_data_dir / "valid.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not valid_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {valid_path}"
        )

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    return train, valid


def train_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> CalibratedClassifierCV:
    """Train and calibrate the F1 classification model."""

    target_column = "won"

    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]

    X_valid = valid.drop(columns=[target_column])
    y_valid = valid[target_column]

    classifier = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        class_weight={0: 1.0, 1: 20.0},
        random_state=settings.random_seed,
    )

    # Train the base classifier using the shared training framework.
    classifier_trainer = Trainer(classifier)

    classifier_trainer.train(
        X_train,
        y_train,
    )

    # Calibration is F1-specific logic and therefore remains
    # inside the F1 pipeline.
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(classifier),
        method="sigmoid",
    )

    calibration_trainer = Trainer(calibrated)

    calibration_trainer.train(
        X_valid,
        y_valid,
    )

    return calibrated


def main() -> None:
    """Train the F1 model and save the final artifact."""

    train, valid = load_training_data()

    model = train_model(
        train,
        valid,
    )

    artifact_manager = ArtifactManager("f1")

    model_path = artifact_manager.save_model(model)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()