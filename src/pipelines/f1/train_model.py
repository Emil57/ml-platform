from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import SGDClassifier

from ml_platform.config import settings
from ml_platform.training.trainer import Trainer
from pipelines.f1 import FEATURE_DATA_DIR, MODELS_DIR


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load featured training and validation datasets."""

    feature_data_dir = Path(FEATURE_DATA_DIR)

    train_path = feature_data_dir / "train.csv"
    valid_path = feature_data_dir / "valid.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not valid_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {valid_path}")

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

    # Use the shared training framework for model training.
    classifier_trainer = Trainer(classifier)

    classifier_trainer.train(
        X_train=X_train,
        y_train=y_train,
    )

    # Calibration is F1-specific logic, so it remains
    # inside the F1 pipeline rather than inside Trainer.
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(classifier),
        method="sigmoid",
    )

    calibrated_trainer = Trainer(calibrated)

    calibrated_trainer.train(
        X_train=X_valid,
        y_train=y_valid,
    )

    return calibrated


def main() -> None:
    """Train the F1 model and save it."""

    train, valid = load_training_data()

    model = train_model(train, valid)

    models_dir = Path(MODELS_DIR)
    model_path = models_dir / "model.pkl"

    # Use the shared Trainer for model persistence as well.
    trainer = Trainer(model)
    trainer.save(model_path)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
