import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml_platform.exceptions import TrainingError
from ml_platform.training import Trainer


@pytest.fixture
def training_data():
    """Create a small dataset for training tests."""

    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
        }
    )

    y = pd.Series([2, 4, 6, 8, 10])

    return X, y


def test_train(training_data):
    """Trainer should successfully train a model."""

    X, y = training_data

    trainer = Trainer(LinearRegression())

    trained_model = trainer.train(
        X_train=X,
        y_train=y,
    )

    assert trained_model is trainer.model


def test_model_is_fitted(training_data):
    """Trainer should return a fitted model."""

    X, y = training_data

    trainer = Trainer(LinearRegression())

    trainer.train(
        X_train=X,
        y_train=y,
    )

    assert hasattr(trainer.model, "coef_")


def test_save(training_data, tmp_path):
    """Trainer should persist the trained model."""

    X, y = training_data

    trainer = Trainer(LinearRegression())

    trainer.train(
        X_train=X,
        y_train=y,
    )

    model_path = tmp_path / "model.pkl"

    trainer.save(model_path)

    assert model_path.exists()

    loaded_model = joblib.load(model_path)

    assert loaded_model is not None


class FailingModel:
    """Model used to simulate a training failure."""

    def fit(self, X, y):
        raise ValueError("Training failed")


def test_training_failure(training_data):
    """Trainer should raise TrainingError when training fails."""

    X, y = training_data

    trainer = Trainer(FailingModel())

    with pytest.raises(TrainingError, match="Training failed"):
        trainer.train(
            X_train=X,
            y_train=y,
        )
