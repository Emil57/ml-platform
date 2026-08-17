from unittest.mock import MagicMock

import pytest

from ml_platform.exceptions import ModelLoadError
from ml_platform.serving.loader import MLflowModelLoader
from ml_platform.serving.predictor import MLflowPredictor


def test_load_model() -> None:
    model = MagicMock()
    model.predict.return_value = [1.0, 2.0]

    load_model = MagicMock(return_value=model)

    loader = MLflowModelLoader(load_model=load_model)

    result = loader.load("models:/california-housing/3")

    assert isinstance(result, MLflowPredictor)
    assert result._model is model

    load_model.assert_called_once_with("models:/california-housing/3")


def test_load_model_failure() -> None:
    load_model = MagicMock(side_effect=Exception("MLflow unavailable"))

    loader = MLflowModelLoader(load_model=load_model)

    with pytest.raises(ModelLoadError, match="Failed to load model"):
        loader.load("models:/california-housing/3")


def test_load_model_without_predict() -> None:
    load_model = MagicMock(return_value=object())

    loader = MLflowModelLoader(load_model=load_model)

    with pytest.raises(
        ModelLoadError,
        match="does not implement predict",
    ):
        loader.load("models:/california-housing/3")
