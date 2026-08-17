from unittest.mock import MagicMock

import pytest

from ml_platform.exceptions import PredictionError
from ml_platform.serving.predictor import MLflowPredictor


def test_predict() -> None:
    model = MagicMock()

    expected = [123.4, 456.7]
    model.predict.return_value = expected

    predictor = MLflowPredictor(model)

    inputs = [
        {"feature_a": 1.0, "feature_b": 2.0},
        {"feature_a": 3.0, "feature_b": 4.0},
    ]

    result = predictor.predict(inputs)

    assert result == expected

    model.predict.assert_called_once_with(inputs)


def test_prediction_failure() -> None:
    model = MagicMock()

    model.predict.side_effect = Exception("Prediction failed")

    predictor = MLflowPredictor(model)

    with pytest.raises(PredictionError, match="Model prediction failed"):
        predictor.predict([{"feature_a": 1.0}])
