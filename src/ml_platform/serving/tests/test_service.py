from unittest.mock import MagicMock

from ml_platform.data import loader
from ml_platform.serving.schemas import (
    ModelReference,
    PredictionRequest,
)
from ml_platform.serving.service import PredictionService


def test_prediction_service() -> None:
    resolver = MagicMock()
    loader = MagicMock()
    predictor = MagicMock()

    resolver.resolve.return_value = "models:/california-housing/3"
    loader.load.return_value = predictor
    predictor.predict.return_value = [123.4, 456.7]

    request = PredictionRequest(
        inputs=[{"x": 1}],
    )

    service = PredictionService(
        resolver=resolver,
        loader=loader,
        model_name="test-model",
        model_version="1",
    )

    response = service.predict(request)

    assert response.model_name == "test-model"
    assert response.model_version == "1"

    expected_reference = ModelReference(
        name="test-model",
        version="1",
    )

    resolver.resolve.assert_called_once_with(expected_reference)

    loader.load.assert_called_once_with("models:/california-housing/3")
    predictor.predict.assert_called_once_with(
        request.inputs,
    )
