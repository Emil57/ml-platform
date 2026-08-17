from unittest.mock import MagicMock

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
        model_name="california-housing",
        model_version="3",
        inputs=[
            {"feature_a": 1.0},
            {"feature_a": 2.0},
        ],
    )

    service = PredictionService(
        resolver=resolver,
        loader=loader,
    )

    response = service.predict(request)

    assert response.model_name == "california-housing"
    assert response.model_version == "3"
    assert response.predictions == [123.4, 456.7]
    assert response.request_id

    expected_reference = ModelReference(
        name="california-housing",
        version="3",
    )

    resolver.resolve.assert_called_once_with(expected_reference)

    loader.load.asserast_called_once_with("models:/california-housing/3")

    predictor.predict.assert_called_once_with(
        request.inputs,
    )
