from ml_platform.serving.schemas import (
    ModelReference,
    PredictionRequest,
    PredictionResponse,
)


def test_prediction_request_creation() -> None:
    request = PredictionRequest(
        model_name="california-housing",
        model_version="1",
        inputs=[
            {
                "MedInc": 8.3,
                "HouseAge": 41.0,
            }
        ],
    )

    assert request.model_name == "california-housing"
    assert request.model_version == "1"
    assert len(request.inputs) == 1


def test_prediction_request_without_version() -> None:
    request = PredictionRequest(
        model_name="california-housing",
        inputs=[],
    )

    assert request.model_version is None


def test_model_reference() -> None:
    reference = ModelReference(
        name="california-housing",
        version="1",
    )

    assert reference.name == "california-housing"
    assert reference.version == "1"
    assert reference.alias is None


def test_prediction_response() -> None:
    response = PredictionResponse(
        model_name="california-housing",
        model_version="1",
        predictions=[2.5, 3.1],
        request_id="test-request",
    )

    assert response.predictions == [2.5, 3.1]