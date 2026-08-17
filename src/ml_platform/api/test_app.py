from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from ml_platform.api.app import app
from ml_platform.api.dependencies import get_prediction_service
from ml_platform.exceptions import (
    InvalidPredictionInputError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
)
from ml_platform.serving.schemas import PredictionRequest, PredictionResponse

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


class FakePredictionService:
    """Fake prediction service used for API tests."""

    def __init__(self, response: PredictionResponse | Exception) -> None:
        self._response = response

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if isinstance(self._response, Exception):
            raise self._response

        return self._response


def override_prediction_service(
    response: PredictionResponse | Exception,
) -> None:
    app.dependency_overrides[get_prediction_service] = lambda: FakePredictionService(
        response
    )


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction() -> None:
    expected_response = PredictionResponse(
        model_name="test-model",
        model_version="1",
        predictions=[42],
        request_id="test-request-id",
    )

    override_prediction_service(expected_response)

    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "test-model",
        "model_version": "1",
        "predictions": [42],
        "request_id": "test-request-id",
    }


def test_prediction_model_not_found() -> None:
    override_prediction_service(
        ModelNotFoundError("Unable to resolve model 'test-model'.")
    )

    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Unable to resolve model 'test-model'."}


def test_prediction_model_load_error() -> None:
    override_prediction_service(
        ModelLoadError("Failed to load model from URI: models:/test-model/1")
    )

    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to load the requested model."}


def test_prediction_invalid_input() -> None:
    override_prediction_service(
        InvalidPredictionInputError("Invalid prediction input.")
    )

    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid prediction input."}


def test_prediction_error() -> None:
    override_prediction_service(PredictionError("Prediction failed internally."))

    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed."}


def test_prediction_missing_model_name() -> None:
    response = client.post(
        "/predict",
        json={
            "model_version": "1",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 422


def test_prediction_empty_model_name() -> None:
    response = client.post(
        "/predict",
        json={
            "model_name": "",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 422


def test_prediction_missing_inputs() -> None:
    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
        },
    )

    assert response.status_code == 422


def test_prediction_invalid_inputs() -> None:
    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "1",
            "inputs": "invalid",
        },
    )

    assert response.status_code == 422


def test_prediction_without_model_version() -> None:
    expected_response = PredictionResponse(
        model_name="test-model",
        model_version="unknown",
        predictions=[42],
        request_id="test-request-id",
    )

    override_prediction_service(expected_response)

    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 200
    assert response.json()["model_version"] == "unknown"


def test_prediction_empty_model_version() -> None:
    response = client.post(
        "/predict",
        json={
            "model_name": "test-model",
            "model_version": "",
            "inputs": [{"feature": 10}],
        },
    )

    assert response.status_code == 422
