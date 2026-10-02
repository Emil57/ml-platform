import logging
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from ml_platform.exceptions import ModelLoadError, PredictionError
from ml_platform.serving import service as service_module
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


def test_prediction_logs_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = MagicMock()
    loader = MagicMock()
    predictor = MagicMock()

    resolver.resolve.return_value = "models:/test-model/1"
    loader.load.return_value = predictor
    predictor.predict.return_value = [42]

    monkeypatch.setattr(
        service_module,
        "perf_counter",
        MagicMock(side_effect=[10.0, 10.018]),
    )

    service = PredictionService(
        resolver=resolver,
        loader=loader,
        model_name="test-model",
        model_version="1",
        model_alias="champion",
    )

    with caplog.at_level(logging.INFO):
        response = service.predict(
            PredictionRequest(inputs=[{"feature": 10}]),
            request_id="request-123",
        )

    assert response.predictions == [42]
    assert response.request_id == "request-123"
    assert "event=prediction_succeeded" in caplog.text
    assert "request_id=request-123" in caplog.text
    assert "model_name=test-model" in caplog.text
    assert "model_version=1" in caplog.text
    assert "model_alias=champion" in caplog.text
    assert "latency_ms=18.000" in caplog.text
    assert "status=success" in caplog.text


def test_prediction_generates_request_id() -> None:
    resolver = MagicMock()
    loader = MagicMock()
    predictor = MagicMock()

    resolver.resolve.return_value = "models:/test-model/1"
    loader.load.return_value = predictor
    predictor.predict.return_value = [42]

    service = PredictionService(
        resolver=resolver,
        loader=loader,
        model_name="test-model",
        model_version="1",
    )

    response = service.predict(
        PredictionRequest(inputs=[{"feature": 10}]),
    )

    UUID(response.request_id)


def test_prediction_logs_model_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = MagicMock()
    loader = MagicMock()

    resolver.resolve.return_value = "models:/test-model/1"
    loader.load.side_effect = ModelLoadError("Unable to load model.")

    monkeypatch.setattr(
        service_module,
        "perf_counter",
        MagicMock(side_effect=[10.0, 10.025]),
    )

    service = PredictionService(
        resolver=resolver,
        loader=loader,
        model_name="test-model",
        model_version="1",
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ModelLoadError, match="Unable to load model"):
            service.predict(
                PredictionRequest(inputs=[{"feature": 10}]),
                request_id="request-load-failure",
            )

    assert "event=model_load_failed" in caplog.text
    assert "request_id=request-load-failure" in caplog.text
    assert "status=failure" in caplog.text
    assert "error_type=ModelLoadError" in caplog.text


def test_prediction_logs_inference_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resolver = MagicMock()
    loader = MagicMock()
    predictor = MagicMock()

    resolver.resolve.return_value = "models:/test-model/1"
    loader.load.return_value = predictor
    predictor.predict.side_effect = PredictionError("Inference failed.")

    monkeypatch.setattr(
        service_module,
        "perf_counter",
        MagicMock(side_effect=[10.0, 10.012]),
    )

    service = PredictionService(
        resolver=resolver,
        loader=loader,
        model_name="test-model",
        model_version="1",
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(PredictionError, match="Inference failed"):
            service.predict(
                PredictionRequest(inputs=[{"feature": 10}]),
                request_id="request-prediction-failure",
            )

    assert "event=prediction_failed" in caplog.text
    assert "request_id=request-prediction-failure" in caplog.text
    assert "status=failure" in caplog.text
    assert "error_type=PredictionError" in caplog.text


def test_prediction_does_not_log_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "do-not-log-this-value"

    resolver = MagicMock()
    loader = MagicMock()
    predictor = MagicMock()

    resolver.resolve.return_value = "models:/test-model/1"
    loader.load.return_value = predictor
    predictor.predict.return_value = [42]

    service = PredictionService(
        resolver=resolver,
        loader=loader,
        model_name="test-model",
        model_version="1",
    )

    with caplog.at_level(logging.INFO):
        service.predict(
            PredictionRequest(inputs=[{"api_token": secret}]),
            request_id="request-private",
        )

    assert secret not in caplog.text
    assert "api_token" not in caplog.text
