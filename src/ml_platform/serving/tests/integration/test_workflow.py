import logging
import re
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from ml_platform.config import settings


def test_registered_alias_to_prediction(
    serving_client: TestClient,
    model_store: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert serving_client.get("/health").json() == {"status": "ok"}
    expected = {
        "model_name": model_store["name"],
        "model_version": model_store["v2"],
        "model_alias": "champion",
    }
    ready = serving_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model": expected}
    metadata = serving_client.get("/model")
    assert metadata.status_code == 200
    assert metadata.json() == expected
    with caplog.at_level(logging.INFO, logger="ml_platform.serving.service"):
        response = serving_client.post(
            "/predict",
            headers={"X-Request-ID": "integration-request"},
            json={"inputs": [{"secret": "private-input"}, {"x": 5}]},
        )
    assert response.status_code == 200
    assert response.json() == {
        "model_name": model_store["name"],
        "model_version": model_store["v2"],
        "predictions": [2.0, 2.0],
        "request_id": "integration-request",
    }
    assert response.headers["X-Request-ID"] == "integration-request"
    messages = [
        r.getMessage()
        for r in caplog.records
        if "event=prediction_succeeded" in r.getMessage()
    ]
    assert len(messages) == 1
    message = messages[0]
    for field in (
        "event=prediction_succeeded",
        "request_id=integration-request",
        f"model_name={model_store['name']}",
        f"model_version={model_store['v2']}",
        "model_alias=champion",
        "status=success",
    ):
        assert field in message
    latency = re.search(r"latency_ms=(\d+\.\d+)", message)
    assert latency is not None and float(latency.group(1)) >= 0
    assert "private-input" not in caplog.text


def test_alias_is_pinned_after_loading(
    serving_client: TestClient, model_store: dict[str, str]
) -> None:
    assert serving_client.get("/ready").status_code == 200
    registry = MlflowClient(
        tracking_uri=model_store["uri"], registry_uri=model_store["uri"]
    )
    try:
        registry.set_registered_model_alias(
            model_store["name"], "champion", model_store["v1"]
        )
        response = serving_client.post("/predict", json={"inputs": [{"x": 1}]})
        assert response.json()["model_version"] == model_store["v2"]
        assert response.json()["predictions"] == [2.0]
        assert serving_client.get("/model").json()["model_version"] == model_store["v2"]
    finally:
        registry.set_registered_model_alias(
            model_store["name"], "champion", model_store["v2"]
        )


def test_generated_request_id(serving_client: TestClient) -> None:
    response = serving_client.post("/predict", json={"inputs": [{"x": 1}]})
    assert response.status_code == 200
    identifier = response.headers["X-Request-ID"]
    UUID(identifier)
    assert response.json()["request_id"] == identifier


@pytest.mark.parametrize("body", [{}, {"inputs": "invalid"}, {"inputs": []}])
def test_invalid_requests(serving_client: TestClient, body: dict) -> None:
    response = serving_client.post("/predict", json=body)
    assert response.status_code == 422
    UUID(response.headers["X-Request-ID"])


def test_invalid_alias(
    serving_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "serving_model_alias", "missing")
    with caplog.at_level(logging.ERROR):
        response = serving_client.post(
            "/predict",
            headers={"X-Request-ID": "bad-alias"},
            json={"inputs": [{"x": 1}]},
        )
    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == "bad-alias"
    assert "event=model_resolution_failed" in caplog.text
    assert serving_client.get("/health").status_code == 200
    for endpoint in ("/ready", "/model"):
        failure = serving_client.get(endpoint)
        assert failure.status_code == 503
        assert failure.json() == {"detail": "Model is not ready."}


def test_real_model_load_failure(
    serving_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "serving_model_selector", "version")
    monkeypatch.setattr(settings, "serving_model_version", "1")
    monkeypatch.setattr(settings, "serving_model_name", "load-failure-model")
    with caplog.at_level(logging.ERROR):
        response = serving_client.post(
            "/predict",
            headers={"X-Request-ID": "load-failure"},
            json={"inputs": [{"x": 1}]},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to load the requested model."}
    assert response.headers["X-Request-ID"] == "load-failure"
    assert "event=model_load_failed" in caplog.text
    for endpoint in ("/ready", "/model"):
        assert serving_client.get(endpoint).status_code == 503


def test_real_inference_failure(
    serving_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "serving_model_selector", "version")
    monkeypatch.setattr(settings, "serving_model_version", "1")
    monkeypatch.setattr(settings, "serving_model_name", "inference-failure-model")
    assert serving_client.get("/ready").status_code == 200
    with caplog.at_level(logging.ERROR):
        response = serving_client.post(
            "/predict",
            headers={"X-Request-ID": "inference-failure"},
            json={"inputs": [{"x": "private-inference-payload"}]},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed."}
    assert response.headers["X-Request-ID"] == "inference-failure"
    assert "event=prediction_failed" in caplog.text
    assert "model_version=1" in caplog.text
    assert "error_type=PredictionError" in caplog.text
    assert "private-inference-payload" not in caplog.text
