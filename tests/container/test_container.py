"""Tests for the serving Docker image, run against real containers.

Run with: uv run pytest tests/container
"""

import time

import pytest

from tests.container.conftest import Container, StartContainer, docker


def test_health_endpoint_is_reachable(start_container: StartContainer) -> None:
    container = start_container()

    assert container.request("GET", "/health") == (200, {"status": "ok"})


def test_docker_reports_container_healthy(start_container: StartContainer) -> None:
    container = start_container(extra_args=("--health-interval=1s",))

    deadline = time.monotonic() + 60
    status = ""
    while time.monotonic() < deadline:
        status = docker(
            "inspect", "-f", "{{.State.Health.Status}}", container.id
        ).stdout.strip()
        if status == "healthy":
            break
        time.sleep(1)

    assert status == "healthy", container.logs()


def test_runtime_configuration_overrides_defaults(
    start_container: StartContainer,
) -> None:
    container = start_container(
        env={
            "SERVING_PORT": "9000",
            "SERVING_MODEL_NAME": "configured-model",
            "SERVING_MODEL_SELECTOR": "alias",
            "SERVING_MODEL_ALIAS": "staging",
            "MLFLOW_TRACKING_URI": "sqlite:////tmp/empty.db",
            "MLFLOW_REGISTRY_URI": "sqlite:////tmp/empty.db",
        },
        container_port=9000,
    )

    # No such model is registered, so readiness fails, and the logs show
    # which model the container was configured to serve.
    status, _ = container.request("GET", "/ready")

    assert status == 503
    logs = container.logs()
    assert "Uvicorn running on http://0.0.0.0:9000" in logs
    assert "model_name=configured-model" in logs
    assert "model_alias=staging" in logs


def test_prediction_with_registered_model(
    start_container: StartContainer, model_store: dict[str, str]
) -> None:
    container: Container = start_container(
        env={
            "MLFLOW_TRACKING_URI": model_store["uri"],
            "MLFLOW_REGISTRY_URI": model_store["uri"],
            "SERVING_MODEL_NAME": model_store["name"],
            "SERVING_MODEL_SELECTOR": "alias",
            "SERVING_MODEL_ALIAS": "champion",
        },
        extra_args=("-v", f"{model_store['root']}:{model_store['root']}"),
    )

    status, ready = container.request("GET", "/ready")
    assert status == 200, container.logs()
    assert ready["model"]["model_version"] == model_store["version"]

    status, response = container.request(
        "POST", "/predict", {"inputs": [{"x1": 4.0, "x2": 0.0}]}
    )
    assert status == 200, container.logs()
    assert response["model_name"] == model_store["name"]
    assert response["predictions"] == [pytest.approx(8.0)]


def test_invalid_configuration_stops_container(image: str) -> None:
    result = docker("run", "--rm", "-e", "SERVING_PORT=not-a-port", image, check=False)

    assert result.returncode == 1
    assert "Invalid configuration" in result.stderr
    assert "serving_port" in result.stderr


def test_container_runs_as_non_root_user(image: str) -> None:
    result = docker("run", "--rm", image, "id", "-u")

    assert result.stdout.strip() != "0"


def test_image_contains_no_local_files(image: str) -> None:
    result = docker("run", "--rm", image, "ls", "-A", "/app")

    assert result.stdout.split() == [".venv", "src"]
