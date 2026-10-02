from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

from ml_platform.api.app import app
from ml_platform.api.dependencies import get_prediction_service
from ml_platform.config import settings


class ConstantModel(PythonModel):
    def __init__(self, value: float, fail: bool = False) -> None:
        self.value = value
        self.fail = fail

    def predict(
        self, context: Any, model_input: list[Any], params: Any = None
    ) -> list[float]:
        if self.fail:
            raise ValueError("private-inference-payload")
        return [self.value for _ in model_input]


@pytest.fixture(scope="module")
def model_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    root = tmp_path_factory.mktemp("serving-mlflow")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = client.create_experiment(
            "serving-integration", artifact_location=(root / "artifacts").as_uri()
        )
        name = "serving-test-model"
        versions = []
        for value in (1.0, 2.0):
            with mlflow.start_run(experiment_id=experiment):
                info = mlflow.pyfunc.log_model(
                    name="model",
                    python_model=ConstantModel(value),
                    pip_requirements=[],
                )
                registered = mlflow.register_model(info.model_uri, name)
                versions.append(str(registered.version))
        client.set_registered_model_alias(name, "champion", versions[1])

        with mlflow.start_run(experiment_id=experiment):
            failing = mlflow.pyfunc.log_model(
                name="failing-model",
                python_model=ConstantModel(0.0, fail=True),
                pip_requirements=[],
            )
            mlflow.register_model(failing.model_uri, "inference-failure-model")

        broken = root / "broken-model"
        broken.mkdir()
        client.create_registered_model("load-failure-model")
        client.create_model_version("load-failure-model", source=broken.as_uri())
        yield {"uri": uri, "name": name, "v1": versions[0], "v2": versions[1]}
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


@pytest.fixture
def serving_client(
    model_store: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[TestClient]:
    # Keep downloads and other relative MLflow output inside the test workspace.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "mlflow_tracking_uri", model_store["uri"])
    monkeypatch.setattr(settings, "mlflow_registry_uri", model_store["uri"])
    monkeypatch.setattr(settings, "serving_model_name", model_store["name"])
    monkeypatch.setattr(settings, "serving_model_selector", "alias")
    monkeypatch.setattr(settings, "serving_model_alias", "champion")
    get_prediction_service.cache_clear()
    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        get_prediction_service.cache_clear()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)
