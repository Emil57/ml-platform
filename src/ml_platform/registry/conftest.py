import mlflow
import mlflow.sklearn
import pytest
from sklearn.linear_model import LinearRegression

from ml_platform.registry.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path) -> ModelRegistry:
    database_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    return ModelRegistry()


@pytest.fixture
def registered_model(registry: ModelRegistry) -> str:
    model_name = "test-model"

    model = LinearRegression()
    model.fit([[1], [2], [3]], [1, 2, 3])

    for _ in range(2):
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name,
            )

    return model_name
