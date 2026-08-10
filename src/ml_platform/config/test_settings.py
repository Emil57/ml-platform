from ml_platform.config import settings
from ml_platform.config.settings import Settings


def test_default_environment():
    assert settings.environment == "development"


def test_random_seed():
    assert settings.random_seed == 42


def test_debug_default():
    assert settings.debug is False


def test_environment_override(monkeypatch):
    monkeypatch.setenv("ML_RANDOM_SEED", "123")

    settings = Settings()

    assert settings.random_seed == 123


def test_mlflow_default_settings():
    settings = Settings()

    assert settings.mlflow_tracking_uri == "mlruns"
    assert settings.mlflow_registry_uri == "mlruns"
    assert settings.mlflow_experiment_name == "default"


def test_mlflow_tracking_uri_override(monkeypatch):
    monkeypatch.setenv(
        "ML_MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://localhost:5000"


def test_mlflow_registry_uri_override(monkeypatch):
    monkeypatch.setenv(
        "ML_MLFLOW_REGISTRY_URI",
        "http://localhost:5000",
    )

    settings = Settings()

    assert settings.mlflow_registry_uri == "http://localhost:5000"


def test_mlflow_experiment_name_override(monkeypatch):
    monkeypatch.setenv(
        "ML_MLFLOW_EXPERIMENT_NAME",
        "ca_house_prediction",
    )

    settings = Settings()

    assert settings.mlflow_experiment_name == "ca_house_prediction"
