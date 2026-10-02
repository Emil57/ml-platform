import pytest
from pydantic import ValidationError

from ml_platform.config import settings
from ml_platform.config.settings import Settings


def test_default_environment():
    assert settings.environment == "development"


def test_random_seed():
    assert settings.random_seed == 42


def test_debug_default():
    assert settings.debug is False


def test_environment_override(monkeypatch):
    monkeypatch.setenv("RANDOM_SEED", "123")

    settings = Settings()

    assert settings.random_seed == 123


def test_mlflow_default_settings():
    settings = Settings()

    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_registry_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_experiment_name == "default"


def test_mlflow_tracking_uri_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    settings = Settings()

    assert settings.mlflow_tracking_uri == "http://localhost:5000"


def test_mlflow_registry_uri_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_REGISTRY_URI",
        "http://localhost:5000",
    )

    settings = Settings()

    assert settings.mlflow_registry_uri == "http://localhost:5000"


def test_mlflow_experiment_name_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_EXPERIMENT_NAME",
        "ca_house_prediction",
    )

    settings = Settings()

    assert settings.mlflow_experiment_name == "ca_house_prediction"


def test_serving_default_settings():
    settings = Settings()

    assert settings.serving_model_name == "f1-predictor"
    assert settings.serving_model_alias is None
    assert settings.serving_host == "0.0.0.0"
    assert settings.serving_port == 8000


def test_serving_model_name_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_MODEL_NAME",
        "my-model",
    )

    settings = Settings()
    assert settings.serving_model_name == "my-model"


def test_serving_model_alias_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_MODEL_ALIAS",
        "production",
    )

    settings = Settings()
    assert settings.serving_model_alias == "production"


def test_serving_host_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_HOST",
        "127.0.0.1",
    )

    settings = Settings()

    assert settings.serving_host == "127.0.0.1"


def test_serving_port_override(monkeypatch):
    monkeypatch.setenv(
        "SERVING_PORT",
        "9000",
    )

    settings = Settings()

    assert settings.serving_port == 9000


def test_serving_port_must_be_valid():
    with pytest.raises(ValueError):
        Settings(serving_port=0)


def test_serving_port_cannot_exceed_maximum():
    with pytest.raises(ValueError):
        Settings(serving_port=65536)


def test_serving_settings_have_defaults() -> None:
    settings = Settings()

    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_registry_uri == "sqlite:///mlflow.db"
    assert settings.serving_model_name == "f1-predictor"
    assert settings.serving_model_version == "1"
    assert settings.serving_model_alias is None
    assert settings.serving_host == "0.0.0.0"
    assert settings.serving_port == 8000


def test_serving_settings_can_be_overridden_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///test.db")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "sqlite:///registry.db")
    monkeypatch.setenv("SERVING_MODEL_NAME", "test-model")
    monkeypatch.setenv("SERVING_MODEL_VERSION", "2")
    monkeypatch.setenv("SERVING_MODEL_ALIAS", "champion")
    monkeypatch.setenv("SERVING_HOST", "127.0.0.1")
    monkeypatch.setenv("SERVING_PORT", "9000")
    monkeypatch.setenv("SERVING_ENVIRONMENT", "testing")

    settings = Settings()

    assert settings.mlflow_tracking_uri == "sqlite:///test.db"
    assert settings.mlflow_registry_uri == "sqlite:///registry.db"
    assert settings.serving_model_name == "test-model"
    assert settings.serving_model_version == "2"
    assert settings.serving_model_alias == "champion"
    assert settings.serving_host == "127.0.0.1"
    assert settings.serving_port == 9000
    assert settings.serving_environment == "testing"


def test_invalid_serving_port_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Settings(serving_port=0)

    with pytest.raises(ValidationError):
        Settings(serving_port=70000)


def test_empty_serving_model_name_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Settings(serving_model_name="")


def test_empty_serving_model_version_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Settings(serving_model_version="")


def test_empty_serving_host_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        Settings(serving_host="")


def test_alias_selection_requires_alias() -> None:
    with pytest.raises(ValidationError, match="SERVING_MODEL_ALIAS"):
        Settings(serving_model_selector="alias", serving_model_alias=None)


def test_alias_selector_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVING_MODEL_SELECTOR", "alias")
    monkeypatch.setenv("SERVING_MODEL_ALIAS", "champion")
    configured = Settings()
    assert configured.serving_model_selector == "alias"
    assert configured.serving_model_alias == "champion"


def test_invalid_model_selector() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"serving_model_selector": "latest"})
