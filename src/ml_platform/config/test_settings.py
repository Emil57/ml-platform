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
