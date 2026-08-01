from ml_platform.config import settings


def test_default_environment():
    assert settings.environment == "development"


def test_random_seed():
    assert settings.random_seed == 42


def test_debug_default():
    assert settings.debug is False
