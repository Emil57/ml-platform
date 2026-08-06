import pytest

from ml_platform.exceptions import (
    ConfigurationError,
    DatasetError,
    DatasetNotFoundError,
    DatasetValidationError,
    MLPlatformError,
    PredictionError,
    TrainingError,
)


def test_base_exception_inherits_from_exception():
    """The base platform exception should inherit from Exception."""
    assert issubclass(MLPlatformError, Exception)


@pytest.mark.parametrize(
    "exception_cls",
    [
        DatasetError,
        DatasetNotFoundError,
        DatasetValidationError,
        ConfigurationError,
        TrainingError,
        PredictionError,
    ],
)
def test_exceptions_inherit_from_ml_platform_error(exception_cls):
    """All custom exceptions should inherit from MLPlatformError."""
    assert issubclass(exception_cls, MLPlatformError)


@pytest.mark.parametrize(
    ("exception_cls", "message"),
    [
        (DatasetNotFoundError, "Dataset not found."),
        (DatasetValidationError, "Dataset validation failed."),
        (ConfigurationError, "Invalid configuration."),
        (TrainingError, "Training failed."),
        (PredictionError, "Prediction failed."),
    ],
)
def test_exception_message(exception_cls, message):
    """Custom exceptions should preserve the provided message."""
    with pytest.raises(exception_cls, match=message):
        raise exception_cls(message)
