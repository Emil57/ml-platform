"""Custom exceptions used throughout the ML Platform."""


class MLPlatformError(Exception):
    """Base exception for all ML Platform errors."""


# ============================================================================
# Dataset Exceptions
# ============================================================================


class DatasetError(MLPlatformError):
    """Base class for dataset-related errors."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found."""


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationError(MLPlatformError):
    """Raised when application configuration is invalid."""


# ============================================================================
# Training Exceptions
# ============================================================================


class TrainingError(MLPlatformError):
    """Raised when model training fails."""


# ============================================================================
# Prediction Exceptions
# ============================================================================


class PredictionError(MLPlatformError):
    """Raised when model prediction fails."""


# ============================================================================
# Evaluation Exceptions
# ============================================================================


class EvaluationError(Exception):
    """Raised when model evaluation fails."""


# ============================================================================
# Artifact Exceptions
# ============================================================================


class ArtifactError(Exception):
    """Raised when an artifact operation fails."""
