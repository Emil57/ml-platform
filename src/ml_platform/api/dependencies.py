from functools import lru_cache

import mlflow

from ml_platform.config import settings
from ml_platform.serving.loader import MLflowModelLoader
from ml_platform.serving.resolver import MLflowModelResolver
from ml_platform.serving.service import PredictionService


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)

    # The resolver connects to MLflow on first use, so an unreachable
    # registry is reported as "not ready" (503) instead of a server error.
    resolver = MLflowModelResolver()
    loader = MLflowModelLoader()

    return PredictionService(
        resolver=resolver,
        loader=loader,
        model_name=settings.serving_model_name,
        model_version=(
            settings.serving_model_version
            if settings.serving_model_selector == "version"
            else None
        ),
        model_alias=(
            settings.serving_model_alias
            if settings.serving_model_selector == "alias"
            else None
        ),
    )
