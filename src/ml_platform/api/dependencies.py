import mlflow
from mlflow import MlflowClient

from ml_platform.serving.loader import MLflowModelLoader
from ml_platform.serving.resolver import MLflowModelResolver
from ml_platform.serving.service import PredictionService
from ml_platform.config import settings


def get_prediction_service() -> PredictionService:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)

    client = MlflowClient()

    resolver = MLflowModelResolver(client)
    loader = MLflowModelLoader()

    return PredictionService(
        resolver=resolver,
        loader=loader,
        model_name=settings.serving_model_name,
        model_version=settings.serving_model_version,
        model_alias=settings.serving_model_alias,
    )
