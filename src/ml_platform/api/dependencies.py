from mlflow import MlflowClient

from ml_platform.serving.loader import MLflowModelLoader
from ml_platform.serving.resolver import MLflowModelResolver
from ml_platform.serving.service import PredictionService


def get_prediction_service() -> PredictionService:
    client = MlflowClient()

    resolver = MLflowModelResolver(client)
    loader = MLflowModelLoader()

    return PredictionService(
        resolver=resolver,
        loader=loader,
    )
