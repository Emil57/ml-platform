from collections.abc import Callable
from typing import Any

import mlflow

from ml_platform.exceptions import ModelLoadError
from ml_platform.serving.contracts import ModelLoader, Predictor
from ml_platform.serving.predictor import MLflowPredictor


class MLflowModelLoader(ModelLoader):
    """Load models from MLflow."""

    def __init__(
        self,
        load_model: Callable[[str], Any] = mlflow.pyfunc.load_model,
    ) -> None:
        self._load_model = load_model

    def load(self, model_uri: str) -> Predictor:
        """Load a model from an MLflow model URI."""
        try:
            model = self._load_model(model_uri)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model from URI: {model_uri}") from exc

        if not callable(getattr(model, "predict", None)):
            raise ModelLoadError(
                f"Loaded model does not implement predict(): {model_uri}"
            )

        return MLflowPredictor(model)
