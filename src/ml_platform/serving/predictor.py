from typing import Any

from ml_platform.exceptions import PredictionError
from ml_platform.serving.contracts import Predictor


class MLflowPredictor(Predictor):
    """Execute predictions using a loaded MLflow model."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, inputs: list[dict[str, Any]]) -> list[Any]:
        """Generate predictions using the loaded model."""
        try:
            predictions = self._model.predict(inputs)
        except Exception as exc:
            raise PredictionError("Model prediction failed.") from exc

        return predictions
