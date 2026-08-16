from typing import Any, Protocol

from ml_platform.serving.schemas import (
    ModelReference,
    PredictionRequest,
    PredictionResponse,
)


class Predictor(Protocol):
    def predict(self, inputs: list[dict[str, Any]]) -> list[Any]:
        """Generate predictions for the provided inputs."""
        ...


class ModelResolver(Protocol):
    def resolve(self, reference: ModelReference) -> Predictor:
        """Resolve a model reference into a predictor."""
        ...


class PredictionService(Protocol):
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Execute a prediction request."""
        ...


class ModelLoader(Protocol):
    def load(self, model_uri: str) -> Predictor:
        """Load a model from the specified URI."""
        ...
