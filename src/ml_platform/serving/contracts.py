from typing import Any, Protocol

from ml_platform.serving.schemas import (
    ModelMetadata,
    ModelReference,
    PredictionRequest,
    PredictionResponse,
    ResolvedModel,
)


class Predictor(Protocol):
    def predict(self, inputs: list[dict[str, Any]]) -> list[Any]:
        """Generate predictions for the provided inputs."""
        ...


class ModelResolver(Protocol):
    def resolve(self, reference: ModelReference) -> ResolvedModel:
        """Resolve a model reference to a model URI."""
        ...


class PredictionService(Protocol):
    def get_model_metadata(self) -> ModelMetadata:
        """Return the identity of the loaded model."""
        ...

    def check_readiness(self) -> ModelMetadata:
        """Ensure the model is loaded and available for inference."""
        ...

    def predict(
        self, request: PredictionRequest, *, request_id: str | None = None
    ) -> PredictionResponse:
        """Execute a prediction request."""
        ...


class ModelLoader(Protocol):
    def load(self, model_uri: str) -> Predictor:
        """Load a model from the specified URI."""
        ...
