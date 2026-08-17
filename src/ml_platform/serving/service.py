from uuid import uuid4

from ml_platform.serving.contracts import ModelLoader, ModelResolver
from ml_platform.serving.schemas import (
    ModelReference,
    PredictionRequest,
    PredictionResponse,
)


class PredictionService:
    """Orchestrate model resolution, loading, and inference."""

    def __init__(
        self,
        resolver: ModelResolver,
        loader: ModelLoader,
    ) -> None:
        self._resolver = resolver
        self._loader = loader

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """Generate predictions for a prediction request."""
        reference = ModelReference(
            name=request.model_name,
            version=request.model_version,
        )

        model_uri = self._resolver.resolve(reference)

        predictor = self._loader.load(model_uri)

        predictions = predictor.predict(request.inputs)

        return PredictionResponse(
            model_name=request.model_name,
            model_version=request.model_version or "unknown",
            predictions=predictions,
            request_id=str(uuid4()),
        )
