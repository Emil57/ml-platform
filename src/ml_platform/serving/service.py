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
        model_name: str,
        model_version: str,
        model_alias: str | None = None,
    ) -> None:
        self._resolver = resolver
        self._loader = loader
        self._model_name = model_name
        self._model_version = model_version
        self._model_alias = model_alias

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """Generate predictions for a prediction request."""
        reference = ModelReference(
            name=self._model_name,
            version=self._model_version,
            alias=self._model_alias,
        )

        model_uri = self._resolver.resolve(reference)

        predictor = self._loader.load(model_uri)

        predictions = predictor.predict(request.inputs)

        return PredictionResponse(
            model_name=self._model_name,
            model_version=self._model_version,
            predictions=predictions,
            request_id=str(uuid4()),
        )
