from time import perf_counter
from uuid import uuid4

from ml_platform.exceptions import ModelLoadError, ModelNotFoundError
from ml_platform.serving.contracts import ModelLoader, ModelResolver
from ml_platform.serving.schemas import (
    ModelReference,
    PredictionRequest,
    PredictionResponse,
)
from ml_platform.utils import get_logger

logger = get_logger(__name__)


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
        self, request: PredictionRequest, *, request_id: str | None = None
    ) -> PredictionResponse:
        """Generate predictions for a prediction request."""
        reference = ModelReference(
            name=self._model_name,
            version=self._model_version,
            alias=self._model_alias,
        )
        effective_request_id = request_id or str(uuid4())
        started_at = perf_counter()

        try:
            model_uri = self._resolver.resolve(reference)
        except ModelNotFoundError as exc:
            latency_ms = (perf_counter() - started_at) * 1000
            logger.exception(
                "event=model_resolution_failed request_id=%s "
                "model_name=%s model_version=%s model_alias=%s"
                "latency_ms=%.3f status=failure error_type=%s error=%s",
                effective_request_id,
                self._model_name,
                self._model_version,
                self._model_alias,
                latency_ms,
                type(exc).__name__,
                str(exc),
            )
            raise

        try:
            predictor = self._loader.load(model_uri)

        except (ModelLoadError, Exception) as exc:
            latency_ms = (perf_counter() - started_at) * 1000
            logger.exception(
                "event=model_load_failed request_id=%s "
                "model_name=%s model_version=%s model_alias=%s"
                "latency_ms=%.3f status=failure error_type=%s error=%s",
                effective_request_id,
                self._model_name,
                self._model_version,
                self._model_alias,
                latency_ms,
                type(exc).__name__,
                str(exc),
            )
            raise

        try:
            predictions = predictor.predict(request.inputs)
        except Exception as exc:
            latency_ms = (perf_counter() - started_at) * 1000
            logger.exception(
                "event=prediction_failed request_id=%s "
                "model_name=%s model_version=%s model_alias=%s"
                "latency_ms=%.3f status=failure error_type=%s error=%s",
                effective_request_id,
                self._model_name,
                self._model_version,
                self._model_alias,
                latency_ms,
                type(exc).__name__,
                str(exc),
            )
            raise

        latency_ms = (perf_counter() - started_at) * 1000

        logger.info(
            "event=prediction_succeeded request_id=%s "
            "model_name=%s model_version=%s model_alias=%s "
            "latency_ms=%.3f status=success",
            effective_request_id,
            self._model_name,
            self._model_version,
            self._model_alias,
            latency_ms,
        )

        return PredictionResponse(
            model_name=self._model_name,
            model_version=self._model_version,
            predictions=predictions,
            request_id=effective_request_id,
        )
