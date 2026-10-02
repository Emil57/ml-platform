import logging
from threading import Lock
from time import perf_counter
from uuid import uuid4

from ml_platform.serving.contracts import ModelLoader, ModelResolver, Predictor
from ml_platform.serving.schemas import (
    ModelMetadata,
    ModelReference,
    PredictionRequest,
    PredictionResponse,
    ResolvedModel,
)
from ml_platform.utils import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Load one model per deployment and instrument prediction requests."""

    def __init__(
        self,
        resolver: ModelResolver,
        loader: ModelLoader,
        model_name: str,
        model_version: str | None,
        model_alias: str | None = None,
    ) -> None:
        self._resolver = resolver
        self._loader = loader
        self._reference = ModelReference(
            name=model_name, version=model_version, alias=model_alias
        )
        self._resolved: ResolvedModel | None = None
        self._predictor: Predictor | None = None
        self._load_lock = Lock()

    def _log(
        self, event: str, request_id: str, started_at: float, error: Exception | None
    ) -> None:
        resolved = self._resolved
        logger.log(
            logging.ERROR if error else logging.INFO,
            "event=%s request_id=%s model_name=%s model_version=%s "
            "model_alias=%s latency_ms=%.3f status=%s error_type=%s",
            event,
            request_id,
            self._reference.name,
            resolved.version if resolved else self._reference.version,
            resolved.alias if resolved else self._reference.alias,
            (perf_counter() - started_at) * 1000,
            "failure" if error else "success",
            type(error).__name__ if error else None,
        )

    def _ensure_loaded(self, request_id: str, started_at: float) -> None:
        with self._load_lock:
            if self._predictor is not None:
                return
            try:
                resolved = self._resolver.resolve(self._reference)
            except Exception as exc:
                self._log("model_resolution_failed", request_id, started_at, exc)
                raise
            self._resolved = resolved
            try:
                predictor = self._loader.load(resolved.uri)
            except Exception as exc:
                self._log("model_load_failed", request_id, started_at, exc)
                self._resolved = None
                raise
            self._predictor = predictor

    def check_readiness(self) -> ModelMetadata:
        """Load lazily; failed loads may be retried on the next request."""
        self._ensure_loaded(str(uuid4()), perf_counter())
        assert self._resolved is not None
        return ModelMetadata(
            model_name=self._resolved.name,
            model_version=self._resolved.version,
            model_alias=self._resolved.alias,
        )

    def get_model_metadata(self) -> ModelMetadata:
        return self.check_readiness()

    def predict(
        self, request: PredictionRequest, *, request_id: str | None = None
    ) -> PredictionResponse:
        effective_request_id = request_id or str(uuid4())
        started_at = perf_counter()
        self._ensure_loaded(effective_request_id, started_at)
        assert self._predictor is not None and self._resolved is not None
        try:
            predictions = self._predictor.predict(request.inputs)
        except Exception as exc:
            self._log("prediction_failed", effective_request_id, started_at, exc)
            raise
        response = PredictionResponse(
            model_name=self._resolved.name,
            model_version=self._resolved.version,
            predictions=predictions,
            request_id=effective_request_id,
        )
        self._log("prediction_succeeded", effective_request_id, started_at, None)
        return response
