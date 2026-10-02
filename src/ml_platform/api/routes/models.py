from fastapi import APIRouter, Depends, HTTPException

from ml_platform.api.dependencies import get_prediction_service
from ml_platform.exceptions import ModelLoadError, ModelNotFoundError
from ml_platform.serving.contracts import PredictionService
from ml_platform.serving.schemas import ModelMetadata, ReadinessResponse

router = APIRouter()


@router.get("/ready", response_model=ReadinessResponse)
def ready(
    service: PredictionService = Depends(get_prediction_service),  # noqa: B008
) -> ReadinessResponse:
    try:
        metadata = service.check_readiness()
    except (ModelNotFoundError, ModelLoadError) as exc:
        raise HTTPException(status_code=503, detail="Model is not ready.") from exc
    return ReadinessResponse(model=metadata)


@router.get("/model", response_model=ModelMetadata)
def model_metadata(
    service: PredictionService = Depends(get_prediction_service),  # noqa: B008
) -> ModelMetadata:
    try:
        return service.get_model_metadata()
    except (ModelNotFoundError, ModelLoadError) as exc:
        raise HTTPException(status_code=503, detail="Model is not ready.") from exc
