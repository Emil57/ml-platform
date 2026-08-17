from fastapi import APIRouter, Depends

from ml_platform.api.dependencies import get_prediction_service
from ml_platform.serving.contracts import PredictionService
from ml_platform.serving.schemas import (
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),  # noqa: B008
) -> PredictionResponse:
    return service.predict(request)
