from fastapi import APIRouter, Depends, Request

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
    http_request: Request,
    service: PredictionService = Depends(get_prediction_service),  # noqa: B008
) -> PredictionResponse:
    request_id = str(http_request.state.request_id)

    return service.predict(request, request_id=request_id)
