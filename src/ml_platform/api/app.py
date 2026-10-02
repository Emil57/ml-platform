from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from ml_platform.api.handlers import (
    invalid_prediction_input_handler,
    model_load_error_handler,
    model_not_found_handler,
    prediction_error_handler,
    serving_error_handler,
)
from ml_platform.api.routes.predictions import router as prediction_router
from ml_platform.exceptions import (
    InvalidPredictionInputError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
    ServingError,
)

app = FastAPI(
    title="ML Plaform API",
    description="API for serving machine learning inference",
    version="0.1.0",
)


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(prediction_router)

app.add_exception_handler(
    ModelNotFoundError,
    model_not_found_handler,
)

app.add_exception_handler(
    ModelLoadError,
    model_load_error_handler,
)

app.add_exception_handler(
    InvalidPredictionInputError,
    invalid_prediction_input_handler,
)

app.add_exception_handler(
    PredictionError,
    prediction_error_handler,
)

app.add_exception_handler(
    ServingError,
    serving_error_handler,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
