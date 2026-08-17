from fastapi import FastAPI

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
