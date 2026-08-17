from fastapi import Request
from fastapi.responses import JSONResponse

from ml_platform.exceptions import (
    InvalidPredictionInputError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
    ServingError,
)


def model_not_found_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    error = exc
    assert isinstance(error, ModelNotFoundError)

    return JSONResponse(
        status_code=404,
        content={"detail": str(error)},
    )


def invalid_prediction_input_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    error = exc
    assert isinstance(error, InvalidPredictionInputError)

    return JSONResponse(
        status_code=422,
        content={"detail": str(error)},
    )


def model_load_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Unable to load the requested model."},
    )


def prediction_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Prediction failed."},
    )


def serving_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "A serving error occurred."},
    )