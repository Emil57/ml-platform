from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_name: str = Field(min_length=1)
    model_version: str | None = Field(default=None, min_length=1)
    inputs: list[dict[str, Any]]

class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[Any]
    request_id: str

class ModelReference(BaseModel):
    name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    alias: str | None = Field(default=None, min_length=1)