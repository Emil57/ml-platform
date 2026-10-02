from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    inputs: list[dict[str, Any]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[Any]
    request_id: str


class ModelReference(BaseModel):
    name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    alias: str | None = Field(default=None, min_length=1)


class ResolvedModel(BaseModel):
    """Registry identity pinned to a concrete version."""

    name: str
    version: str
    alias: str | None = None
    uri: str


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    model_alias: str | None = None


class ReadinessResponse(BaseModel):
    status: str = "ready"
    model: ModelMetadata
