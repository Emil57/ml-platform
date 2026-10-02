"""
Platform settings.

This module defines the application's configuration using
Pydantic Settings.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Platform configuration."""

    environment: str = "development"
    debug: bool = False
    random_seed: int = 42

    train_split: float = 0.8
    validation_split: float = 0.2

    log_level: str = "INFO"

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", min_length=1)
    mlflow_registry_uri: str = Field(default="sqlite:///mlflow.db", min_length=1)
    mlflow_experiment_name: str = Field(default="default", min_length=1)

    serving_model_name: str = Field(default="f1-predictor", min_length=1)
    serving_model_selector: Literal["version", "alias"] = "version"
    serving_model_version: str = Field(default="1", min_length=1)
    serving_model_alias: str | None = Field(default=None, min_length=1)
    serving_host: str = Field(default="0.0.0.0", min_length=1)
    serving_port: int = Field(default=8000, ge=1, le=65535)
    serving_environment: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_model_selector(self) -> "Settings":
        if self.serving_model_selector == "alias" and not self.serving_model_alias:
            raise ValueError("SERVING_MODEL_ALIAS is required for alias selection")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
