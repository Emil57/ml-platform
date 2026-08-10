"""
Platform settings.

This module defines the application's configuration using
Pydantic Settings.
"""

from pathlib import Path

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

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        extra="ignore",
    )
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
