"""
Platform settings.

This module defines the application's configuration using
Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform configuration."""

    environment: str = "development"
    debug: bool = False
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_prefix="ML_",
        extra="ignore",
    )
