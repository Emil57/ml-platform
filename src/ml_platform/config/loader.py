from .settings import Settings


def load_settings() -> Settings:
    """Create and return the platform settings."""
    return Settings()


settings = load_settings()
