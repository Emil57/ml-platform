"""Run the serving API using platform settings."""

import uvicorn

from ml_platform.config import settings


def main() -> None:
    """Start the serving API."""
    uvicorn.run(
        "ml_platform.api.app:app",
        host=settings.serving_host,
        port=settings.serving_port,
    )


if __name__ == "__main__":
    main()
