"""Run the serving API using platform settings."""

import sys

import uvicorn
from pydantic import ValidationError

try:
    from ml_platform.config import settings
except ValidationError as exc:
    # Settings load from the environment on import; fail with a short,
    # readable message instead of a traceback when they are invalid.
    sys.exit(f"Invalid configuration:\n{exc}")


def main() -> None:
    """Start the serving API."""
    uvicorn.run(
        "ml_platform.api.app:app",
        host=settings.serving_host,
        port=settings.serving_port,
    )


if __name__ == "__main__":
    main()
