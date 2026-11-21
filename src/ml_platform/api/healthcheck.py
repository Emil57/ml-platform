"""Container health check: exit 0 when the API's /health endpoint responds.

Run with ``python -m ml_platform.api.healthcheck``. It uses only the standard
library and never loads the model, so it stays fast and lightweight.
"""

import os
import sys
import urllib.request


def check(port: int, timeout: float = 4.0) -> bool:
    """Return True when GET /health on localhost answers with HTTP 200."""

    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except OSError:
        return False


def main() -> None:
    port = int(os.environ.get("SERVING_PORT", "8000"))
    sys.exit(0 if check(port) else 1)


if __name__ == "__main__":
    main()
