import os
import subprocess
import sys
from unittest.mock import MagicMock

import ml_platform.api.__main__ as server


def test_main_uses_serving_host_and_port(monkeypatch) -> None:
    run = MagicMock()
    monkeypatch.setattr(server.uvicorn, "run", run)
    monkeypatch.setattr(server.settings, "serving_host", "127.0.0.1")
    monkeypatch.setattr(server.settings, "serving_port", 9000)

    server.main()

    run.assert_called_once_with(
        "ml_platform.api.app:app",
        host="127.0.0.1",
        port=9000,
    )


def test_invalid_configuration_exits_with_clear_error() -> None:
    env = {**os.environ, "SERVING_PORT": "not-a-port"}

    result = subprocess.run(
        [sys.executable, "-m", "ml_platform.api"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "Invalid configuration" in result.stderr
    assert "serving_port" in result.stderr
    assert "Traceback" not in result.stderr
