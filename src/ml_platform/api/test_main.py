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
