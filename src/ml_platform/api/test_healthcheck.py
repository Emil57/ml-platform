import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ml_platform.api import healthcheck


class _Handler(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(self.status if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[HTTPServer]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def test_check_passes_when_health_responds(server: HTTPServer) -> None:
    assert healthcheck.check(server.server_address[1]) is True


def test_check_fails_when_health_returns_error(
    server: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_Handler, "status", 500)

    assert healthcheck.check(server.server_address[1]) is False


def test_check_fails_when_nothing_listens() -> None:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    httpd.server_close()

    assert healthcheck.check(port, timeout=1) is False


def test_main_uses_serving_port(
    server: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERVING_PORT", str(server.server_address[1]))

    with pytest.raises(SystemExit) as exc:
        healthcheck.main()

    assert exc.value.code == 0
