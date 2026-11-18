"""Fixtures for testing the serving Docker image from outside the container.

Set ML_PLATFORM_IMAGE to test an image that is already built; otherwise the
image is built from the repository's Dockerfile once per test session.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_LABEL = "ml-platform.container-test=true"
STARTUP_TIMEOUT = 90.0


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check, timeout=900
    )


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return docker("info", check=False).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _docker_available():
        return
    skip = pytest.mark.skip(reason="Docker daemon is not available")
    for item in items:
        if "tests/container" in str(item.path):
            item.add_marker(skip)


@dataclass
class Container:
    id: str
    port: int

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def logs(self) -> str:
        result = docker("logs", self.id, check=False)
        return result.stdout + result.stderr

    def wait_until_ready(self, timeout: float = STARTUP_TIMEOUT) -> None:
        """Poll /health until it answers instead of sleeping a fixed time."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = docker("inspect", "-f", "{{.State.Running}}", self.id).stdout
            if state.strip() != "true":
                pytest.fail(f"Container exited during startup:\n{self.logs()}")
            try:
                if self.request("GET", "/health")[0] == 200:
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        pytest.fail(f"API did not become ready in {timeout}s:\n{self.logs()}")


@pytest.fixture(scope="session")
def image() -> str:
    prebuilt = os.environ.get("ML_PLATFORM_IMAGE")
    if prebuilt:
        return prebuilt

    tag = "ml-platform:container-test"
    result = docker("build", "-t", tag, str(REPO_ROOT), check=False)
    if result.returncode != 0:
        pytest.fail(f"docker build failed:\n{result.stdout}\n{result.stderr}")
    return tag


@pytest.fixture(scope="session", autouse=True)
def _remove_leftover_containers() -> Iterator[None]:
    yield
    if not _docker_available():
        return
    ids = docker("ps", "-aq", "--filter", f"label={TEST_LABEL}").stdout.split()
    if ids:
        docker("rm", "-f", *ids, check=False)


StartContainer = Callable[..., Container]


@pytest.fixture
def start_container(image: str) -> Iterator[StartContainer]:
    """Start containers for a test; they are removed even if the test fails."""

    started: list[str] = []

    def start(
        env: dict[str, str] | None = None,
        container_port: int = 8000,
        extra_args: tuple[str, ...] = (),
        wait: bool = True,
    ) -> Container:
        env_args = [arg for k, v in (env or {}).items() for arg in ("-e", f"{k}={v}")]
        container_id = docker(
            "run",
            "-d",
            "--label",
            TEST_LABEL,
            "-p",
            f"127.0.0.1::{container_port}",
            *env_args,
            *extra_args,
            image,
        ).stdout.strip()
        started.append(container_id)

        mapping = docker("port", container_id, f"{container_port}/tcp").stdout
        container = Container(container_id, int(mapping.split()[0].rsplit(":", 1)[1]))
        if wait:
            container.wait_until_ready()
        return container

    yield start

    if started:
        docker("rm", "-f", *started, check=False)


@pytest.fixture(scope="session")
def model_store() -> Iterator[dict[str, str]]:
    """A SQLite MLflow registry with one sklearn model aliased as 'champion'.

    The directory is mounted into the container at the same path, so the
    artifact paths stored in the registry resolve inside the container too.
    """

    import mlflow
    import pandas as pd
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.linear_model import LinearRegression

    root = Path(tempfile.mkdtemp(prefix="ml-platform-container-"))
    uri = f"sqlite:///{root / 'mlflow.db'}"
    previous = (mlflow.get_tracking_uri(), mlflow.get_registry_uri())
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = client.create_experiment(
            "container-test", artifact_location=(root / "artifacts").as_uri()
        )
        features = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.0, 1.0, 0.0]})
        model = LinearRegression().fit(features, [2.0, 4.0, 6.0])
        with mlflow.start_run(experiment_id=experiment):
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=infer_signature(features, model.predict(features)),
            )
        name = "container-test-model"
        version = mlflow.register_model(info.model_uri, name).version
        client.set_registered_model_alias(name, "champion", str(version))

        # The container runs as a non-root user that must read and write here.
        for path in [root, *root.rglob("*")]:
            path.chmod(0o777 if path.is_dir() else 0o666)

        yield {"root": str(root), "uri": uri, "name": name, "version": str(version)}
    finally:
        mlflow.set_tracking_uri(previous[0])
        mlflow.set_registry_uri(previous[1])
        shutil.rmtree(root, ignore_errors=True)
