# Machine Learning Platform

A production-oriented machine learning platform for building reproducible training pipelines and serving registered models through an HTTP API.

The repository separates reusable platform capabilities from model-specific pipelines. Shared concerns such as configuration, data handling, experiment tracking, model registration, evaluation, artifacts, and serving live in
`src/ml_platform`; individual projects live in `src/pipelines`.

## Architecture

```text
                         +--------------------+
                         |  Model Pipelines   |
                         |  F1 / CA Housing   |
                         +---------+----------+
                                   |
              +--------------------+--------------------+
              |                                         |
     +--------v---------+                      +--------v---------+
     | Training Lifecycle|                      | Serving Lifecycle |
     | data -> train ->  |                      | API -> resolve -> |
     | evaluate -> track |                      | load -> predict   |
     +--------+---------+                      +--------+---------+
              |                                         |
              +--------------------+--------------------+
                                   |
                         +---------v----------+
                         | Shared Platform    |
                         | configuration,     |
                         | artifacts, logging,|
                         | exceptions, tests  |
                         +--------------------+
```

The platform is intentionally modular. Pipelines own domain data, feature
engineering, and model choices; platform packages provide reusable contracts
and infrastructure around them.

## Repository Layout

```text
.
|-- .dvc/                # DVC configuration
|-- .github/             # Repository automation
|-- artifacts/           # Local generated artifacts
|-- docs/                # Cross-cutting project documentation
|-- mlruns/              # Local MLflow experiment output
|-- src/
|   |-- examples/        # Runnable usage examples
|   |-- ml_platform/     # Reusable platform capabilities
|   |   |-- api/         # HTTP application boundary
|   |   |-- artifacts/
|   |   |-- config/
|   |   |-- data/
|   |   |-- evaluation/
|   |   |-- exceptions/
|   |   |-- registry/
|   |   |-- serving/
|   |   |-- tracking/
|   |   |-- training/
|   |   `-- utils/
|   `-- pipelines/       # Model-specific, DVC-backed pipelines
|       |-- ca_house_prediction/
|       `-- f1/
|-- .env.example         # Configuration template
|-- pyproject.toml       # Dependencies and tool configuration
`-- uv.lock              # Locked dependency graph
```

## Platform Capabilities

The current platform includes configuration, data management, training,
evaluation, experiment tracking, artifact management, model registration,
and model serving. The FastAPI application exposes health and prediction
endpoints; model identity is configured at deployment time rather than sent
by each client request.

For package-level design and usage, see:

| Area | Documentation |
| --- | --- |
| Platform overview | [ml_platform](src/ml_platform/README.md) |
| Data | [data](src/ml_platform/data/README.md) |
| Training | [training](src/ml_platform/training/README.md) |
| Evaluation | [evaluation](src/ml_platform/evaluation/README.md) |
| Tracking | [tracking](src/ml_platform/tracking/README.md) |
| Artifacts | [artifacts](src/ml_platform/artifacts/README.md) |
| Serving | [serving](src/ml_platform/serving/README.md) |
| F1 pipeline | [f1](src/pipelines/f1/README.md) |
| California Housing pipeline | [ca_house_prediction](src/pipelines/ca_house_prediction/README.md) |

## Configuration

Configuration is managed with Pydantic Settings. Copy `.env.example` to
`.env` and adjust values for your environment. Environment variables take
precedence over defaults.

MLflow tracking and registry URIs, serving model selection, server host, and
server port are configurable. A deployment serves one configured model
version. `SERVING_MODEL_SELECTOR` explicitly selects `version` (the default)
or `alias`. Alias deployments resolve and load a concrete version once.

## Getting Started

The project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run quality checks:

```bash
uv run ruff check .
uv run mypy src
```

Start the prediction API using the serving settings from `.env`:

```bash
uv run python -m ml_platform.api
```

The API exposes `GET /health`, `GET /ready`, `GET /model`, and `POST /predict`.
See the [serving workflow](src/ml_platform/serving/README.md) for configuration,
model readiness, metadata, request IDs, and isolated integration tests.

## Running in Docker

The serving API ships as a Docker image. The same image runs in every
environment; all configuration is passed in at runtime.

### Prerequisites

- Docker 23 or newer (BuildKit is used for the build cache).
- An MLflow tracking server or registry that holds the model to serve.
- A free host port (8000 in the examples below).

### Build the image

From the repository root:

```bash
docker build -t ml-platform .
```

The build installs only the dependencies the API needs (no dev tools, DVC or
Kaggle) and produces an image of about 1.2 GB. Dependencies are installed in
their own layer, so rebuilding after a source change is fast. The image runs
as a non-root user (`app`, uid 10001) and contains no `.env` file,
credentials, MLflow data or tests (see `.dockerignore`).

### Configure the container

Settings are read from environment variables. Pass them with `-e` or with an
env file. Start from `.env.example`:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking server |
| `MLFLOW_REGISTRY_URI` | `sqlite:///mlflow.db` | MLflow model registry |
| `SERVING_MODEL_NAME` | `f1-predictor` | Registered model to serve |
| `SERVING_MODEL_SELECTOR` | `version` | `version` or `alias` |
| `SERVING_MODEL_VERSION` | `1` | Version used when the selector is `version` |
| `SERVING_MODEL_ALIAS` | none | Alias used when the selector is `alias` |
| `SERVING_HOST` | `0.0.0.0` | Address the server binds to |
| `SERVING_PORT` | `8000` | Port inside the container |
| `SERVING_ENVIRONMENT` | none | Environment label |

`SERVING_MODEL_ALIAS` only takes effect with `SERVING_MODEL_SELECTOR=alias`.
If the server needs credentials, pass them the same way at runtime (for
example `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` or
`MLFLOW_TRACKING_TOKEN`). Never add them to the image.

#### MLflow connectivity

The default `sqlite:///mlflow.db` points to `/home/app/mlflow.db` inside the
container, which is empty and lost when the container stops. `/ready` returns
503 until the container points at a registry that holds the model. Point the container at a real
registry instead:

- **MLflow server on another host:** `MLFLOW_TRACKING_URI=http://mlflow.example.com:5000`.
- **MLflow server on your machine:** use `http://host.docker.internal:5000`
  and, on Linux, add `--add-host=host.docker.internal:host-gateway` to
  `docker run`.
- **Local SQLite registry:** mount the directory at the same absolute path
  so the artifact paths stored in the registry still resolve, and make it
  writable for uid 10001:

  ```bash
  docker run --rm -p 8000:8000 \
    -v "$PWD:$PWD" \
    -e MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db" \
    -e MLFLOW_REGISTRY_URI="sqlite:///$PWD/mlflow.db" \
    ml-platform
  ```

### Run the container

```bash
docker run -d --name ml-platform \
  --env-file .env \
  -p 8000:8000 \
  ml-platform
```

Serve a model by alias without rebuilding:

```bash
docker run -d --name ml-platform \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e MLFLOW_REGISTRY_URI=http://mlflow:5000 \
  -e SERVING_MODEL_NAME=f1-predictor \
  -e SERVING_MODEL_SELECTOR=alias \
  -e SERVING_MODEL_ALIAS=champion \
  ml-platform
```

`-p HOST:CONTAINER` maps ports. If you change `SERVING_PORT`, map that port,
for example `-e SERVING_PORT=9000 -p 8080:9000`.

The model loads on the first `/ready`, `/model` or `/predict` request and
stays loaded. Moving an alias does not affect a running container; restart it
to pick up the new version.

### Check the service

```bash
curl http://localhost:8000/health   # the process is up
curl http://localhost:8000/ready    # the model is loaded (503 if not)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"feature": 1.0}]}'
```

The image has a Docker health check that calls `/health` every 30 seconds
(after a 30 second start period, 3 retries). It never loads the model or runs
inference. Inspect it with:

```bash
docker ps                                   # STATUS shows (healthy) or (unhealthy)
docker inspect --format '{{json .State.Health}}' ml-platform
```

### Logs, stop and clean up

```bash
docker logs -f ml-platform        # application and request logs
docker stop ml-platform           # stop the container
docker rm ml-platform             # remove it
docker image rm ml-platform       # remove the image
```

### Container tests

`tests/container` builds the image and tests it from outside: health, Docker
health status, runtime configuration, a real prediction, invalid
configuration, the non-root user and the image contents. CI runs them on
every pull request. Locally:

```bash
uv run pytest tests/container
```

Set `ML_PLATFORM_IMAGE=ml-platform` to test an image you already built. The
tests are skipped when no Docker daemon is available, and they always remove
their containers.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Container exits with `Invalid configuration` | A setting has a wrong value, such as a non-numeric `SERVING_PORT` or `SERVING_MODEL_SELECTOR=alias` without `SERVING_MODEL_ALIAS`. The message names the setting. |
| `/health` works but `/ready` returns 503 | The model could not be found or loaded. Check `MLFLOW_TRACKING_URI`, the model name and version or alias, and `docker logs` for `model_resolution_failed` or `model_load_failed`. |
| Connection refused on the host | The port mapping does not match `SERVING_PORT`, or the container stopped (`docker ps -a`). |
| Cannot reach an MLflow server on your machine | `localhost` inside the container is the container itself. Use `host.docker.internal` (see MLflow connectivity). |
| `unable to open database file` with SQLite | The mounted directory is not writable by uid 10001, or the path inside the container differs from the host path. |
| Container shows `(unhealthy)` | The API stopped answering. Check `docker logs` and `docker inspect` for the health check output. |

## Pipelines

Each pipeline owns its own DVC definition. Run a pipeline from its directory:

```bash
cd src/pipelines/f1
uv run dvc repro
```

Use the equivalent command from `src/pipelines/ca_house_prediction` for the
California Housing pipeline.
