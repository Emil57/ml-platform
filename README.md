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
version. `SERVING_MODEL_ALIAS` is available for a future alias-based rollout;
when a version is configured, the version takes precedence.

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

The API exposes `GET /health` and `POST /predict`. Refer to the serving and
API source packages for the request and response contracts.

## Pipelines

Each pipeline owns its own DVC definition. Run a pipeline from its directory:

```bash
cd src/pipelines/f1
uv run dvc repro
```

Use the equivalent command from `src/pipelines/ca_house_prediction` for the
California Housing pipeline.
