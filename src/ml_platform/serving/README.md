# Serving

The serving engine connects registered MLflow models to the FastAPI prediction API. One deployment loads and retains one concrete model version.

It provides a framework-independent interface between the prediction API, model resolution, model loading, and model execution.

## Architecture

```text
Prediction Request
        │
        ▼
PredictionService
        │
        ▼
ModelResolver
        │
        ▼
ModelLoader
        │
        ▼
Predictor
        │
        ▼
Model
```

Each component has a single responsibility:

| Component           | Responsibility                         |
| ------------------- | -------------------------------------- |
| `PredictionService` | Orchestrates the prediction workflow   |
| `ModelResolver`     | Identifies the requested model/version |
| `ModelLoader`       | Loads the resolved model               |
| `Predictor`         | Executes inference                     |

This separation keeps the serving layer independent from ML frameworks and model storage implementations.

## Contracts

### `PredictionRequest`

Defines the input to a prediction operation:

```text
inputs
```

A request contains a nonempty list of input dictionaries. Model identity comes from deployment settings.

### `PredictionResponse`

Defines the result of a prediction:

```text
model_name
model_version
predictions
request_id
```

### `Predictor`

The model execution contract:

```python
predict(inputs) -> predictions
```

### `ModelResolver`

The model identification contract:

```python
resolve(reference) -> ResolvedModel(name, version, alias, uri)
```

### `ModelLoader`

The model loading contract:

```python
load(model_uri) -> predictor
```

### `PredictionService`

The orchestration contract:

```python
predict(request, request_id=None) -> response
```

## Model Selection

Set `SERVING_MODEL_SELECTOR=version` (the default) with:

```text
name + version
```

for deterministic model selection. Set `SERVING_MODEL_SELECTOR=alias` with:

```text
name + alias
```

for lifecycle selection such as `champion`. Alias selection requires a nonempty alias. The inactive selector setting is ignored.

The resolver returns concrete version metadata and a version-specific loading URI. The first readiness, metadata, or prediction request loads the model. Successful loads are cached per process; failures can be retried. Loading is synchronized across concurrent requests. Moving an alias does not change an already loaded deployment; restart to pick up the new version.

## Error Handling

Serving-specific exceptions are defined in `exceptions.py`:

```text
ServingError
├── ModelNotFoundError
├── ModelLoadError
├── InvalidPredictionInputError
└── PredictionError
```

These exceptions represent domain-level failures. HTTP-specific error handling belongs to the API layer.

## Observability

Each HTTP request receives an `X-Request-ID` header. Clients may provide the
header; otherwise the API generates a UUID. The same identifier is returned in
the response header and included in serving logs.

Prediction logs contain:

- request ID;
- configured model name, version, and optional alias;
- end-to-end serving latency in milliseconds;
- success or failure status;
- failure event type and exception type when applicable.

Prediction latency includes resolution and loading on a cold request, and inference against the cached model on subsequent requests. Readiness and metadata requests can load the model first. Inputs, predictions, raw exception messages, and chained tracebacks are omitted from application logs because model exceptions may contain sensitive inputs.

## Package Structure

```text
serving/
├── __init__.py
├── schemas.py
├── contracts.py
└── exceptions.py
```

Tests:

```text
tests/serving/
├── test_schemas.py
└── test_contracts.py
```

## Scope

The serving module owns resolution, model loading, prediction, and operational metadata. The API layer owns HTTP endpoints and status codes.

## Complete Serving Workflow

```text
MLflow Registry -> lifecycle alias -> concrete version -> model loading
    -> PredictionService -> FastAPI POST /predict -> prediction response
```

Configure tracking and registry URIs with `MLFLOW_TRACKING_URI` and `MLFLOW_REGISTRY_URI`. Select the model with `SERVING_MODEL_NAME`, `SERVING_MODEL_SELECTOR`, and either `SERVING_MODEL_VERSION` or `SERVING_MODEL_ALIAS`. Start with `uv run python -m ml_platform.api`; host and port come from platform settings.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Process liveness; 200 without accessing MLflow. |
| `GET /ready` | Loads the model; 200 with readiness and metadata, or generic 503. |
| `GET /model` | Loaded model name, concrete version, and selected alias; generic 503 if unavailable. |
| `POST /predict` | Predictions, model name/version, and request ID. |

Metadata excludes artifact locations and credentials. Readiness verifies model availability, not whether every possible input succeeds. Malformed prediction bodies return 422; missing aliases or versions return 404; loading and inference failures return 500. Request ID headers are returned for handled failures as well as successful responses.

Run `uv run pytest src/ml_platform/serving/tests/integration -q` for isolated registry-to-API tests. They use temporary SQLite storage and artifact directories, register two distinct versions, assign an alias, and exercise the real resolver, loader, serving engine, and FastAPI TestClient. No external MLflow server, Uvicorn process, or dataset download is needed. Fixtures restore settings, dependency overrides, cached services, and MLflow URIs.
