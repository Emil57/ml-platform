# Serving

The `serving` module defines the contracts and core abstractions required to serve trained machine learning models.

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
model_name
model_version / alias
inputs
```

A request must identify the model using either a specific version or an alias.

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
resolve(reference) -> predictor/model target
```

### `ModelLoader`

The model loading contract:

```python
load(model_uri) -> predictor
```

### `PredictionService`

The orchestration contract:

```python
predict(request) -> response
```

## Model Selection

Models can be requested using either:

```text
name + version
```

for deterministic model selection, or:

```text
name + alias
```

for lifecycle-based selection such as `production`.

A request cannot specify both a version and an alias.

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

This module defines **what the serving system must do**, not how models are served in production.

Concrete implementations such as MLflow model loading, prediction services, HTTP endpoints, caching, deployment, and observability are implemented in subsequent serving stories.
