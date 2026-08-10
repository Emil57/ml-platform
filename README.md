# Machine Learning Platform

A production-oriented Machine Learning Platform that evolves from individual ML experiments into a reusable, testable, and reproducible ML system.

The project separates reusable ML infrastructure from model-specific pipelines.

## Architecture

```text
machine-learning/
│
├── artifacts/
│   ├── f1/
│   └── ca_house_prediction/
│
├── src/
│   ├── ml_platform/
│   │   ├── config/
│   │   ├── data/
│   │   ├── exceptions/
│   │   ├── training/
│   │   └── utils/
│   │
│   └── pipelines/
│       ├── f1/
│       └── ca_house_prediction/
│
├── docs/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Platform

The `ml_platform` package contains reusable infrastructure shared across ML projects.

Current components:

* Configuration management
* Centralized exceptions
* Centralized logging
* Data management
* Training framework
* Model persistence
* Testing utilities

See the [Platform Documentation](src/ml_platform/README.md).

## Data Management

The shared data-management layer provides:

* Data source abstractions
* Kaggle integration
* Dataset loading
* Data validation
* Train/validation/test splitting

See the [Data Management Documentation](src/ml_platform/data/README.md).

## Training Framework

The shared training framework provides:

* Standardized model training
* Centralized logging
* Training error handling
* Model persistence

See the [Training Framework Documentation](src/ml_platform/training/README.md).

## Example Pipelines

### Formula 1

A classification pipeline that predicts whether a Formula 1 driver will win a race.

See the [F1 Pipeline Documentation](src/pipelines/f1/README.md).

### California Housing

A regression pipeline that predicts California housing prices.

See the [California Housing Documentation](src/pipelines/ca_house_prediction/README.md).

## DVC

Each ML pipeline has its own DVC pipeline.

```bash
cd src/pipelines/f1
uv run dvc repro
```

or:

```bash
cd src/pipelines/ca_house_prediction
uv run dvc repro
```

See the pipeline-specific documentation for more details.

## Development

The project uses `uv` for environment and dependency management.

```bash
uv sync
```

Run the complete test suite:

```bash
uv run pytest
```

Run code-quality checks:

```bash
uv run ruff check .
uv run black --check .
uv run mypy src/
```

More information is available in the [Development Documentation](docs/development.md).

## Documentation

| Document                                                          | Description                                 |
| ----------------------------------------------------------------- | ------------------------------------------- |
| [Platform](src/ml_platform/README.md)                             | Platform architecture and shared components |
| [Data Management](src/ml_platform/data/README.md)                 | ML-008 data-management layer                |
| [Training Framework](src/ml_platform/training/README.md)          | ML-009 training framework                   |
| [F1 Pipeline](src/pipelines/f1/README.md)                         | F1 model documentation                      |
| [California Housing](src/pipelines/ca_house_prediction/README.md) | California Housing model documentation      |
| [Architecture](docs/architecture.md)                              | Overall system architecture                 |
| [Development](docs/development.md)                                | Development workflow and tooling            |
| [Roadmap](docs/roadmap.md)                                        | Platform roadmap                            |
