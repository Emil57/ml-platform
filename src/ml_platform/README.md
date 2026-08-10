# ML Platform

The `ml_platform` package contains reusable infrastructure shared across machine learning projects.

The platform is designed to separate common ML engineering concerns from model-specific implementation.

## Purpose

The platform provides standardized components for:

* Configuration
* Data management
* Training
* Exception handling
* Logging
* Model persistence
* Testing

Individual ML pipelines consume these components instead of implementing the same functionality independently.

## Architecture

```text
                         ml_platform
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
       config              data              training
          │                   │                   │
          │            ┌──────┼──────┐            │
          │            │      │      │            │
          │            ▼      ▼      ▼            │
          │          source loader validator      │
          │                         │             │
          │                      splitter         │
          │                                       │
          └───────────────┬───────────────────────┘
                          │
                          ▼
                     ML Pipelines
```

## Package Structure

```text
src/ml_platform/
├── __init__.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   ├── validator.py
│   ├── sources/
│   └── tests/
│
├── exceptions/
│   └── __init__.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── test_trainer.py
│
└── utils/
    └── ...
```

## Components

### Configuration

Centralizes application configuration using `pydantic-settings`.

See the configuration implementation for environment variables and defaults.

### Data

Provides reusable dataset acquisition, loading, validation, and splitting functionality.

See the [Data Management Documentation](data/README.md).

### Exceptions

Provides centralized exceptions shared across platform components.

Examples include:

* Dataset-related exceptions
* Configuration exceptions
* Training exceptions
* Prediction exceptions

### Training

Provides the reusable `Trainer` component for standardized model training and model persistence.

See the [Training Framework Documentation](training/README.md).

### Utilities

Contains reusable platform utilities such as centralized logging.

## Design Principles

### Reusability

Common functionality belongs in the platform and should not be duplicated across pipelines.

### Separation of Concerns

The platform provides infrastructure while pipelines provide model-specific logic.

```text
ml_platform
    ↓
Reusable infrastructure

pipelines
    ↓
Model-specific implementation
```

### Reproducibility

Platform components use centralized configuration and deterministic behavior where appropriate.

### Testability

Platform components are independently testable and should avoid unnecessary coupling to individual ML projects.

### Extensibility

Platform abstractions should make it possible to introduce new implementations without modifying existing consumers.

## Current Components

| Component            | Status      |
| -------------------- | ----------- |
| Configuration        | Implemented |
| Exceptions           | Implemented |
| Centralized Logging  | Implemented |
| Data Management      | Implemented |
| Training Framework   | Implemented |
| Evaluation Framework | Planned     |
| Model Registry       | Planned     |
| Experiment Tracking  | Planned     |

## Adding a New Platform Component

New reusable functionality should generally follow this structure:

```text
src/ml_platform/<component>/
├── __init__.py
├── implementation.py
├── tests/
└── README.md
```

The component should:

1. Have a clearly defined responsibility.
2. Avoid model-specific logic.
3. Provide a reusable public interface.
4. Integrate with shared configuration and logging where appropriate.
5. Include unit tests.
6. Document its public API and design decisions.
