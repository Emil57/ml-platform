# System Architecture

## Overview

The Machine Learning Platform follows a layered architecture that separates reusable ML infrastructure from model-specific pipelines.

```text
                    Machine Learning Platform
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
        ml_platform                        pipelines
             │                                 │
     ┌───────┼────────┐               ┌────────┴────────┐
     │       │        │               │                 │
     ▼       ▼        ▼               ▼                 ▼
   Data   Training  Config            F1          California
     │       │        │
     └───────┼────────┘
             │
             ▼
       Shared Services
             │
             ▼
          Artifacts
```

## Layers

### Platform Layer

Located under:

```text
src/ml_platform/
```

Contains reusable infrastructure.

### Pipeline Layer

Located under:

```text
src/pipelines/
```

Contains model-specific workflows.

### Artifact Layer

Located under:

```text
artifacts/
```

Contains generated datasets, trained models, and evaluation metrics.

### Orchestration Layer

DVC pipelines are defined by:

```text
src/pipelines/*/dvc.yaml
```

DVC connects pipeline stages and tracks dependencies and outputs.

## Separation of Concerns

The platform should provide reusable infrastructure.

```text
Platform
    ↓
How ML work is performed

Pipeline
    ↓
What ML problem is being solved
```

For example:

The platform provides `Trainer`.

The F1 pipeline decides to use `SGDClassifier`.

The California Housing pipeline decides to use `LinearRegression`.

The platform should not contain those model-specific decisions.

## Data Flow

```text
External Data Source
        ↓
Data Acquisition
        ↓
Raw Dataset
        ↓
Data Loading
        ↓
Validation
        ↓
Splitting
        ↓
Feature Engineering
        ↓
Training
        ↓
Model Artifact
        ↓
Evaluation
        ↓
Metrics
```

## Reproducibility

Reproducibility is supported through:

* Centralized random seeds
* DVC pipeline tracking
* Version-controlled source code
* Deterministic data splitting
* Tracked model artifacts
* Tracked evaluation metrics

## Extensibility

The architecture is designed so that new ML projects can reuse the existing platform.

For example:

```text
New ML Project
      │
      ├── DataSource
      ├── DataLoader
      ├── DataValidator
      ├── DataSplitter
      ├── Trainer
      └── Evaluation
```

The new project should only implement functionality that is specific to its ML problem.

## Architectural Principle

The main design goal is:

> Build reusable ML infrastructure once and consume it across multiple ML projects.
