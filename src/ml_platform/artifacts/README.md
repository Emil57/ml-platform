# Artifact Management

The `ml_platform.artifacts` module provides a standardized interface for storing, loading, and managing machine learning artifacts across ML projects.

The goal is to prevent individual pipelines from implementing their own model persistence and artifact-directory logic.

Instead, ML projects use the shared `ArtifactManager` component to consistently manage models and metadata.

---

## Responsibilities

The artifact management layer is responsible for:

* Standardizing artifact directory structures.
* Saving trained models.
* Loading previously trained models.
* Managing artifact metadata.
* Creating required artifact directories.
* Providing consistent artifact paths across ML projects.
* Integrating with the platform's centralized configuration.

The artifact layer does **not** decide how a model is trained or evaluated.

Those responsibilities belong to:

* `ml_platform.training`
* `ml_platform.evaluation`

---

## Architecture

```text
                    ML Pipeline
                         │
                         ▼
                ┌─────────────────┐
                │  ArtifactManager │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Model Artifact          Metadata
              │                     │
              ▼                     ▼
        models/model.pkl       metadata.json
```

The overall platform workflow is:

```text
Data
  │
  ▼
Training
  │
  ▼
Evaluation
  │
  ▼
ArtifactManager
  │
  ├── Model
  └── Metadata
```

---

# Package Structure

```text
src/
└── ml_platform/
    └── artifacts/
        ├── __init__.py
        ├── manager.py
        ├── metadata.py
        ├── test_manager.py
        └── test_metadata.py
```

### `manager.py`

Contains the `ArtifactManager` class.

Responsible for:

* Model persistence.
* Model loading.
* Metadata persistence.
* Artifact directory management.

### `metadata.py`

Contains the `ArtifactMetadata` structure used to represent metadata associated with an artifact.

### `__init__.py`

Exports the public artifact-management API:

```python
from ml_platform.artifacts import ArtifactManager
from ml_platform.artifacts import ArtifactMetadata
```

This prevents consumers from depending directly on internal implementation modules.

---

# Artifact Directory Structure

Artifacts are stored under the repository-level `artifacts/` directory.

Each ML project receives its own artifact namespace.

```text
artifacts/
│
├── f1/
│   ├── data/
│   │   ├── raw/
│   │   ├── prepared/
│   │   └── featured/
│   │
│   ├── models/
│   │   └── model.pkl
│   │
│   └── metrics/
│       └── metrics.json
│
└── ca_house_prediction/
    ├── data/
    │   ├── raw/
    │   └── prepared/
    │
    ├── models/
    │   └── model.pkl
    │
    └── metrics/
        └── metrics.json
```

The artifact manager is primarily responsible for the model and metadata portions of this structure.

---

# Configuration

Artifact locations are controlled by the platform configuration rather than by individual pipelines.

The platform defines a centralized artifact root:

```python
from ml_platform.config import settings

print(settings.artifacts_dir)
```

The artifact root should resolve to the repository-level `artifacts/` directory.

This is important because pipelines can be executed from different working directories.

For example:

```text
machine-learning/
├── artifacts/
└── src/
    └── pipelines/
        └── f1/
```

The artifact manager should always resolve artifacts to:

```text
machine-learning/artifacts/
```

rather than:

```text
src/pipelines/f1/artifacts/
```

Using a centralized configuration prevents relative-path problems when running pipelines through DVC.

---

# ArtifactManager

`ArtifactManager` provides the primary interface for artifact persistence.

Example:

```python
from ml_platform.artifacts import ArtifactManager

artifact_manager = ArtifactManager("f1")
```

The project name determines the artifact namespace.

For example:

```python
ArtifactManager("f1")
```

uses:

```text
artifacts/f1/
```

while:

```python
ArtifactManager("ca_house_prediction")
```

uses:

```text
artifacts/ca_house_prediction/
```

---

# Saving Models

A trained model can be persisted using `save_model()`.

```python
from ml_platform.artifacts import ArtifactManager

artifact_manager = ArtifactManager("f1")

model_path = artifact_manager.save_model(model)

print(model_path)
```

The model is saved under:

```text
artifacts/f1/models/model.pkl
```

The manager creates the required directory automatically if it does not exist.

For California Housing:

```python
artifact_manager = ArtifactManager(
    "ca_house_prediction"
)

model_path = artifact_manager.save_model(model)
```

The resulting artifact is:

```text
artifacts/ca_house_prediction/models/model.pkl
```

---

# Loading Models

Previously saved models can be loaded through `load_model()`.

```python
from ml_platform.artifacts import ArtifactManager

artifact_manager = ArtifactManager("f1")

model = artifact_manager.load_model()
```

This allows evaluation or inference pipelines to consume the model without knowing the underlying filesystem implementation.

For example:

```text
Training Pipeline
       │
       ▼
ArtifactManager.save_model()
       │
       ▼
model.pkl
       │
       ▼
ArtifactManager.load_model()
       │
       ▼
Evaluation / Inference
```

A missing model results in a meaningful `FileNotFoundError`.

---

# Metadata

Artifacts can also have associated metadata.

Metadata can contain information such as:

* Project name.
* Artifact type.
* Artifact name.
* Model version.
* Training configuration.
* Random seed.
* Dataset information.
* Creation timestamp.

The metadata structure is represented by `ArtifactMetadata`.

Example:

```python
from ml_platform.artifacts import ArtifactMetadata

metadata = ArtifactMetadata(
    project_name="f1",
    artifact_type="model",
    artifact_name="model.pkl",
)
```

Metadata can be serialized using the model's serialization interface:

```python
data = metadata.model_dump()
```

This allows metadata to be persisted alongside the artifact.

---

# Why Metadata Matters

A model file by itself does not provide enough information to reproduce or understand a training run.

For example:

```text
model.pkl
```

does not tell us:

```text
Which dataset was used?
Which configuration was used?
Which model version is this?
Which random seed was used?
When was it trained?
Which experiment produced it?
```

Metadata provides this context.

Conceptually:

```text
model.pkl
    +
metadata.json
    │
    ▼
Reproducible Model Artifact
```

This becomes especially important as the platform introduces experiment tracking and model versioning.

---

# Integration with Training

The artifact manager is intentionally separate from the training framework.

The responsibilities are:

| Component         | Responsibility                 |
| ----------------- | ------------------------------ |
| `Trainer`         | Train models                   |
| `Evaluator`       | Evaluate models                |
| `ArtifactManager` | Persist and retrieve artifacts |

A pipeline therefore follows:

```python
from ml_platform.artifacts import ArtifactManager
from ml_platform.training import Trainer

trainer = Trainer(model)

trainer.train(
    X_train,
    y_train,
)

artifact_manager = ArtifactManager(
    "my_project"
)

artifact_manager.save_model(model)
```

This separation keeps the training framework generic and prevents it from becoming responsible for project-specific storage concerns.

---

# Integration with Evaluation

Evaluation pipelines can retrieve the trained model using the artifact manager:

```python
from ml_platform.artifacts import ArtifactManager

artifact_manager = ArtifactManager(
    "ca_house_prediction"
)

model = artifact_manager.load_model()
```

The evaluator can then operate on the loaded model:

```python
from ml_platform.evaluation import Evaluator

evaluator = Evaluator(model)

result = evaluator.evaluate(
    X_test,
    y_test,
)
```

This creates a standardized workflow:

```text
Training
   │
   ▼
ArtifactManager
   │
   ▼
model.pkl
   │
   ▼
ArtifactManager
   │
   ▼
Evaluator
   │
   ▼
metrics.json
```

---

# Integration with DVC

DVC remains responsible for versioning and tracking generated artifacts.

`ArtifactManager` and DVC therefore have different responsibilities.

### ArtifactManager

Responsible for:

* Where artifacts are stored.
* Saving models.
* Loading models.
* Managing artifact metadata.

### DVC

Responsible for:

* Tracking artifact files.
* Defining pipeline dependencies.
* Reproducing pipeline stages.
* Versioning generated datasets and models.
* Tracking metrics.

For example:

```text
Python Pipeline
      │
      ▼
ArtifactManager
      │
      ▼
artifacts/f1/models/model.pkl
      │
      ▼
DVC
      │
      ▼
Tracked Artifact
```

The DVC pipeline can therefore declare:

```yaml
outs:
  - ../../../artifacts/f1/models/model.pkl
```

while Python code simply uses:

```python
artifact_manager = ArtifactManager("f1")

artifact_manager.save_model(model)
```

The pipeline does not need to know the implementation details of artifact storage.

---

# Current Consumers

The artifact management framework is currently consumed by:

### F1

```text
src/pipelines/f1/
```

The trained calibrated classifier is stored as:

```text
artifacts/f1/models/model.pkl
```

### California Housing

```text
src/pipelines/ca_house_prediction/
```

The trained regression model is stored as:

```text
artifacts/ca_house_prediction/models/model.pkl
```

Both pipelines use the shared artifact management component instead of implementing their own model persistence logic.

---

# Testing

The artifact management module has dedicated unit tests.

```text
src/ml_platform/artifacts/
├── test_manager.py
└── test_metadata.py
```

The tests cover:

* Model saving.
* Model loading.
* Model directory creation.
* Missing model handling.
* Metadata creation.
* Metadata serialization.

Run the artifact tests with:

```bash
uv run pytest src/ml_platform/artifacts -v
```

Run the complete test suite with:

```bash
uv run pytest
```

---

# Error Handling

Artifact operations should provide meaningful errors when an operation cannot be completed.

For example, attempting to load a model that does not exist should raise:

```python
FileNotFoundError
```

rather than allowing a lower-level filesystem error to propagate without context.

This allows pipelines and applications to distinguish artifact-related failures from training or evaluation failures.

---

# Design Principles

The artifact management layer follows several principles.

### Centralized

Artifact paths are managed by the platform rather than duplicated across individual projects.

### Reusable

The same API can be used by any ML project:

```python
ArtifactManager("project_name")
```

### Configuration-driven

The artifact root comes from centralized platform configuration.

### Separation of concerns

Training, evaluation, and artifact persistence remain separate responsibilities.

### Reproducibility

Artifacts are stored in predictable locations and can be tracked by DVC.

### Extensibility

The current implementation focuses on model persistence and metadata, but the architecture can later support:

* Multiple model versions.
* Dataset metadata.
* Feature artifacts.
* Preprocessing pipelines.
* Model signatures.
* Experiment identifiers.
* Cloud artifact storage.
* Artifact registries.

---

# Future Architecture

The artifact management layer can eventually evolve into a broader artifact registry:

```text
                  ArtifactManager
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
        Models       Datasets      Metadata
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                 Artifact Registry
```

This provides the foundation for future capabilities such as model versioning and experiment tracking.

---

# ML-011 Status

**Completed**

Implemented:

* `ArtifactManager`
* Standard artifact directory structure
* Model saving
* Model loading
* Metadata management
* Centralized artifact configuration
* Artifact unit tests
* F1 pipeline integration
* California Housing pipeline integration
* DVC pipeline integration

The artifact management layer is now available as a reusable platform component for future ML projects.
