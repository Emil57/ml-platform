# machine-learning
Playground for Machine Learning and MLOps


# Target Architecture

```
machine-learning/
├── .github/
├── .dvc/
├── docs/
├── platform/
│   ├── api/
│   ├── orchestrator/
│   ├── registry/
│   ├── deployment/
│   └── monitoring/
├── src/
│   ├── common/
│   │   ├── logging/
│   │   ├── metrics/
│   │   ├── preprocessing/
│   │   ├── validation/
│   │   ├── exceptions/
│   │   └── utils/
│   │
│   ├── model1/
│   └── model2/
│
├── tests/
└── pyproject.toml
```
## Platform Architecture

The repository is organized around a reusable Machine Learning Platform rather than individual ML projects.

The `ml_platform` package contains reusable infrastructure that will be shared across future machine learning applications.

Current platform modules include:

- Configuration
- Data
- Models
- Training
- Evaluation
- Utilities

Additional platform capabilities will be introduced incrementally throughout the project roadmap.


## Configuration

The platform uses `pydantic-settings` to centralize configuration.

Configuration values can be supplied through environment variables with the `ML_` prefix.

Example:

```bash
ML_DEBUG=true
ML_RANDOM_SEED=123
```

Applications should import the shared configuration instance:

```python
from ml_platform.config import settings

print(settings.environment)
```

## Data Management Layer

The `ml_platform.data` package provides reusable components for acquiring, loading, validating, and splitting datasets across machine learning projects.

The goal of this layer is to centralize common data-management functionality so that individual ML projects do not need to implement their own dataset loading, validation, and splitting logic.

### Architecture

```text
                    ml_platform.data
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
        DataLoader    DataValidator  DataSplitter
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                      ML Pipeline
```

Data acquisition is separated from data loading:

```text
                    Data Sources
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
             Kaggle             Local
                │                 │
                └────────┬────────┘
                         ▼
                    DataLoader
                         │
                         ▼
                  DataValidator
                         │
                         ▼
                    DataSplitter
                    /    |     \
                   /     |      \
                train   validation  test
```

### Package Structure

```text
src/
└── ml_platform/
    └── data/
        ├── __init__.py
        ├── loader.py
        ├── splitter.py
        ├── validator.py
        │
        ├── sources/
        │   ├── __init__.py
        │   ├── base.py
        │   └── kaggle.py
        │
        └── tests/
            ├── test_loader.py
            ├── test_splitter.py
            └── test_validator.py
```

### Data Sources

The `sources` package provides an abstraction for acquiring datasets from external or local sources.

The base `DataSource` interface defines the contract that data sources must implement:

```python
from abc import ABC, abstractmethod
from pathlib import Path


class DataSource(ABC):
    """Interface for dataset sources."""

    @abstractmethod
    def fetch(self, destination: Path) -> Path:
        """Fetch data and return the local path."""
        raise NotImplementedError
```

This allows different data providers to be implemented without coupling the rest of the platform to a specific provider.

Currently supported sources include:

* Kaggle
* Local filesystem

Additional sources such as cloud object storage or databases can be added in the future without changing the rest of the data-management layer.

### Kaggle Data Source

The Kaggle source encapsulates communication with the Kaggle API.

Example:

```python
from pathlib import Path

from ml_platform.data.sources.kaggle import KaggleSource


source = KaggleSource(
    "owner/dataset-name"
)

data_path = source.fetch(
    Path("data/raw")
)
```

The Kaggle-specific authentication and download logic remains inside the source implementation rather than being duplicated across individual ML projects.

### Data Loader

`DataLoader` is responsible for loading datasets from local files into pandas DataFrames.

For example:

```python
from ml_platform.data import DataLoader


loader = DataLoader()

data = loader.load_csv(
    "data/raw/dataset.csv"
)
```

The loader validates that the requested file exists before attempting to read it.

A missing dataset results in a `FileNotFoundError`.

### Data Validator

`DataValidator` provides reusable checks that can be performed before data enters the ML pipeline.

#### Empty datasets

```python
validator.validate_not_empty(data)
```

Raises a `ValueError` when the dataset contains no rows.

#### Required columns

```python
validator.validate_columns(
    data,
    ["feature_1", "feature_2", "target"],
)
```

Raises a `ValueError` when one or more required columns are missing.

#### Missing values

```python
validator.validate_no_missing_values(data)
```

Raises a `ValueError` when missing values are detected.

These validations provide a common baseline for data quality checks while keeping project-specific validation rules inside the individual ML projects.

### Data Splitter

`DataSplitter` provides a standardized way of splitting datasets into training, validation, and test sets.

Example:

```python
from ml_platform.data import DataSplitter


splitter = DataSplitter()

train, validation, test = splitter.split(
    data,
)
```

The default configuration produces approximately:

```text
64% → Training
16% → Validation
20% → Test
```

The splitter uses the centralized `settings.random_seed` value by default to ensure reproducibility across the platform.

A custom seed can still be provided when required for experimentation:

```python
train, validation, test = splitter.split(
    data,
    random_state=123,
)
```

### Typical Data Pipeline

An ML project can combine these components into a standard data-management workflow:

```python
from pathlib import Path

from ml_platform.data import (
    DataLoader,
    DataSplitter,
    DataValidator,
)
from ml_platform.data.sources.kaggle import KaggleSource


# 1. Acquire data
source = KaggleSource("owner/dataset-name")
source.fetch(Path("data/raw"))


# 2. Load data
loader = DataLoader()
data = loader.load_csv("data/raw/dataset.csv")


# 3. Validate data
validator = DataValidator()

validator.validate_not_empty(data)

validator.validate_columns(
    data,
    ["feature_1", "feature_2", "target"],
)

validator.validate_no_missing_values(data)


# 4. Split data
splitter = DataSplitter()

train, validation, test = splitter.split(data)
```

This establishes a standardized flow:

```text
Data Acquisition
       ↓
Data Loading
       ↓
Data Validation
       ↓
Train / Validation / Test Split
       ↓
Model Training
```

### Separation of Responsibilities

The data-management layer intentionally separates several responsibilities:

| Component       | Responsibility                                      |
| --------------- | --------------------------------------------------- |
| `DataSource`    | Defines the interface for obtaining datasets        |
| `KaggleSource`  | Downloads datasets from Kaggle                      |
| `DataLoader`    | Loads local dataset files into DataFrames           |
| `DataValidator` | Performs common data-quality checks                 |
| `DataSplitter`  | Creates reproducible train/validation/test datasets |
| DVC             | Handles dataset versioning and reproducibility      |

Kaggle is therefore treated as a **data acquisition mechanism**, while DVC remains responsible for **dataset versioning**.

This separation allows an ML project to change its data source without changing its training or validation logic.

### Testing

Each component of the data-management layer has dedicated unit tests.

```text
src/ml_platform/data/tests/
├── test_loader.py
├── test_splitter.py
└── test_validator.py
```

The tests cover:

* Successful CSV loading
* Missing dataset files
* Empty datasets
* Missing required columns
* Missing values
* Dataset splitting
* Preservation of all dataset rows
* Reproducibility of dataset splits

The complete test suite can be executed with:

```bash
uv run pytest
```

Code quality checks can be executed with:

```bash
uv run ruff check .
uv run black --check .
uv run mypy src/
```

### Design Direction

The current implementation establishes the foundation for a reusable data-management layer.

Future data sources can be added through the `DataSource` abstraction:

```text
DataSource
├── KaggleSource
├── LocalSource
├── S3Source
├── GCSDataSource
└── DatabaseSource
```

Future iterations may also introduce:

* Additional data-quality checks
* Dataset schemas
* Dataset metadata
* Cloud storage integrations
* Data profiling
* Data versioning metadata
* Additional file formats

These additions can be implemented without coupling individual ML projects to the underlying data infrastructure.

---

## ML-008 — Data Management Layer

### Objective

Create a reusable data-management layer that centralizes dataset acquisition, loading, validation, and splitting across ML projects.

### Implemented

* `DataSource` abstraction for dataset acquisition.
* Kaggle data-source integration.
* CSV data loading through `DataLoader`.
* Dataset validation through `DataValidator`.
* Train/validation/test splitting through `DataSplitter`.
* Centralized random seed integration through `ml_platform.config.settings`.
* Unit tests for data loading, validation, and splitting.
* Package-level exports through `ml_platform.data`.

### Architecture

```text
Data Source
    ↓
DataLoader
    ↓
DataValidator
    ↓
DataSplitter
    ↓
ML Training Pipeline
```

Kaggle is treated as a **data-acquisition mechanism**, while DVC remains responsible for **dataset versioning**.

### Status

**In Progress — Integration Pending**

The data-management components and unit tests have been implemented. The next step is to integrate the layer into an existing example ML project and replace its current dataset-management logic with the shared `ml_platform.data` components.

