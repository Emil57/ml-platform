# Data Management Layer

The `ml_platform.data` package provides reusable infrastructure for acquiring, loading, validating, and splitting datasets.

This component was introduced as part of **ML-008 — Data Management Layer**.

## Objective

Centralize common dataset-management functionality so individual ML projects do not need to implement their own data loading, validation, and splitting logic.

## Architecture

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
               Train  Validation  Test
```

The resulting workflow is:

```text
Data Acquisition
       ↓
Data Loading
       ↓
Data Validation
       ↓
Train / Validation / Test Split
       ↓
Model Pipeline
```

## Package Structure

```text
src/ml_platform/data/
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

## Data Sources

`DataSource` defines the interface for obtaining datasets.

```python
from abc import ABC, abstractmethod
from pathlib import Path


class DataSource(ABC):

    @abstractmethod
    def fetch(self, destination: Path) -> Path:
        raise NotImplementedError
```

This allows the platform to support multiple data providers without coupling downstream components to a specific source.

### Current Sources

* Kaggle
* Local filesystem

Potential future sources:

* Amazon S3
* Google Cloud Storage
* Azure Blob Storage
* Databases

## Kaggle Source

`KaggleSource` encapsulates Kaggle API interaction.

```python
from pathlib import Path

from ml_platform.data.sources.kaggle import KaggleSource


source = KaggleSource(
    "owner/dataset-name"
)

source.fetch(
    Path("data/raw")
)
```

Kaggle authentication and download logic remain inside `KaggleSource`.

This means pipelines do not need to directly interact with the Kaggle API.

## Data Loader

`DataLoader` loads datasets from local files.

Example:

```python
from ml_platform.data import DataLoader

loader = DataLoader()

data = loader.load_csv(
    "data/raw/dataset.csv"
)
```

The loader verifies that the requested file exists before loading it.

## Data Validator

`DataValidator` provides reusable data-quality checks.

### Empty Dataset

```python
validator.validate_not_empty(data)
```

### Required Columns

```python
validator.validate_columns(
    data,
    ["feature_1", "feature_2", "target"],
)
```

### Missing Values

```python
validator.validate_no_missing_values(data)
```

These are baseline validations. Project-specific validation rules remain inside the corresponding pipeline.

## Data Splitter

`DataSplitter` creates reproducible training, validation, and test datasets.

```python
from ml_platform.data import DataSplitter

splitter = DataSplitter()

train, validation, test = splitter.split(data)
```

The default split is approximately:

```text
64% → Train
16% → Validation
20% → Test
```

The splitter uses the centralized random seed from platform configuration.

A custom seed can be provided:

```python
train, validation, test = splitter.split(
    data,
    random_state=123,
)
```

## Responsibilities

| Component       | Responsibility                             |
| --------------- | ------------------------------------------ |
| `DataSource`    | Defines data acquisition interface         |
| `KaggleSource`  | Downloads datasets from Kaggle             |
| `DataLoader`    | Loads local datasets                       |
| `DataValidator` | Performs common data-quality checks        |
| `DataSplitter`  | Creates reproducible data partitions       |
| DVC             | Tracks pipeline dependencies and artifacts |

## Separation of Responsibilities

Kaggle and DVC serve different purposes.

```text
Kaggle
  ↓
Data Acquisition

DVC
  ↓
Pipeline Reproducibility
Artifact Tracking
Dependency Tracking
```

Kaggle is therefore not responsible for dataset versioning within the ML pipeline.

## Example

A pipeline can combine the platform components:

```python
from pathlib import Path

from ml_platform.data import (
    DataLoader,
    DataSplitter,
    DataValidator,
)
from ml_platform.data.sources.kaggle import KaggleSource


source = KaggleSource("owner/dataset-name")
source.fetch(Path("data/raw"))

loader = DataLoader()
data = loader.load_csv("data/raw/dataset.csv")

validator = DataValidator()

validator.validate_not_empty(data)
validator.validate_columns(
    data,
    ["feature_1", "feature_2", "target"],
)
validator.validate_no_missing_values(data)

splitter = DataSplitter()

train, validation, test = splitter.split(data)
```

## Testing

The data-management layer has dedicated tests:

```text
tests/
├── test_loader.py
├── test_splitter.py
└── test_validator.py
```

Run them with:

```bash
uv run pytest src/ml_platform/data -v
```

## Design Decisions

### Why abstract data sources?

The platform should not depend directly on Kaggle.

The abstraction allows future sources to be introduced without modifying `DataLoader`, `DataValidator`, or `DataSplitter`.

### Why separate acquisition from loading?

Downloading data and reading data are different responsibilities.

```text
DataSource
    ↓
Acquires data

DataLoader
    ↓
Reads data
```

This allows locally available data to use the same loading and validation workflow.

## Future Improvements

Potential future improvements include:

* Dataset schemas
* Additional data-quality rules
* Dataset metadata
* Data profiling
* Cloud storage sources
* Additional file formats
* Dataset version metadata
