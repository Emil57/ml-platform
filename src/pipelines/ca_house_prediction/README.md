# California Housing Prediction Pipeline

The California Housing pipeline is a regression machine learning project that predicts median house values using California housing data.

## Objective

Predict:

```text
MedHouseVal
```

using available housing features.

The current implementation uses median income (`MedInc`) as the primary model feature.

## Dataset

The pipeline uses the California Housing Prices dataset from Kaggle.

The dataset is acquired through the shared `KaggleSource` component.

## Pipeline

```text
Kaggle Dataset
      │
      ▼
get_data.py
      │
      ▼
Raw Data
      │
      ▼
DataLoader
      │
      ▼
DataValidator
      │
      ▼
DataSplitter
      │
      ├── Train
      ├── Validation
      └── Test
      │
      ▼
Trainer
      │
      ▼
model.pkl
      │
      ▼
evaluate_model.py
      │
      ▼
metrics.json
```

## Data Preparation

The preparation stage:

1. Loads the raw housing dataset.
2. Renames dataset columns to standardized feature names.
3. Selects the model features.
4. Separates features and target.
5. Splits the dataset into training, validation, and test partitions.
6. Saves the resulting datasets as artifacts.

## Features

The dataset contains several housing-related variables.

The current model uses:

```text
MedInc
```

as the input feature.

The target is:

```text
MedHouseVal
```

## Data Splitting

The shared `DataSplitter` component is responsible for creating:

```text
Train
Validation
Test
```

The split uses the centralized platform random seed to provide reproducible results.

## Model

The pipeline currently uses:

```python
LinearRegression()
```

The model is trained through the shared `ml_platform.training.Trainer`.

Example:

```python
from ml_platform.training import Trainer

trainer = Trainer(model)

trainer.train(
    X_train=X_train,
    y_train=y_train,
)

trainer.save(model_path)
```

## Evaluation

The model is evaluated using:

* Mean Squared Error
* R² Score

Evaluation metrics are stored in:

```text
artifacts/ca_house_prediction/metrics/metrics.json
```

## Artifacts

```text
artifacts/ca_house_prediction/
├── data/
│   ├── raw/
│   │   └── housing.csv
│   │
│   └── prepared/
│       ├── X_train.csv
│       ├── X_validation.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_validation.csv
│       └── y_test.csv
│
├── models/
│   └── model.pkl
│
└── metrics/
    └── metrics.json
```

## DVC Pipeline

The workflow is orchestrated through DVC.

```text
get_data
    ↓
prepare
    ↓
train
    ↓
evaluate
```

Run the pipeline:

```bash
cd src/pipelines/ca_house_prediction
uv run dvc repro
```

Check pipeline status:

```bash
uv run dvc status
```

## Reproducibility

DVC tracks:

* Pipeline dependencies
* Dataset outputs
* Model outputs
* Evaluation metrics

The shared platform configuration also provides a centralized random seed.

## Limitations

The current implementation uses a simple linear regression model and a limited feature set.

It is primarily intended as a reference implementation demonstrating how a model-specific pipeline consumes the shared ML platform.

## Future Improvements

Potential improvements include:

* Additional housing features
* More sophisticated preprocessing
* Feature engineering
* Multiple regression algorithms
* Hyperparameter optimization
* Cross-validation
* Experiment tracking
* Model comparison
