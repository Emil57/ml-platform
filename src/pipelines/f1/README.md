# Formula 1 Prediction Pipeline

The Formula 1 pipeline is a classification machine learning project that predicts whether a driver will win a Formula 1 race.

## Objective

Predict:

```text
won = 1
```

when the driver wins the race, and:

```text
won = 0
```

otherwise.

## Dataset

The pipeline uses the Formula 1 World Championship dataset from Kaggle:

```text
rohanrao/formula-1-world-championship-1950-2020
```

The dataset contains historical Formula 1 information including:

* Races
* Drivers
* Constructors
* Results
* Qualifying
* Circuits
* Constructor results
* Other historical race information

Dataset acquisition is handled through the shared `KaggleSource` component.

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
prepare_data.py
      │
      ▼
Prepared Data
      │
      ▼
feature_engineering.py
      │
      ├── train.csv
      ├── valid.csv
      └── test.csv
      │
      ▼
train_model.py
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

1. Loads the raw Formula 1 datasets.
2. Joins the relevant datasets.
3. Converts race dates.
4. Sorts historical observations.
5. Produces a prepared dataset.

## Feature Engineering

The feature-engineering stage creates historical driver and constructor features.

Examples include:

* Recent driver points
* Recent finishing position
* Recent podiums
* Recent DNFs
* Recent constructor points
* Recent constructor finishing position
* Recent constructor podiums
* Recent constructor DNFs
* Grid position
* Driver
* Constructor
* Circuit
* Race round

Historical rolling features use previous races to avoid using future information.

## Data Leakage

The pipeline intentionally avoids using post-race information when creating predictive features.

Features such as final race status, final race time, and completed laps are excluded when they would introduce target leakage.

## Dataset Splitting

The F1 pipeline uses a time-based split.

```text
Training
≤ 2016

Validation
2017–2019

Test
2020
```

This better represents the real-world prediction scenario because the model is trained on historical races and evaluated on future races.

## Model

The pipeline uses an `SGDClassifier` with logistic loss.

The classifier is configured to account for the imbalance between winners and non-winners.

Probability calibration is applied using a sigmoid calibration approach.

## Training

The pipeline uses the shared platform `Trainer` component.

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

* Accuracy
* F1 Score
* Log Loss
* Brier Score
* ROC AUC

Metrics are saved to:

```text
artifacts/f1/metrics/metrics.json
```

## Artifacts

```text
artifacts/f1/
├── data/
│   ├── raw/
│   ├── prepared/
│   └── featured/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
│
├── models/
│   └── model.pkl
│
└── metrics/
    └── metrics.json
```

## DVC Pipeline

The pipeline is orchestrated using DVC.

```text
get_data
    ↓
prepare
    ↓
feature_engineering
    ↓
train
    ↓
evaluate
```

Run the pipeline:

```bash
cd src/pipelines/f1
uv run dvc repro
```

Check pipeline status:

```bash
uv run dvc status
```

## Reproducibility

The pipeline uses DVC to track stage dependencies and outputs.

This allows the complete workflow to be reproduced when inputs or pipeline code change.

## Limitations

Current limitations include:

* Historical dataset ends in 2020.
* The model is focused on race-win prediction.
* Feature engineering is limited to currently available historical features.
* The model does not currently represent all real-world race conditions.

## Future Improvements

Potential improvements include:

* Updated Formula 1 data
* Additional race features
* Weather information
* Practice-session information
* Better handling of categorical variables
* Hyperparameter optimization
* Experiment tracking
* Model comparison
