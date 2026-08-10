# Evaluation Framework

The `ml_platform.evaluation` package provides reusable components for evaluating machine learning models using a standardized interface.

The goal is to prevent individual ML projects from duplicating evaluation logic while providing consistent metrics, error handling, and logging across the platform.

## Responsibilities

The evaluation framework is responsible for:

* Evaluating trained machine learning models.
* Supporting classification and regression metrics.
* Providing a consistent evaluation interface.
* Handling evaluation failures through platform exceptions.
* Integrating centralized logging.
* Returning standardized evaluation results.
* Allowing individual ML pipelines to define the metrics relevant to their problem.

The framework intentionally does **not** determine which metrics a model should use. Individual pipelines provide the appropriate metrics when calling the evaluator.

---

## Architecture

```text
                    ML Pipeline
                         │
                         ▼
                    Trained Model
                         │
                         ▼
                     Evaluator
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Classification          Regression
          Metrics                 Metrics
              │                     │
              └──────────┬──────────┘
                         ▼
                 Evaluation Results
                         │
                         ▼
                    metrics.json
```

The pipeline remains responsible for selecting the appropriate metrics, while `Evaluator` provides the reusable execution mechanism.

---

## Package Structure

```text
src/
└── ml_platform/
    └── evaluation/
        ├── __init__.py
        ├── evaluator.py
        └── tests/
            └── test_evaluator.py
```

---

## Evaluator

The `Evaluator` class provides the common evaluation workflow.

Example:

```python
from ml_platform.evaluation import Evaluator

evaluator = Evaluator(model)

metrics = evaluator.evaluate(
    X_test,
    y_test,
    metrics={
        "accuracy": accuracy_score,
        "f1_score": f1_score,
    },
)
```

The evaluator:

1. Receives a trained model.
2. Receives evaluation features and targets.
3. Generates predictions.
4. Executes the requested metrics.
5. Logs the evaluation process.
6. Returns the calculated metrics.

---

## Classification

Classification models can use common metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Log Loss
* Brier Score

Example:

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

metrics = evaluator.evaluate(
    X_test,
    y_test,
    metrics={
        "accuracy": accuracy_score,
        "f1_score": f1_score,
        "roc_auc": roc_auc_score,
    },
)
```

The evaluation framework does not restrict which classification metrics can be used. Any compatible metric function can be supplied by the pipeline.

---

## Regression

Regression models can use metrics such as:

* Mean Squared Error
* Mean Absolute Error
* Root Mean Squared Error
* R² Score

Example:

```python
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

metrics = evaluator.evaluate(
    X_test,
    y_test,
    metrics={
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
    },
)
```

This allows regression pipelines to use the same evaluation interface as classification pipelines.

---

## Metric Interface

Metrics are provided as a dictionary:

```python
{
    "metric_name": metric_function,
}
```

For example:

```python
{
    "r2_score": r2_score,
    "mean_squared_error": mean_squared_error,
}
```

The metric name becomes the key in the resulting evaluation output.

This provides a consistent representation regardless of the underlying model type.

---

## Evaluation Results

The evaluator currently returns a dictionary containing the calculated metrics.

Example:

```python
{
    "mean_squared_error": 0.523,
    "r2_score": 0.81,
}
```

The dictionary can be persisted as a JSON artifact:

```python
import json

with metrics_path.open("w") as file:
    json.dump(metrics, file, indent=4)
```

This format is compatible with DVC metrics and allows model performance to be tracked across pipeline executions.

---

## Error Handling

Evaluation failures are handled through the shared ML Platform exception hierarchy.

This prevents individual pipelines from implementing their own evaluation error-handling logic.

For example, an evaluation failure should provide a meaningful error describing the evaluation operation that failed rather than exposing an ambiguous generic exception.

The evaluator should preserve the original exception context when raising a platform-specific error.

---

## Logging

The evaluation framework integrates with the centralized logging system provided by `ml_platform`.

Evaluation events should be logged consistently, including:

```text
Evaluation started
        │
        ▼
Predictions generated
        │
        ▼
Metrics calculated
        │
        ▼
Evaluation completed
```

Evaluation failures should also be logged before the corresponding exception is raised.

This allows pipeline execution logs to provide useful information without requiring individual projects to configure their own logging implementation.

---

## Current Consumers

The evaluation framework is currently consumed by:

### California Housing

The California Housing pipeline evaluates a regression model using:

```text
Mean Squared Error
R² Score
```

Pipeline:

```text
Test Data
    │
    ▼
Trained Linear Regression
    │
    ▼
Evaluator
    │
    ├── Mean Squared Error
    └── R² Score
    │
    ▼
metrics.json
```

### Formula 1

The Formula 1 pipeline evaluates a classification model using metrics including:

```text
Accuracy
F1 Score
Log Loss
Brier Score
ROC AUC
```

Pipeline:

```text
Test Data
    │
    ▼
Trained Classification Model
    │
    ▼
Evaluator
    │
    ├── Accuracy
    ├── F1 Score
    ├── Log Loss
    ├── Brier Score
    └── ROC AUC
    │
    ▼
metrics.json
```

---

## DVC Integration

Evaluation results are stored as DVC metrics.

Example:

```text
artifacts/
└── ca_house_prediction/
    └── metrics/
        └── metrics.json
```

and:

```text
artifacts/
└── f1/
    └── metrics/
        └── metrics.json
```

The DVC pipeline connects model evaluation to the generated metrics:

```text
Training
    │
    ▼
model.pkl
    │
    ▼
Evaluation
    │
    ▼
metrics.json
    │
    ▼
DVC Metrics
```

This allows model performance to be tracked alongside the corresponding model and dataset versions.

---

## Testing

The evaluation framework has dedicated unit tests.

```text
src/ml_platform/evaluation/
└── tests/
    └── test_evaluator.py
```

The tests should verify:

* Successful evaluation.
* Multiple metrics.
* Classification metrics.
* Regression metrics.
* Invalid metric behavior.
* Evaluation failures.
* Correct metric results.
* Logging behavior where applicable.

Run the evaluation tests with:

```bash
uv run pytest src/ml_platform/evaluation -v
```

Run the complete test suite with:

```bash
uv run pytest
```

---

## Design Principles

The evaluation framework follows several design principles.

### Reusability

Evaluation logic should be implemented once in `ml_platform` and reused by individual ML projects.

### Separation of Responsibilities

The platform provides the evaluation mechanism.

The individual pipeline determines:

* Which model is evaluated.
* Which dataset is evaluated.
* Which metrics are appropriate.
* Where evaluation artifacts are stored.

### Extensibility

Additional metrics can be introduced without modifying the individual pipelines.

For example:

```python
metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
}
```

### Consistency

All ML projects use the same evaluation interface:

```python
evaluator = Evaluator(model)

metrics = evaluator.evaluate(
    X_test,
    y_test,
    metrics=metrics,
)
```

This makes evaluation behavior predictable across projects.

---

## Future Improvements

Potential future enhancements include:

* Standardized `EvaluationResult` objects.
* Metric metadata.
* Confidence intervals.
* Cross-validation evaluation.
* Threshold optimization.
* Model comparison utilities.
* Evaluation reports.
* Visualization of evaluation results.
* Automated metric validation.
* DVC metric comparison integration.
* Experiment tracking integration.

---

## ML-010 — Evaluation Framework

### Objective

Create a reusable evaluation framework that standardizes model evaluation across ML projects.

### Implemented

* `Evaluator` component.
* Classification metric support.
* Regression metric support.
* Centralized evaluation error handling.
* Centralized logging integration.
* Reusable metric interface.
* DVC-compatible metric output.
* Unit tests.
* Integration with California Housing.
* Integration with Formula 1.

### Status

**Completed**

The evaluation framework provides a common interface for evaluating classification and regression models while allowing individual ML projects to select the metrics appropriate for their use case.
