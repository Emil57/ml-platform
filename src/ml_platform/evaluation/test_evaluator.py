import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ml_platform.evaluation import Evaluator
from ml_platform.exceptions import EvaluationError


def test_evaluator_returns_metrics():
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X, y)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        X,
        y,
        metrics={
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
        },
    )

    assert "mean_squared_error" in results
    assert "r2_score" in results
    assert results["mean_squared_error"] == pytest.approx(0.0)
    assert results["r2_score"] == pytest.approx(1.0)


def test_evaluator_returns_float_metrics():
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X, y)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        X,
        y,
        metrics={"mse": mean_squared_error},
    )

    assert isinstance(results["mse"], float)


def test_evaluator_raises_evaluation_error():
    class BrokenModel:
        def predict(self, X):
            raise RuntimeError("Prediction failed")

    evaluator = Evaluator(BrokenModel())

    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([1, 2, 3])

    with pytest.raises(EvaluationError, match="Model evaluation failed"):
        evaluator.evaluate(
            X,
            y,
            metrics={"mse": mean_squared_error},
        )


def test_evaluator_classification_metrics():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )

    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        X,
        y,
        metrics={
            "accuracy": accuracy_score,
            "f1_score": f1_score,
        },
    )

    assert "accuracy" in results
    assert "f1_score" in results
    assert isinstance(results["accuracy"], float)
    assert isinstance(results["f1_score"], float)


def test_evaluator_probability_metrics():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )

    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        X,
        y,
        metrics={
            "log_loss": log_loss,
            "brier_score": brier_score_loss,
            "roc_auc": roc_auc_score,
        },
    )

    assert "log_loss" in results
    assert "brier_score" in results
    assert "roc_auc" in results

    assert isinstance(results["log_loss"], float)
    assert isinstance(results["brier_score"], float)
    assert isinstance(results["roc_auc"], float)


def test_probability_metric_requires_predict_proba():
    class ModelWithoutProbabilities:
        def predict(self, X):
            return [0, 1, 0]

    evaluator = Evaluator(ModelWithoutProbabilities())

    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(
        EvaluationError,
        match="requires probability predictions",
    ):
        evaluator.evaluate(
            X,
            y,
            metrics={
                "roc_auc": roc_auc_score,
            },
        )
