import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from examples.f1 import FEATURE_DATA_DIR, METRICS_DIR, MODELS_DIR


def load_test_data() -> pd.DataFrame:
    """Load the featured test dataset."""

    test_path = Path(FEATURE_DATA_DIR) / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")

    return pd.read_csv(test_path)


def evaluate_model(
    model,
    test: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate the trained F1 model."""

    target_column = "won"

    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    # Predict probabilities and classes.
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def main() -> None:
    """Load the trained model, evaluate it, and save metrics."""

    model_path = Path(MODELS_DIR) / "model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    test = load_test_data()

    metrics = evaluate_model(model, test)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    metrics_dir = Path(METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "metrics.json"

    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
