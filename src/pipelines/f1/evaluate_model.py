import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from ml_platform.evaluation import Evaluator
from pipelines.f1 import FEATURE_DATA_DIR, METRICS_DIR, MODELS_DIR


def main() -> None:
    model_path = Path(MODELS_DIR) / "model.pkl"
    test_path = Path(FEATURE_DATA_DIR) / "test.csv"

    model = joblib.load(model_path)
    test = __import__("pandas").read_csv(test_path)

    X_test = test.drop(columns=["won"])
    y_test = test["won"]

    evaluator = Evaluator(model)

    metrics = evaluator.evaluate(
        X_test,
        y_test,
        metrics={
            "accuracy": accuracy_score,
            "f1_score": f1_score,
            "log_loss": log_loss,
            "brier_score": brier_score_loss,
            "roc_auc": roc_auc_score,
        },
    )

    metrics_dir = Path(METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / "metrics.json"

    with metrics_path.open("w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
