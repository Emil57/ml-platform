import json

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from ml_platform.artifacts import ArtifactManager
from ml_platform.evaluation import Evaluator
from pipelines.ca_house_prediction import (
    DATA_DIR,
    METRICS_DIR,
)


def main() -> None:
    """Evaluate the California Housing model."""

    test_data = pd.read_csv(DATA_DIR / "test.csv")

    X_test = test_data[["MedInc"]]
    y_test = test_data["MedHouseVal"]

    artifact_manager = ArtifactManager("ca_house_prediction")

    model = artifact_manager.load_model()
    evaluator = Evaluator(model)

    result = evaluator.evaluate(
        X_test,
        y_test,
        metrics={
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
        },
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "metrics.json"

    with metrics_path.open("w") as file:
        json.dump(result, file, indent=4)

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
