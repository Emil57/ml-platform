import json
from pathlib import Path

import joblib
from sklearn.metrics import mean_squared_error, r2_score

from ml_platform.data import DataLoader

ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[3] / "artifacts" / "ca_house_prediction"
)

DATA_DIR = ARTIFACTS_DIR / "data"
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


def main() -> None:
    """Evaluate the California Housing model."""

    # Load test data
    loader = DataLoader()

    test_data = loader.load_csv(DATA_DIR / "test.csv")

    X_test = test_data[["MedInc"]]
    y_test = test_data["MedHouseVal"]

    # Load trained model
    model = joblib.load(MODEL_DIR / "model.pkl")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "Coefficient": model.coef_.tolist(),
        "Intercept": (
            model.intercept_.tolist()
            if hasattr(model.intercept_, "tolist")
            else model.intercept_
        ),
        "Mean Squared Error": mse,
        "R2 Score": r2,
    }

    with open(METRICS_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    main()
