import json
import os

import joblib
from sklearn.metrics import mean_squared_error, r2_score

from src.examples.ca_house_prediction.prepare_data import X_test, y_test

try:
    model_dir = os.path.join(
        os.path.abspath("../../"), "artifacts", "ca_house_prediction", "models"
    )

    model = joblib.load(f"{model_dir}/model.pkl")

    y_pred = model.predict(X_test)

    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    metrics_dir = os.path.join(
        os.path.abspath("../../"), "artifacts", "ca_house_prediction", "metrics"
    )
    os.makedirs(metrics_dir, exist_ok=True)

    # After training and prediction
    metrics = {
        "Coefficient": model.coef_.tolist(),  # convert numpy array to list
        "Intercept": (
            model.intercept_.tolist()
            if hasattr(model.intercept_, "tolist")
            else model.intercept_
        ),
        "Mean Squared Error": mean_squared_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred),
    }

    with open(f"{metrics_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

except Exception as e:
    print(f"Error: {e}")
