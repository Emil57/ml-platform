import json
import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.examples.f1 import METRICS_DIR, MODELS_DIR
from src.examples.f1.feature_engineering import test, test_te

try:
    model = joblib.load(f"{MODELS_DIR}/model.pkl")

    # Predict probabilities and classes
    y_proba = model.predict_proba(test_te)[:, 1]
    y_pred = model.predict(test_te)

    print("Accuracy:", accuracy_score(test["won"], y_pred))
    print("F1 score:", f1_score(test["won"], y_pred))
    print("Log loss:", log_loss(test["won"], y_proba))
    print("Brier score:", brier_score_loss(test["won"], y_proba))
    print("ROC AUC:", roc_auc_score(test["won"], y_proba))

    metrics = {
        "accuracy": accuracy_score(test["won"], y_pred),
        "f1_score": f1_score(test["won"], y_pred),
        "log_loss": log_loss(test["won"], y_proba),
        "brier_score": brier_score_loss(test["won"], y_proba),
        "roc_auc": roc_auc_score(test["won"], y_proba),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(f"{METRICS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f)

except Exception as e:
    print(f"Error: {e}")
