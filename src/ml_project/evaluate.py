"""Evaluate saved models on the full dataset and write metrics to `metrics/metrics.json`."""

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def main():
    df = pd.read_csv("data/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    results = {}
    os.makedirs("metrics", exist_ok=True)
    for name in ["logistic", "random_forest"]:
        model = joblib.load(f"models/{name}.joblib")
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        results[name] = {"accuracy": float(acc)}
    with open("metrics/metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("Saved metrics/metrics.json")


if __name__ == "__main__":
    main()
