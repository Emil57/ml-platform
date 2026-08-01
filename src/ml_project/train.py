"""Train multiple simple models and save them under `models/`."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    os.makedirs("models", exist_ok=True)
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.joblib")
        print(f"Saved models/{name}.joblib")


if __name__ == "__main__":
    main()
