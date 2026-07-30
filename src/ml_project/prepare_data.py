"""Prepare a small synthetic dataset and save to `data/train.csv`."""

import os

import pandas as pd
from sklearn.datasets import make_classification


def main():
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=5, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/train.csv", index=False)
    print("Saved data/train.csv")


if __name__ == "__main__":
    main()
