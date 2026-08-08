from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from ml_platform.data import DataLoader


DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "ca_house_prediction"
    / "data"
)

MODEL_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "ca_house_prediction"
    / "models"
)


def main() -> None:
    """Train the California Housing model."""

    loader = DataLoader()

    train_data = loader.load_csv(
        DATA_DIR / "train.csv"
    )

    X_train = train_data[["MedInc"]]
    y_train = train_data["MedHouseVal"]

    print(
        f"Data Shape: {X_train.shape}, "
        f"{y_train.shape}"
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.pkl"

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()