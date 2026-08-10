from pathlib import Path

from sklearn.linear_model import LinearRegression

from ml_platform.artifacts import ArtifactManager
from ml_platform.data import DataLoader
from ml_platform.training import Trainer

DATA_DIR = (
    Path(__file__).resolve().parents[3] / "artifacts" / "ca_house_prediction" / "data"
)


def main() -> None:
    """Train the California Housing model."""

    loader = DataLoader()

    train_data = loader.load_csv(DATA_DIR / "train.csv")

    X_train = train_data[["MedInc"]]
    y_train = train_data["MedHouseVal"]

    print(f"Data Shape: {X_train.shape}, " f"{y_train.shape}")

    model = LinearRegression()

    trainer = Trainer(model)

    trainer.train(
        X_train,
        y_train,
    )

    artifact_manager = ArtifactManager("ca_house_prediction")

    model_path = artifact_manager.save_model(model)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
