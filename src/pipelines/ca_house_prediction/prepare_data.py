from pathlib import Path

from ml_platform.data import (
    DataLoader,
    DataSplitter,
    DataValidator,
)

DATA_DIR = (
    Path(__file__).resolve().parents[3] / "artifacts" / "ca_house_prediction" / "data"
)

DATASET_PATH = DATA_DIR / "housing.csv"


def main() -> None:
    """Prepare the California Housing dataset."""

    # Load raw data
    loader = DataLoader()
    data = loader.load_csv(DATASET_PATH)

    print(data.head())

    # Validate raw data
    validator = DataValidator()

    validator.validate_not_empty(data)

    validator.validate_columns(
        data,
        [
            "median_income",
            "median_house_value",
        ],
    )

    # Project-specific transformation
    data = data.rename(
        columns={
            "median_income": "MedInc",
            "housing_median_age": "HouseAge",
            "total_rooms": "AveRooms",
            "total_bedrooms": "AveBedrms",
            "population": "Population",
            "households": "Households",
            "median_house_value": "MedHouseVal",
        }
    )

    # Select model features and target
    model_data = data[["MedInc", "MedHouseVal"]]

    # Split dataset
    splitter = DataSplitter()

    train, validation, test = splitter.split(model_data)

    print(
        f"Split data: {len(train)} train samples, "
        f"{len(validation)} validation samples, "
        f"{len(test)} test samples."
    )

    # Save datasets
    train.to_csv(DATA_DIR / "train.csv", index=False)
    validation.to_csv(DATA_DIR / "validation.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Saved prepared data to {DATA_DIR}")


if __name__ == "__main__":
    main()
