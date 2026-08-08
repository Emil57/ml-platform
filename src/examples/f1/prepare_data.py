from pathlib import Path

import pandas as pd

from examples.f1 import PREPARED_DATA_DIR, RAW_DATA_DIR
from ml_platform.data import DataLoader, DataValidator


def prepare_data() -> pd.DataFrame:
    """Load, validate, merge, and sort Formula 1 data."""

    loader = DataLoader()
    validator = DataValidator()

    raw_data_dir = Path(RAW_DATA_DIR)

    # Load raw datasets
    results = loader.load_csv(raw_data_dir / "results.csv")
    races = loader.load_csv(raw_data_dir / "races.csv")
    drivers = loader.load_csv(raw_data_dir / "drivers.csv")
    constructors = loader.load_csv(raw_data_dir / "constructors.csv")
    qualifying = loader.load_csv(raw_data_dir / "qualifying.csv")
    constructor_results = loader.load_csv(raw_data_dir / "constructor_results.csv")

    print("All files loaded successfully.")

    # Validate datasets
    validator.validate_not_empty(results)
    validator.validate_not_empty(races)
    validator.validate_not_empty(drivers)
    validator.validate_not_empty(constructors)
    validator.validate_not_empty(qualifying)
    validator.validate_not_empty(constructor_results)

    # Merge datasets
    df = results.merge(
        races,
        on="raceId",
        suffixes=("", "_race"),
    )

    df = df.merge(
        drivers,
        on="driverId",
        suffixes=("", "_driver"),
    )

    df = df.merge(
        constructors,
        on="constructorId",
        suffixes=("", "_constructor"),
    )

    print("Merged dataframes successfully.")

    # Create race date
    df["race_date"] = pd.to_datetime(df["date"])

    # Sort chronologically by driver
    df = df.sort_values(["driverId", "race_date"])

    print("Sorted dataframe by driverId and race_date.")

    return df


def main() -> None:
    """Prepare and save Formula 1 data."""

    df = prepare_data()

    prepared_data_dir = Path(PREPARED_DATA_DIR)
    prepared_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = prepared_data_dir / "prepared_data.csv"

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Output File: {output_path}")


if __name__ == "__main__":
    main()
