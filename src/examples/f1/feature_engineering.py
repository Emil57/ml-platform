from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from examples.f1 import FEATURE_DATA_DIR
from examples.f1.prepare_data import prepare_data


def add_driver_rollups(data: pd.DataFrame) -> pd.DataFrame:
    """Add recent driver performance features."""

    data["driver_recent_points"] = data.groupby("driverId")["points"].transform(
        lambda series: series.shift().rolling(5, min_periods=1).mean()
    )

    data["driver_recent_finpos"] = data.groupby("driverId")["positionOrder"].transform(
        lambda series: series.shift().rolling(5, min_periods=1).mean()
    )

    data["driver_recent_podiums"] = data.groupby("driverId")["positionOrder"].transform(
        lambda series: (series.shift() <= 3).rolling(5, min_periods=1).sum()
    )

    data["driver_recent_dnfs"] = (
        data.groupby("driverId")["statusId"].transform(
            lambda series: (series.shift().isin([3, 4, 5]))
            .rolling(5, min_periods=1)
            .sum()
        )
        if "statusId" in data.columns
        else np.nan
    )

    return data


def add_constructor_rollups(data: pd.DataFrame) -> pd.DataFrame:
    """Add recent constructor performance features."""

    data["constructor_recent_points"] = data.groupby("constructorId")[
        "points"
    ].transform(lambda series: series.shift().rolling(5, min_periods=1).mean())

    data["constructor_recent_finpos"] = data.groupby("constructorId")[
        "positionOrder"
    ].transform(lambda series: series.shift().rolling(5, min_periods=1).mean())

    data["constructor_recent_podiums"] = data.groupby("constructorId")[
        "positionOrder"
    ].transform(lambda series: (series.shift() <= 3).rolling(5, min_periods=1).sum())

    data["constructor_recent_dnfs"] = (
        data.groupby("constructorId")["statusId"].transform(
            lambda series: (series.shift().isin([3, 4, 5]))
            .rolling(5, min_periods=1)
            .sum()
        )
        if "statusId" in data.columns
        else np.nan
    )

    return data


def feature_engineer(data: pd.DataFrame) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create F1 features and split data chronologically."""

    data = add_driver_rollups(data)
    data = add_constructor_rollups(data)

    # Convert grid position 0 to missing.
    if "grid" in data.columns:
        data["grid_pos"] = data["grid"].replace(0, np.nan)

    # Target: driver won the race.
    data["won"] = (data["positionOrder"] == 1).astype(int)

    # Time-based split to prevent future data leakage.
    train = data[data["year"] <= 2016].copy()

    valid = data[(data["year"] >= 2017) & (data["year"] <= 2019)].copy()

    test = data[data["year"] == 2020].copy()

    # Select features while avoiding post-race information.
    feature_columns = [
        "grid_pos",
        "driver_recent_points",
        "driver_recent_finpos",
        "driver_recent_podiums",
        "driver_recent_dnfs",
        "constructorId",
        "driverId",
        "circuitId",
        "round",
    ]

    # Fill missing values using statistics calculated from training data.
    for column in feature_columns:
        if train[column].dtype != "O":
            median = train[column].median()

            train[column] = train[column].fillna(median)
            valid[column] = valid[column].fillna(median)
            test[column] = test[column].fillna(median)
        else:
            train[column] = train[column].fillna("UNK")
            valid[column] = valid[column].fillna("UNK")
            test[column] = test[column].fillna("UNK")

    # Target encoding for categorical features.
    categorical_columns = [
        "constructorId",
        "driverId",
        "circuitId",
    ]

    encoder = TargetEncoder(smooth=0.3)

    train_encoded = encoder.fit_transform(
        train[categorical_columns],
        train["won"],
    )

    valid_encoded = encoder.transform(
        valid[categorical_columns],
    )

    test_encoded = encoder.transform(
        test[categorical_columns],
    )

    # Convert encoded categorical features to DataFrames.
    train_encoded = pd.DataFrame(
        train_encoded,
        columns=categorical_columns,
        index=train.index,
    )

    valid_encoded = pd.DataFrame(
        valid_encoded,
        columns=categorical_columns,
        index=valid.index,
    )

    test_encoded = pd.DataFrame(
        test_encoded,
        columns=categorical_columns,
        index=test.index,
    )

    # Replace categorical columns with their encoded values.
    train[categorical_columns] = train_encoded
    valid[categorical_columns] = valid_encoded
    test[categorical_columns] = test_encoded

    # Keep only the final feature columns.
    train_features = train[feature_columns + ["won"]]
    valid_features = valid[feature_columns + ["won"]]
    test_features = test[feature_columns + ["won"]]

    return train_features, valid_features, test_features


def main() -> None:
    """Run F1 feature engineering and save the resulting datasets."""

    data = prepare_data()

    train, valid, test = feature_engineer(data)

    output_dir = Path(FEATURE_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    valid.to_csv(
        output_dir / "valid.csv",
        index=False,
    )

    test.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    print(f"Saved featured data to {output_dir}")


if __name__ == "__main__":
    main()
