import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

from src.examples.f1 import FEATURE_DATA_DIR
from src.examples.f1.prepare_data import df


def add_driver_rollups(d):
    d["driver_recent_points"] = d.groupby("driverId")["points"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    d["driver_recent_finpos"] = d.groupby("driverId")["positionOrder"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    d["driver_recent_podiums"] = d.groupby("driverId")["positionOrder"].transform(
        lambda s: (s.shift() <= 3).rolling(5, min_periods=1).sum()
    )
    d["driver_recent_dnfs"] = (
        d.groupby("driverId")["statusId"].transform(
            lambda s: (s.shift().isin([3, 4, 5])).rolling(5, min_periods=1).sum()
        )
        if "statusId" in d.columns
        else np.nan
    )
    return d


def add_constructor_rollups(d):
    d["constructor_recent_points"] = d.groupby("constructorId")["points"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    d["constructor_recent_finpos"] = d.groupby("constructorId")[
        "positionOrder"
    ].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
    d["constructor_recent_podiums"] = d.groupby("constructorId")[
        "positionOrder"
    ].transform(lambda s: (s.shift() <= 3).rolling(5, min_periods=1).sum())
    d["constructor_recent_dnfs"] = (
        d.groupby("constructorId")["statusId"].transform(
            lambda s: (s.shift().isin([3, 4, 5])).rolling(5, min_periods=1).sum()
        )
        if "statusId" in d.columns
        else np.nan
    )
    return d


try:
    df = add_driver_rollups(df)
    df = add_constructor_rollups(df)

    if "grid" in df.columns:
        df["grid_pos"] = df["grid"].replace(0, np.nan)
    # 0 often means started from pit/unknown else: df["grid_pos"] = np.nan

    # 6) Target: driver won the race
    df["won"] = (df["positionOrder"] == 1).astype(int)

    # 7) Train/validation split by time (e.g., train <= 2016, validate 2017–2019,
    # test 2020)
    train = df[df["year"] <= 2016]
    valid = df[(df["year"] >= 2017) & (df["year"] <= 2019)]
    test = df[df["year"] == 2020]

    # 8) Select features (avoid leakage: no post-race info
    # like final time, status, laps completed)
    feat_cols = [
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

    # Handle missing values
    for c in feat_cols:
        train[c] = train[c].fillna(
            train[c].median() if train[c].dtype != "O" else "UNK"
        )
        valid[c] = valid[c].fillna(
            train[c].median() if train[c].dtype != "O" else "UNK"
        )
        test[c] = test[c].fillna(train[c].median() if train[c].dtype != "O" else "UNK")

    # 9) Encode high-cardinality categoricals with target encoding (avoid leakage: fit
    # on train only)
    cat_cols = ["constructorId", "driverId", "circuitId"]
    te = TargetEncoder(smooth=0.3)
    train_te = te.fit_transform(train[cat_cols], train["won"])
    valid_te = te.transform(valid[cat_cols])
    test_te = te.transform(test[cat_cols])

    # Save featured data
    os.makedirs(FEATURE_DATA_DIR, exist_ok=True)
    train_te = pd.DataFrame(
        te.fit_transform(train[feat_cols], train["won"]),
        columns=feat_cols,
        index=train.index,
    )
    valid_te = pd.DataFrame(
        te.transform(valid[feat_cols]), columns=feat_cols, index=valid.index
    )
    test_te = pd.DataFrame(
        te.transform(test[feat_cols]), columns=feat_cols, index=test.index
    )

    train_te.to_csv(os.path.join(FEATURE_DATA_DIR, "train.csv"), index=False)
    valid_te.to_csv(os.path.join(FEATURE_DATA_DIR, "valid.csv"), index=False)
    test_te.to_csv(os.path.join(FEATURE_DATA_DIR, "test.csv"), index=False)

    print(f"Saved featured data to {FEATURE_DATA_DIR}")
except Exception as e:
    print(f"Error in feature engineering: {e}")
    raise
