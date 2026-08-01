import os

import pandas as pd

from src.examples.f1 import PREPARED_DATA_DIR, RAW_DATA_DIR

try:
    results = pd.read_csv(f"{RAW_DATA_DIR}/results.csv")
    races = pd.read_csv(f"{RAW_DATA_DIR}/races.csv")
    drivers = pd.read_csv(f"{RAW_DATA_DIR}/drivers.csv")
    constructors = pd.read_csv(f"{RAW_DATA_DIR}/constructors.csv")
    qualifying = pd.read_csv(f"{RAW_DATA_DIR}/qualifying.csv")
    constructor_results = pd.read_csv(f"{RAW_DATA_DIR}/constructor_results.csv")
    print("All files loaded successfully.")

    df = results.merge(races, on="raceId", suffixes=("", "_race"))
    df = df.merge(drivers, on="driverId", suffixes=("", "_driver"))
    df = df.merge(constructors, on="constructorId", suffixes=("", "_constructor"))
    print("Merged dataframes successfully.")

    # Keep pre-race features only: grid, qualifying, historical rollups
    df["race_date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["driverId", "race_date"])
    print("Sorted dataframe by driverId and race_date.")

    # Save prepared data
    os.makedirs(PREPARED_DATA_DIR, exist_ok=True)
    df.to_csv(f"{PREPARED_DATA_DIR}/prepared_data.csv", index=False)
    print(f"Output File: {PREPARED_DATA_DIR}/prepared_data.csv")

except FileNotFoundError as e:
    print(f"Error: {e}")
    raise
