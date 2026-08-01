import os

import kaggle

from src.examples.f1 import KAGGLE_DATASET, RAW_DATA_DIR

try:
    # Make sure output directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Load the latest version
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=RAW_DATA_DIR, unzip=True)
    print(f"Downloaded {KAGGLE_DATASET} to {RAW_DATA_DIR}")
except Exception as e:
    print(f"Error: {e}")
