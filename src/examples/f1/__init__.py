"F1 artifacts and files"

import os

KAGGLE_DATASET = "rohanrao/formula-1-world-championship-1950-2020"

RAW_DATA_DIR = os.path.join(os.path.abspath("../../"), "artifacts", "f1", "data", "raw")

PREPARED_DATA_DIR = os.path.join(
    os.path.abspath("../../"), "artifacts", "f1", "data", "prepared"
)

FEATURE_DATA_DIR = os.path.join(
    os.path.abspath("../../"), "artifacts", "f1", "data", "featured"
)

MODELS_DIR = os.path.join(os.path.abspath("../../"), "artifacts", "f1", "models")

METRICS_DIR = os.path.join(os.path.abspath("../../"), "artifacts", "f1", "metrics")
