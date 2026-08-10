"""F1 artifacts and files."""

from pathlib import Path

KAGGLE_DATASET = "rohanrao/formula-1-world-championship-1950-2020"


# Repository root:
# machine-learning/
# └── src/
#     └── examples/
#         └── f1/
#             └── __init__.py
#
# parents[0] = f1
# parents[1] = examples
# parents[2] = src
# parents[3] = machine-learning

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "f1"

RAW_DATA_DIR = ARTIFACTS_DIR / "data" / "raw"

PREPARED_DATA_DIR = ARTIFACTS_DIR / "data" / "prepared"

FEATURE_DATA_DIR = ARTIFACTS_DIR / "data" / "featured"

MODELS_DIR = ARTIFACTS_DIR / "models"

METRICS_DIR = ARTIFACTS_DIR / "metrics"
