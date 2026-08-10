from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "ca_house_prediction"

DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"


from ml_platform.evaluation.evaluator import Evaluator
from ml_platform.evaluation.result import EvaluationResult

__all__ = [
    "Evaluator",
    "EvaluationResult",
]
