from collections.abc import Callable

import pandas as pd

from ml_platform.exceptions import EvaluationError


class Evaluator:
    """Reusable model evaluation component."""

    PROBABILITY_METRICS = {
        "log_loss",
        "brier_score",
        "roc_auc",
    }

    def __init__(self, model):
        self.model = model

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: dict[str, Callable],
    ) -> dict[str, float]:
        """Evaluate a model using the provided metrics."""

        try:
            predictions = self.model.predict(X)

            probabilities = None

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X)[:, 1]

            results: dict[str, float] = {}

            for name, metric in metrics.items():
                if name in self.PROBABILITY_METRICS:
                    if probabilities is None:
                        raise EvaluationError(
                            f"Metric '{name}' requires probability predictions."
                        )

                    results[name] = float(metric(y, probabilities))
                else:
                    results[name] = float(metric(y, predictions))

            return results

        except EvaluationError:
            raise

        except Exception as exc:
            raise EvaluationError(f"Model evaluation failed: {exc}") from exc
