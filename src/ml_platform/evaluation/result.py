from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Standardized model evaluation results."""

    metrics: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        """Return evaluation metrics as a dictionary."""
        return self.metrics
