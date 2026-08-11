from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow

from ml_platform.config import settings
from ml_platform.exceptions import ExperimentError, RunError


class ExperimentTracker:
    """Manage ML experiment tracking through MLflow."""

    def __init__(
        self,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name or settings.mlflow_experiment_name

        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri

        self.registry_uri = registry_uri or settings.mlflow_registry_uri

        self._configure_mlflow()

    def _configure_mlflow(self) -> None:
        """Configure MLflow using platform settings."""

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(self.registry_uri)

        except Exception as exc:
            raise ExperimentError("Failed to configure MLflow tracking.") from exc

    def _get_or_create_experiment(self) -> str:
        """Get the configured experiment or create it."""

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is not None:
                return experiment.experiment_id

            return mlflow.create_experiment(
                self.experiment_name,
            )

        except Exception as exc:
            raise ExperimentError(
                f"Failed to initialize experiment: " f"{self.experiment_name}"
            ) from exc

    def start_run(self) -> mlflow.ActiveRun:
        """Start a new MLflow run."""

        experiment_id = self._get_or_create_experiment()

        try:
            return mlflow.start_run(
                experiment_id=experiment_id,
            )

        except Exception as exc:
            raise RunError(
                f"Failed to start MLflow run for experiment: " f"{self.experiment_name}"
            ) from exc

    def log_params(
        self,
        params: Mapping[str, Any],
    ) -> None:
        """Log experiment parameters."""

        try:
            mlflow.log_params(dict(params))

        except Exception as exc:
            raise RunError("Failed to log experiment parameters.") from exc

    def log_metrics(
        self,
        metrics: Mapping[str, float],
    ) -> None:
        """Log experiment metrics."""

        try:
            mlflow.log_metrics(dict(metrics))

        except Exception as exc:
            raise RunError("Failed to log experiment metrics.") from exc

    def log_artifact(
        self,
        path: Path,
    ) -> None:
        """Log a local artifact to the active MLflow run."""

        try:
            mlflow.log_artifact(str(path))

        except Exception as exc:
            raise RunError(f"Failed to log artifact: {path}") from exc
