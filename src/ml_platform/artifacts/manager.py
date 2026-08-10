import json
from pathlib import Path

import joblib

from ml_platform.artifacts.metadata import ArtifactMetadata
from ml_platform.config import settings
from ml_platform.exceptions import ArtifactError
from typing import Any

class ArtifactManager:
    """Manage machine learning artifacts for a project."""

    def __init__(self, project_name: str) -> None:
        self.project_dir = settings.artifacts_dir / project_name

        self.models_dir = self.project_dir / "models"
        self.metrics_dir = self.project_dir / "metrics"
        self.metadata_dir = self.project_dir / "metadata"

    def save_model(
        self,
        model: object,
        filename: str = "model.pkl",
    ) -> Path:
        """Save a trained model to the project's artifact directory."""

        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)

            path = self.models_dir / filename

            joblib.dump(model, path)

            return path

        except Exception as exc:
            raise ArtifactError(f"Failed to save model artifact: {filename}") from exc

    def load_model(
        self,
        filename: str = "model.pkl",
    ) -> Any:
        """Load a trained model from the project's artifact directory."""

        path = self.models_dir / filename

        if not path.exists():
            raise ArtifactError(f"Model artifact not found: {path}")

        try:
            return joblib.load(path)

        except Exception as exc:
            raise ArtifactError(f"Failed to load model artifact: {path}") from exc

    def save_metadata(
        self,
        metadata: ArtifactMetadata,
        filename: str = "model.json",
    ) -> Path:
        """Save artifact metadata."""

        try:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

            path = self.metadata_dir / filename

            with path.open("w", encoding="utf-8") as file:
                json.dump(
                    metadata.to_dict(),
                    file,
                    indent=4,
                )

            return path

        except Exception as exc:
            raise ArtifactError(
                f"Failed to save artifact metadata: {filename}"
            ) from exc

    def load_metadata(
        self,
        filename: str = "model.json",
    ) -> ArtifactMetadata:
        """Load artifact metadata."""

        path = self.metadata_dir / filename

        if not path.exists():
            raise ArtifactError(f"Artifact metadata not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            return ArtifactMetadata(**data)

        except Exception as exc:
            raise ArtifactError(f"Failed to load artifact metadata: {path}") from exc
