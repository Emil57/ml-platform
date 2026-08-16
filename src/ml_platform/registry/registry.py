from typing import Any

import mlflow
from mlflow import MlflowClient

from ml_platform.exceptions import ModelLifecycleError, RegistryError


class ModelRegistry:
    """Manage MLflow registered models."""

    def __init__(self) -> None:
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        model_name: str,
    ) -> Any:
        """Register a model artifact in the MLflow Model Registry."""

        try:
            return mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

        except Exception as exc:
            raise RegistryError(f"Failed to register model: {model_name}") from exc

    def get_model(
        self,
        model_name: str,
    ) -> Any:
        """Retrieve a registered model by name."""

        try:
            return self.client.get_registered_model(model_name)

        except Exception as exc:
            raise RegistryError(
                f"Failed to retrieve registered model: {model_name}"
            ) from exc

    def get_model_version(
        self,
        model_name: str,
        version: str,
    ) -> Any:
        """Retrieve a specific registered model version."""

        try:
            return self.client.get_model_version(
                name=model_name,
                version=version,
            )

        except Exception as exc:
            raise RegistryError(
                f"Failed to retrieve model version: " f"{model_name} v{version}"
            ) from exc

    def list_versions(
        self,
        model_name: str,
    ) -> list[Any]:
        """List all versions of a registered model."""

        try:
            return list(self.client.search_model_versions(f"name='{model_name}'"))

        except Exception as exc:
            raise RegistryError(f"Failed to list model versions: {model_name}") from exc

    def set_alias(
        self,
        model_name: str,
        alias: str,
        version: str,
    ) -> None:
        """Assign an alias to a registered model version."""

        try:
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version,
            )

        except Exception as exc:
            raise ModelLifecycleError(
                f"Failed to assign alias '{alias}' "
                f"to model '{model_name}' version '{version}'."
            ) from exc

    def get_model_by_alias(
        self,
        model_name: str,
        alias: str,
    ) -> Any:
        """Retrieve a registered model version by alias."""

        try:
            return self.client.get_model_version_by_alias(
                name=model_name,
                alias=alias,
            )

        except Exception as exc:
            raise ModelLifecycleError(
                f"Failed to retrieve model '{model_name}' " f"using alias '{alias}'."
            ) from exc
