from mlflow import MlflowClient

from ml_platform.exceptions import ModelNotFoundError
from ml_platform.serving.contracts import ModelResolver
from ml_platform.serving.schemas import ModelReference, ResolvedModel


class MLflowModelResolver(ModelResolver):
    """Resolve model references using the MLflow Model Registry."""

    def __init__(self, client: MlflowClient) -> None:
        self._client = client

    def resolve(self, reference: ModelReference) -> ResolvedModel:
        """Resolve a model reference to an MLflow model URI."""
        try:
            if reference.version is not None:
                self._client.get_model_version(
                    name=reference.name,
                    version=reference.version,
                )

                return ResolvedModel(
                    name=reference.name,
                    version=reference.version,
                    uri=f"models:/{reference.name}/{reference.version}",
                )

            if reference.alias is not None:
                model = self._client.get_model_version_by_alias(
                    name=reference.name,
                    alias=reference.alias,
                )

                version = str(model.version)
                return ResolvedModel(
                    name=reference.name,
                    version=version,
                    alias=reference.alias,
                    uri=f"models:/{reference.name}/{version}",
                )

        except Exception as exc:
            raise ModelNotFoundError(
                f"Unable to resolve model '{reference.name}'."
            ) from exc

        raise ModelNotFoundError(
            f"No version or alias specified for model '{reference.name}'."
        )
