from unittest.mock import MagicMock

import pytest

from ml_platform.exceptions import ModelNotFoundError
from ml_platform.serving.resolver import MLflowModelResolver
from ml_platform.serving.schemas import ModelReference


def test_resolve_model_version() -> None:
    client = MagicMock()

    resolver = MLflowModelResolver(client)

    reference = ModelReference(
        name="california-housing",
        version="3",
    )

    result = resolver.resolve(reference)

    assert result.uri == "models:/california-housing/3"
    assert result.version == "3"
    assert result.alias is None

    client.get_model_version.assert_called_once_with(
        name="california-housing",
        version="3",
    )


def test_resolve_model_alias() -> None:
    client = MagicMock()
    client.get_model_version_by_alias.return_value.version = "7"

    resolver = MLflowModelResolver(client)

    reference = ModelReference(
        name="california-housing",
        alias="production",
    )

    result = resolver.resolve(reference)

    assert result.uri == "models:/california-housing/7"
    assert result.version == "7"
    assert result.alias == "production"

    client.get_model_version_by_alias.assert_called_once_with(
        name="california-housing",
        alias="production",
    )


def test_resolve_missing_model_version() -> None:
    client = MagicMock()

    client.get_model_version.side_effect = Exception("Model version not found")

    resolver = MLflowModelResolver(client)

    reference = ModelReference(
        name="california-housing",
        version="999",
    )

    with pytest.raises(ModelNotFoundError):
        resolver.resolve(reference)


def test_resolve_missing_model_alias() -> None:
    client = MagicMock()

    client.get_model_version_by_alias.side_effect = Exception("Model alias not found")

    resolver = MLflowModelResolver(client)

    reference = ModelReference(
        name="california-housing",
        alias="production",
    )

    with pytest.raises(ModelNotFoundError):
        resolver.resolve(reference)


def test_client_is_created_lazily_on_resolve() -> None:
    factory = MagicMock()

    resolver = MLflowModelResolver(client_factory=factory)
    factory.assert_not_called()

    resolver.resolve(ModelReference(name="model", version="1"))
    resolver.resolve(ModelReference(name="model", version="1"))

    factory.assert_called_once_with()


def test_unreachable_registry_raises_model_not_found() -> None:
    factory = MagicMock(side_effect=OSError("unable to open database file"))
    resolver = MLflowModelResolver(client_factory=factory)

    with pytest.raises(ModelNotFoundError):
        resolver.resolve(ModelReference(name="model", version="1"))
