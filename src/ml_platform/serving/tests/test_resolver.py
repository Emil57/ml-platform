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

    assert result == "models:/california-housing/3"

    client.get_model_version.assert_called_once_with(
        name="california-housing",
        version="3",
    )


def test_resolve_model_alias() -> None:
    client = MagicMock()

    resolver = MLflowModelResolver(client)

    reference = ModelReference(
        name="california-housing",
        alias="production",
    )

    result = resolver.resolve(reference)

    assert result == "models:/california-housing@production"

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
