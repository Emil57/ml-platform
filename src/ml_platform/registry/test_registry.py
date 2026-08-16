import pytest

from ml_platform.exceptions import ModelLifecycleError
from ml_platform.registry.registry import ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry()


def test_set_alias(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    registry.set_alias(
        model_name=registered_model,
        alias="champion",
        version="1",
    )

    model_version = registry.get_model_by_alias(
        model_name=registered_model,
        alias="champion",
    )

    assert model_version.version == 1


def test_alias_can_be_reassigned(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    registry.set_alias(
        model_name=registered_model,
        alias="champion",
        version="1",
    )

    registry.set_alias(
        model_name=registered_model,
        alias="champion",
        version="2",
    )

    model_version = registry.get_model_by_alias(
        model_name=registered_model,
        alias="champion",
    )

    assert model_version.version == 2


def test_get_model_by_invalid_alias_raises_error(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    with pytest.raises(ModelLifecycleError):
        registry.get_model_by_alias(
            model_name=registered_model,
            alias="does-not-exist",
        )


def test_set_alias_with_invalid_version_raises_error(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    with pytest.raises(ModelLifecycleError):
        registry.set_alias(
            model_name=registered_model,
            alias="champion",
            version="999",
        )
