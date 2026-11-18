import pytest

from ml_platform.exceptions import RegistryError
from ml_platform.registry.promote import latest_version, main, promote_model
from ml_platform.registry.registry import ModelRegistry


def test_promote_latest_version_by_default(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    version = promote_model(registry, registered_model, alias="production")

    assert version == "2"
    assert registry.get_model_by_alias(registered_model, "production").version == 2


def test_promote_specific_version(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    version = promote_model(registry, registered_model, alias="staging", version="1")

    assert version == "1"
    assert registry.get_model_by_alias(registered_model, "staging").version == 1


def test_promote_missing_version_fails(
    registry: ModelRegistry,
    registered_model: str,
) -> None:
    with pytest.raises(RegistryError):
        promote_model(registry, registered_model, alias="staging", version="99")


def test_latest_version_of_unknown_model_fails(registry: ModelRegistry) -> None:
    with pytest.raises(RegistryError):
        latest_version(registry, "missing-model")


def test_main_promotes_model(
    registry: ModelRegistry,
    registered_model: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--model-name", registered_model, "--alias", "production"])

    assert "v2 -> @production" in capsys.readouterr().out
    assert registry.get_model_by_alias(registered_model, "production").version == 2
