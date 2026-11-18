"""Promote a registered model version by assigning it an MLflow alias.

Usage:
    python -m ml_platform.registry.promote --model-name NAME --alias ALIAS
        [--version VERSION]

When no version is given, the latest registered version is promoted.
"""

import argparse
from collections.abc import Sequence

from ml_platform.exceptions import RegistryError
from ml_platform.registry.registry import ModelRegistry
from ml_platform.utils import get_logger

logger = get_logger(__name__)


def latest_version(registry: ModelRegistry, model_name: str) -> str:
    """Return the highest registered version number for a model."""

    versions = registry.list_versions(model_name)
    if not versions:
        raise RegistryError(f"No registered versions found for model: {model_name}")
    return str(max(int(version.version) for version in versions))


def promote_model(
    registry: ModelRegistry,
    model_name: str,
    alias: str,
    version: str | None = None,
) -> str:
    """Point an alias at a model version and return the promoted version."""

    target_version = version or latest_version(registry, model_name)
    # Fail early with a clear error if the version does not exist.
    registry.get_model_version(model_name, target_version)
    registry.set_alias(model_name=model_name, alias=alias, version=target_version)
    logger.info(
        "Promoted model '%s' version %s to alias '%s'",
        model_name,
        target_version,
        alias,
    )
    return target_version


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument(
        "--version", default=None, help="Version to promote (default: latest)."
    )
    args = parser.parse_args(argv)

    version = promote_model(
        ModelRegistry(),
        model_name=args.model_name,
        alias=args.alias,
        version=args.version or None,
    )
    print(f"{args.model_name} v{version} -> @{args.alias}")


if __name__ == "__main__":
    main()
