from pathlib import Path

import pytest
from sklearn.linear_model import LinearRegression

from ml_platform.artifacts import ArtifactManager, ArtifactMetadata
from ml_platform.exceptions import ArtifactError


def test_save_and_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved model can be loaded back successfully."""

    monkeypatch.setattr(
        "ml_platform.artifacts.manager.settings.artifacts_dir",
        tmp_path,
    )

    manager = ArtifactManager("test_project")

    model = LinearRegression()
    model.fit([[1], [2], [3]], [2, 4, 6])

    path = manager.save_model(model)

    assert path.exists()

    loaded_model = manager.load_model()

    predictions = loaded_model.predict([[4]])

    assert predictions[0] == pytest.approx(8.0)


def test_save_model_creates_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving a model creates the expected model directory."""

    monkeypatch.setattr(
        "ml_platform.artifacts.manager.settings.artifacts_dir",
        tmp_path,
    )

    manager = ArtifactManager("test_project")

    model = LinearRegression()
    model.fit([[1], [2], [3]], [2, 4, 6])

    path = manager.save_model(model)

    expected_directory = tmp_path / "test_project" / "models"

    assert expected_directory.exists()
    assert expected_directory.is_dir()
    assert path == expected_directory / "model.pkl"


def test_load_missing_model_raises_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading a nonexistent model raises a meaningful error."""

    monkeypatch.setattr(
        "ml_platform.artifacts.manager.settings.artifacts_dir",
        tmp_path,
    )

    manager = ArtifactManager("test_project")

    with pytest.raises(
        ArtifactError,
        match="Model artifact not found",
    ):
        manager.load_model()


def test_save_and_load_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact metadata can be persisted and loaded."""

    monkeypatch.setattr(
        "ml_platform.artifacts.manager.settings.artifacts_dir",
        tmp_path,
    )

    manager = ArtifactManager("test_project")

    metadata = ArtifactMetadata.create(
        artifact_name="model.pkl",
        artifact_type="model",
        description="Test model",
    )

    path = manager.save_metadata(metadata)

    assert path.exists()

    loaded_metadata = manager.load_metadata()

    assert loaded_metadata.artifact_name == "model.pkl"
    assert loaded_metadata.artifact_type == "model"
    assert loaded_metadata.description == "Test model"
    assert loaded_metadata.created_at == metadata.created_at


def test_load_missing_metadata_raises_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading missing metadata raises a meaningful error."""

    monkeypatch.setattr(
        "ml_platform.artifacts.manager.settings.artifacts_dir",
        tmp_path,
    )

    manager = ArtifactManager("test_project")

    with pytest.raises(
        ArtifactError,
        match="Artifact metadata not found",
    ):
        manager.load_metadata()
