from ml_platform.artifacts import ArtifactMetadata


def test_create_metadata() -> None:
    """Metadata can be created with the required artifact fields."""

    metadata = ArtifactMetadata.create(
        artifact_name="model.pkl",
        artifact_type="model",
        description="Test model",
    )

    assert metadata.artifact_name == "model.pkl"
    assert metadata.artifact_type == "model"
    assert metadata.description == "Test model"
    assert metadata.created_at is not None


def test_metadata_to_dict() -> None:
    """Metadata can be serialized to a dictionary."""

    metadata = ArtifactMetadata.create(
        artifact_name="model.pkl",
        artifact_type="model",
    )

    result = metadata.to_dict()

    assert result["artifact_name"] == "model.pkl"
    assert result["artifact_type"] == "model"
    assert result["description"] is None
    assert "created_at" in result
    assert result["created_at"] == metadata.created_at
