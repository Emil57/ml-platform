from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class ArtifactMetadata:
    """Metadata associated with an ML artifact."""

    artifact_name: str
    artifact_type: str
    created_at: str
    description: str | None = None

    @classmethod
    def create(
        cls,
        artifact_name: str,
        artifact_type: str,
        description: str | None = None,
    ) -> "ArtifactMetadata":
        """Create metadata with the current timestamp."""

        return cls(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            created_at=datetime.now().isoformat(),
            description=description,
        )

    def to_dict(self) -> dict:
        """Convert metadata to a dictionary."""

        return asdict(self)
