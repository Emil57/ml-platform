from abc import ABC, abstractmethod
from pathlib import Path


class DataSource(ABC):
    """Interface for dataset sources."""

    @abstractmethod
    def fetch(self, destination: Path) -> Path:
        """Fetch data and return the local path."""
        raise NotImplementedError