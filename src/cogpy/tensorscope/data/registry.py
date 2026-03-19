"""Data registry for managing multiple data modalities."""

from __future__ import annotations

from .modality import DataModality


class DataRegistry:
    """
    Registry for data modalities.

    Phase 5: Full multi-modal support
    """

    def __init__(self):
        self._modalities: dict[str, DataModality] = {}
        self._active: str | None = None

    def register(self, name: str, modality: DataModality) -> None:
        """Register a data modality."""
        key = str(name)
        self._modalities[key] = modality
        if self._active is None:
            self._active = key

    def get(self, name: str) -> DataModality | None:
        """Get modality by name."""
        return self._modalities.get(str(name))

    def get_active(self) -> DataModality | None:
        """Get currently active modality."""
        if self._active is None:
            return None
        return self._modalities.get(self._active)

    def get_active_name(self) -> str | None:
        """Get name of currently active modality."""
        return self._active

    def set_active(self, name: str) -> None:
        """Set active modality by name."""
        key = str(name)
        if key not in self._modalities:
            raise ValueError(
                f"Modality {key!r} not registered. Available: {list(self._modalities.keys())}"
            )
        self._active = key

    def list(self) -> list[str]:
        """List registered modality names."""
        return list(self._modalities.keys())

    def to_dict(self) -> dict:
        """Serialize registry (metadata only)."""
        return {
            "modalities": {name: modality.to_dict() for name, modality in self._modalities.items()},
            "active": self._active,
        }
