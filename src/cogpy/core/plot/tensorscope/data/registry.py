"""Data registry for managing multiple data modalities."""

from __future__ import annotations

from typing import Any


class DataRegistry:
    """
    Registry for data modalities.

    Phase 1: Minimal implementation
    Phase 5: Full multi-modal support
    """

    def __init__(self):
        self._modalities: dict[str, Any] = {}

    def register(self, name: str, modality: Any) -> None:
        """Register a data modality."""
        self._modalities[str(name)] = modality

    def get(self, name: str) -> Any | None:
        """Get modality by name."""
        return self._modalities.get(str(name))

    def list(self) -> list[str]:
        """List registered modality names."""
        return sorted(self._modalities.keys())

    def to_dict(self) -> dict:
        """Serialize registry (Phase 1: minimal)."""
        return {"modalities": self.list()}

