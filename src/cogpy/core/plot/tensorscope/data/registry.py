"""
TensorScope data registry.

DataRegistry provides a central location to register and retrieve data objects
used by the application (modalities, aligned time bases, derived windows, etc.).

Phase 0: Stub only.
Phase 5+: Modality-aware registry + normalization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataRegistry:
    """
    Registry of data objects for a TensorScope session.

    Phase 0 stores opaque objects keyed by name. Phase 5 will add schema-aware
    modality registration with explicit contracts.
    """

    _items: dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, obj: Any) -> None:
        """Register a data object under ``name``."""

        self._items[str(name)] = obj

    def get(self, name: str) -> Any:
        """Retrieve a registered object by name."""

        return self._items[name]

    def try_get(self, name: str, default: Any | None = None) -> Any | None:
        """Retrieve a registered object by name, returning ``default`` if missing."""

        return self._items.get(name, default)

    def keys(self) -> list[str]:
        """List registered names."""

        return sorted(self._items.keys())

