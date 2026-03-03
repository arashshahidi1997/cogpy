"""Event registry (Phase 1: stub, Phase 4: full implementation)."""

from __future__ import annotations

from typing import Any


class EventRegistry:
    """
    Registry for event streams.

    Phase 1: Minimal stub
    Phase 4: Full implementation
    """

    def __init__(self):
        self._streams: dict[str, Any] = {}

    def register(self, name: str, stream: Any) -> None:
        """Register event stream."""
        self._streams[str(name)] = stream

    def get(self, name: str) -> Any | None:
        """Get stream by name."""
        return self._streams.get(str(name))

    def list(self) -> list[str]:
        """List registered stream names."""
        return sorted(self._streams.keys())

    def to_dict(self) -> dict:
        """Serialize (Phase 1: minimal)."""
        return {"streams": self.list()}

