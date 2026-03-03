"""Event registry for managing multiple event streams."""

from __future__ import annotations

from .model import EventStream


class EventRegistry:
    """
    Registry for event streams.

    Manages multiple EventStream instances (bursts, ripples, etc.).
    """

    def __init__(self):
        self._streams: dict[str, EventStream] = {}

    def register(self, stream: EventStream) -> None:
        self._streams[str(stream.name)] = stream

    def get(self, name: str) -> EventStream | None:
        return self._streams.get(str(name))

    def list(self) -> list[str]:
        return sorted(self._streams.keys())

    def remove(self, name: str) -> None:
        self._streams.pop(str(name), None)

    def to_dict(self) -> dict:
        return {"streams": {name: stream.to_dict() for name, stream in self._streams.items()}}

