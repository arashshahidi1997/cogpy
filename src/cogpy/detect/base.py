"""
Base class for event detectors (v2.6.1).

Detectors transform data (raw signals, spectrograms, etc.) into `EventCatalog`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import xarray as xr

__all__ = ["EventDetector"]


class EventDetector(ABC):
    """Base class for event detectors."""

    def __init__(self, name: str | None = None) -> None:
        self.name = str(name or self.__class__.__name__)
        self.params: dict[str, Any] = {}

    @abstractmethod
    def detect(self, data: xr.DataArray, **kwargs) -> Any:
        """Run detection and return an `EventCatalog`."""

    @abstractmethod
    def get_event_dims(self) -> list[str]:
        """Return the dimensions events are defined over."""

    def can_accept(self, data: xr.DataArray) -> bool:
        return True

    def needs_transform(self, data: xr.DataArray) -> bool:
        return False

    def get_transform_info(self) -> dict[str, Any]:
        return {
            "required": False,
            "transform_type": None,
            "params": {},
            "implicit": False,
            "explicit": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"detector": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        params = dict((config or {}).get("params") or {})
        return cls(**params)

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({parts})"
