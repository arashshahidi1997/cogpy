"""Base class for detection transforms (v2.6.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import xarray as xr

__all__ = ["Transform"]


class Transform(ABC):
    """Base class for data transforms used in detection pipelines."""

    def __init__(self, name: str | None = None) -> None:
        self.name = str(name or self.__class__.__name__)
        self.params: dict[str, Any] = {}

    @abstractmethod
    def compute(self, data: xr.DataArray) -> xr.DataArray:
        """Compute transformed data."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"transform": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        params = (config or {}).get("params") or {}
        if not isinstance(params, dict):
            raise TypeError("Transform config params must be a dict")
        return cls(**params)

    def __repr__(self) -> str:
        if not self.params:
            return f"{self.name}()"
        inner = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({inner})"
