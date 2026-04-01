"""Envelope / normalization transforms for detection pipelines (v2.6.5)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..utils import hilbert_envelope, zscore_1d
from .base import Transform

__all__ = ["HilbertTransform", "ZScoreTransform"]


class HilbertTransform(Transform):
    """Hilbert envelope (magnitude)."""

    def __init__(self, *, time_dim: str = "time") -> None:
        super().__init__("HilbertTransform")
        self.time_dim = str(time_dim)
        self.params = {"time_dim": self.time_dim}

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        return hilbert_envelope(data, time_dim=self.time_dim)


class ZScoreTransform(Transform):
    """Z-score along the time axis."""

    def __init__(self, *, time_dim: str = "time") -> None:
        super().__init__("ZScoreTransform")
        self.time_dim = str(time_dim)
        self.params = {"time_dim": self.time_dim}

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        if self.time_dim not in data.dims:
            raise ValueError(
                f"Expected time_dim={self.time_dim!r} in data.dims={tuple(data.dims)}"
            )

        axis = int(data.get_axis_num(self.time_dim))
        values = data.data
        try:
            values = values.compute()
        except Exception:  # noqa: BLE001
            pass
        arr = np.asarray(values, dtype=float)

        # Apply zscore_1d along axis.
        def _z(v):
            return zscore_1d(np.asarray(v, dtype=float))

        out = np.apply_along_axis(_z, axis=axis, arr=arr)
        return xr.DataArray(
            out,
            dims=data.dims,
            coords=data.coords,
            attrs=dict(data.attrs),
            name=data.name,
        )
