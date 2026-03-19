"""Filtering transforms for detection pipelines (v2.6.5)."""

from __future__ import annotations

import xarray as xr

from .base import Transform

__all__ = ["BandpassTransform", "HighpassTransform", "LowpassTransform"]


class BandpassTransform(Transform):
    """Bandpass filter using `cogpy.preprocess.filtx.bandpassx`."""

    def __init__(self, *, low: float, high: float, order: int = 4, axis: str = "time") -> None:
        super().__init__("BandpassTransform")
        self.low = float(low)
        self.high = float(high)
        self.order = int(order)
        self.axis = str(axis)
        self.params = {"low": self.low, "high": self.high, "order": self.order, "axis": self.axis}

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        from cogpy.preprocess.filtx import bandpassx

        return bandpassx(data, self.low, self.high, self.order, axis=self.axis)


class HighpassTransform(Transform):
    """Highpass filter using `cogpy.preprocess.filtx.highpassx`."""

    def __init__(self, *, cutoff: float, order: int = 4, axis: str = "time") -> None:
        super().__init__("HighpassTransform")
        self.cutoff = float(cutoff)
        self.order = int(order)
        self.axis = str(axis)
        self.params = {"cutoff": self.cutoff, "order": self.order, "axis": self.axis}

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        from cogpy.preprocess.filtx import highpassx

        return highpassx(data, self.cutoff, self.order, axis=self.axis)


class LowpassTransform(Transform):
    """Lowpass filter using `cogpy.preprocess.filtx.lowpassx`."""

    def __init__(self, *, cutoff: float, order: int = 4, axis: str = "time") -> None:
        super().__init__("LowpassTransform")
        self.cutoff = float(cutoff)
        self.order = int(order)
        self.axis = str(axis)
        self.params = {"cutoff": self.cutoff, "order": self.order, "axis": self.axis}

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        from cogpy.preprocess.filtx import lowpassx

        return lowpassx(data, self.cutoff, self.order, axis=self.axis)

