"""Spectral transforms for detection pipelines (v2.6.5)."""

from __future__ import annotations

import xarray as xr

from .base import Transform

__all__ = ["SpectrogramTransform"]


class SpectrogramTransform(Transform):
    """Compute spectrogram via `cogpy.spectral.specx.spectrogramx`."""

    def __init__(
        self,
        *,
        nperseg: int = 256,
        noverlap: int | None = None,
        bandwidth: float = 4.0,
        axis: str = "time",
    ) -> None:
        super().__init__("SpectrogramTransform")
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap) if noverlap is not None else int(nperseg) // 2
        self.bandwidth = float(bandwidth)
        self.axis = str(axis)
        self.params = {
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "bandwidth": self.bandwidth,
            "axis": self.axis,
        }

    def compute(self, data: xr.DataArray) -> xr.DataArray:
        from cogpy.spectral.specx import spectrogramx

        return spectrogramx(
            data,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            bandwidth=self.bandwidth,
            axis=self.axis,
        )
