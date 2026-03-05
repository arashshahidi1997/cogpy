"""
BurstDetector (v2.6.1).

Wraps `cogpy.core.burst.blob_detection.detect_hmaxima` and returns
`cogpy.core.events.EventCatalog`.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from cogpy.core.events import EventCatalog
from cogpy.core.spectral.specx import spectrogramx

from .base import EventDetector

__all__ = ["BurstDetector"]


class BurstDetector(EventDetector):
    """
    Detect burst peaks using an h-maxima transform.

    Accepts:
    - Spectrogram-like inputs (must include a 'freq' dimension)  -> explicit mode
    - Raw time-series inputs (has 'time' but no 'freq' dimension) -> implicit mode (computes spectrogram)
    """

    def __init__(
        self,
        *,
        h_quantile: float = 0.9,
        h: float | None = None,
        nperseg: int = 256,
        noverlap: int | None = 128,
        bandwidth: float = 4.0,
        footprint=None,
    ) -> None:
        super().__init__("BurstDetector")

        self.h_quantile = float(h_quantile)
        self.h = float(h) if h is not None else None
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap) if noverlap is not None else int(self.nperseg) // 2
        self.bandwidth = float(bandwidth)
        self.footprint = footprint

        self.params = {
            "h_quantile": self.h_quantile,
            "h": self.h,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "bandwidth": self.bandwidth,
            "footprint": self.footprint,
        }

    def can_accept(self, data: xr.DataArray) -> bool:
        dims = set(getattr(data, "dims", ()) or ())
        if "time" not in dims:
            return False
        is_spectrogram = "freq" in dims
        is_raw = "freq" not in dims
        return is_spectrogram or is_raw

    def needs_transform(self, data: xr.DataArray) -> bool:
        return "freq" not in set(getattr(data, "dims", ()) or ())

    def detect(self, data: xr.DataArray, **kwargs: Any) -> EventCatalog:
        from cogpy.core.burst.blob_detection import detect_hmaxima

        if not isinstance(data, xr.DataArray):
            raise TypeError(f"BurstDetector.detect expects xr.DataArray, got {type(data).__name__}")
        if not self.can_accept(data):
            raise ValueError(f"BurstDetector cannot accept data with dims={tuple(getattr(data, 'dims', ()))!r}")

        computed_spec = False
        spec = data
        if self.needs_transform(data):
            computed_spec = True
            spec = spectrogramx(
                data,
                axis="time",
                bandwidth=float(self.bandwidth),
                nperseg=int(self.nperseg),
                noverlap=int(self.noverlap),
            )

        # `detect_hmaxima` uses NumPy + pandas; ensure any lazy/dask arrays are materialized.
        try:
            spec = spec.compute()
        except Exception:  # noqa: BLE001
            pass

        peaks_df = detect_hmaxima(
            spec,
            h_quantile=float(kwargs.get("h_quantile", self.h_quantile)),
            h=kwargs.get("h", self.h),
            footprint=kwargs.get("footprint", self.footprint),
        )

        meta = dict(self.params)
        meta.update(
            {
                "detector": self.name,
                "computed_spectrogram": bool(computed_spec),
            }
        )

        return EventCatalog.from_hmaxima(
            peaks_df,
            label=str(kwargs.get("label", "burst_peak")),
            **meta,
        )

    def get_event_dims(self) -> list[str]:
        return ["time", "freq", "AP", "ML"]

    def get_transform_info(self) -> dict[str, Any]:
        return {
            "required": True,
            "transform_type": "spectrogramx",
            "params": {
                "nperseg": int(self.nperseg),
                "noverlap": int(self.noverlap),
                "bandwidth": float(self.bandwidth),
            },
            "implicit": True,
            "explicit": True,
        }
