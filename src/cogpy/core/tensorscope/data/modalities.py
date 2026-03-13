"""Concrete data modality implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .modality import DataModality


def _sampling_rate_from_time_coord(time_vals: np.ndarray) -> float:
    tv = np.asarray(time_vals, dtype=float)
    if tv.size < 2:
        return 1.0
    dt = float(np.median(np.diff(tv)))
    if not np.isfinite(dt) or dt <= 0:
        return 1.0
    return 1.0 / dt


class GridLFPModality(DataModality):
    """
    Grid LFP data modality.

    Wraps validated (time, AP, ML) data.
    """

    def __init__(self, data: xr.DataArray):
        """
        Initialize with grid data.

        Parameters
        ----------
        data : xr.DataArray
            Validated grid data with dims (time, AP, ML)
        """
        if data.dims != ("time", "AP", "ML"):
            raise ValueError(f"Expected dims (time, AP, ML), got {data.dims}")
        self.data = data
        self._sampling_rate = _sampling_rate_from_time_coord(self.data.time.values)

    def time_bounds(self) -> tuple[float, float]:
        """Get time range."""
        return (
            float(self.data.time.values[0]),
            float(self.data.time.values[-1]),
        )

    def get_window(self, t0: float, t1: float) -> xr.DataArray:
        """Get grid data in time window (time, AP, ML)."""
        return self.data.sel(time=slice(float(t0), float(t1)))

    @property
    def sampling_rate(self) -> float:
        """Sampling rate in Hz."""
        return float(self._sampling_rate)

    @property
    def modality_type(self) -> str:
        return "grid_lfp"

    def to_flat(self) -> "FlatLFPModality":
        """Convert to flat (time, channel) representation."""
        from ..schema import flatten_grid_to_channels

        flat_data = flatten_grid_to_channels(self.data)
        return FlatLFPModality(flat_data)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(
            {
                "shape": dict(self.data.sizes),
                "grid_size": (int(self.data.sizes["AP"]), int(self.data.sizes["ML"])),
            }
        )
        return base


class FlatLFPModality(DataModality):
    """Flat LFP modality with dims (time, channel)."""

    def __init__(self, data: xr.DataArray):
        if data.dims != ("time", "channel"):
            raise ValueError(f"Expected dims (time, channel), got {data.dims}")
        self.data = data
        self._sampling_rate = _sampling_rate_from_time_coord(self.data.time.values)

    def time_bounds(self) -> tuple[float, float]:
        return (float(self.data.time.values[0]), float(self.data.time.values[-1]))

    def get_window(self, t0: float, t1: float) -> xr.DataArray:
        return self.data.sel(time=slice(float(t0), float(t1)))

    @property
    def sampling_rate(self) -> float:
        return float(self._sampling_rate)

    @property
    def modality_type(self) -> str:
        return "flat_lfp"

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({"shape": dict(self.data.sizes), "n_channels": int(self.data.sizes["channel"])})
        return base


class SpectrogramModality(DataModality):
    """
    Spectrogram modality.

    Accepts either:
    - (time, freq, AP, ML)
    - (time, freq, channel)
    """

    _VALID_DIMS: tuple[tuple[str, ...], ...] = (
        ("time", "freq", "AP", "ML"),
        ("time", "freq", "channel"),
    )

    def __init__(self, data: xr.DataArray):
        if data.dims not in self._VALID_DIMS:
            raise ValueError(f"Expected dims {list(self._VALID_DIMS)}, got {data.dims}")
        self.data = data
        self._sampling_rate = _sampling_rate_from_time_coord(self.data.time.values)

    def time_bounds(self) -> tuple[float, float]:
        return (float(self.data.time.values[0]), float(self.data.time.values[-1]))

    def get_window(self, t0: float, t1: float) -> xr.DataArray:
        return self.data.sel(time=slice(float(t0), float(t1)))

    @property
    def sampling_rate(self) -> float:
        return float(self._sampling_rate)

    @property
    def modality_type(self) -> str:
        return "spectrogram"

    def freq_bounds(self) -> tuple[float, float]:
        return (float(self.data.freq.values[0]), float(self.data.freq.values[-1]))

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({"shape": dict(self.data.sizes), "freq_bounds": self.freq_bounds()})
        return base


@dataclass(frozen=True, slots=True)
class SpikeUnit:
    """Single unit spike timestamps (seconds)."""

    unit_id: str
    times_s: np.ndarray


class SpikeTrainsModality(DataModality):
    """
    Spike trains modality.

    Represents irregular spike timestamps per unit.
    """

    def __init__(self, units: list[SpikeUnit] | dict[str, np.ndarray]):
        if isinstance(units, dict):
            unit_list = [SpikeUnit(str(k), np.asarray(v, dtype=float)) for k, v in units.items()]
        else:
            unit_list = [SpikeUnit(str(u.unit_id), np.asarray(u.times_s, dtype=float)) for u in units]

        self.units: list[SpikeUnit] = unit_list

    def time_bounds(self) -> tuple[float, float]:
        if not self.units:
            return (0.0, 0.0)
        t_min = min(float(u.times_s[0]) for u in self.units if u.times_s.size)
        t_max = max(float(u.times_s[-1]) for u in self.units if u.times_s.size)
        return (t_min, t_max)

    def get_window(self, t0: float, t1: float) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        lo = float(t0)
        hi = float(t1)
        for u in self.units:
            ts = u.times_s
            if ts.size == 0:
                out[u.unit_id] = ts
                continue
            i0 = int(np.searchsorted(ts, lo, side="left"))
            i1 = int(np.searchsorted(ts, hi, side="right"))
            out[u.unit_id] = ts[i0:i1]
        return out

    @property
    def sampling_rate(self) -> float | None:
        return None

    @property
    def modality_type(self) -> str:
        return "spikes"

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({"n_units": int(len(self.units))})
        return base
