"""Data modality adapters (Phase 1: minimal GridLFP only)."""

from __future__ import annotations

import xarray as xr


class GridLFPModality:
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
        self.data = data

    def time_bounds(self) -> tuple[float, float]:
        """Get time range."""
        return (
            float(self.data.time.values[0]),
            float(self.data.time.values[-1]),
        )

    def to_dict(self) -> dict:
        """Serialize modality metadata (not data)."""
        return {
            "type": "GridLFP",
            "shape": dict(self.data.sizes),
            "time_bounds": self.time_bounds(),
        }

