"""
Data modality abstraction for multi-modal support.

A modality represents a type of data with its own:
- Sampling rate (nominal, if regularly sampled)
- Time base (timestamps)
- Data structure (xarray grids, spectrograms, spike timestamps, ...)
- Windowing/query interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataModality(ABC):
    """
    Abstract base class for TensorScope data modalities.

    Subclasses implement specific data types:
    - GridLFP: (time, AP, ML) grid data
    - FlatLFP: (time, channel) flat data
    - Spectrogram: (time, freq, AP, ML) or (time, freq, channel)
    - SpikeTrains: irregular timestamps per unit
    """

    @abstractmethod
    def time_bounds(self) -> tuple[float, float]:
        """Get valid time range (t_min, t_max) in seconds."""

    @abstractmethod
    def get_window(self, t0: float, t1: float) -> Any:
        """Get data in time window [t0, t1] in seconds."""

    @property
    @abstractmethod
    def sampling_rate(self) -> float | None:
        """Nominal sampling rate in Hz (None for irregular sampling)."""

    @property
    @abstractmethod
    def modality_type(self) -> str:
        """Modality type identifier: grid_lfp | flat_lfp | spectrogram | spikes."""

    def to_dict(self) -> dict:
        """Serialize modality metadata (not raw data)."""
        return {
            "type": self.modality_type,
            "time_bounds": self.time_bounds(),
            "sampling_rate": self.sampling_rate,
        }

