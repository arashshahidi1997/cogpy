"""TensorScope data registry package (Phase 0 scaffolding)."""

from __future__ import annotations

from .alignment import align_to_common_timebase, find_nearest_time_index
from .modality import DataModality
from .modalities import (
    FlatLFPModality,
    GridLFPModality,
    SpectrogramModality,
    SpikeTrainsModality,
)
from .registry import DataRegistry

__all__ = [
    "DataModality",
    "DataRegistry",
    "GridLFPModality",
    "FlatLFPModality",
    "SpectrogramModality",
    "SpikeTrainsModality",
    "align_to_common_timebase",
    "find_nearest_time_index",
]
