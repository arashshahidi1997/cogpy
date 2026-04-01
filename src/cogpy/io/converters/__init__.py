"""
Schema-first converters and file-format conversion utilities.

This package keeps lightweight, always-available helpers (e.g. BIDS↔Zarr) and
adds optional-dependency interop converters (e.g. MNE) that import lazily.
"""

from __future__ import annotations

from .bids import bids_lfp_to_zarr, zarr_to_dat
from .mne import XarrayToMNEReport, to_mne

# Backward-compatible alias (previously exposed as `xr_to_mne`).
xr_to_mne = to_mne

__all__ = [
    "bids_lfp_to_zarr",
    "zarr_to_dat",
    "to_mne",
    "xr_to_mne",
    "XarrayToMNEReport",
]
