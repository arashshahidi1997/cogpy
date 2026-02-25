from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "validate_ieeg_grid",
    "validate_multichannel",
    "validate_spectrogram4d",
    "validate_burst_peaks",
    "AtlasImageOverlay",
]


def validate_ieeg_grid(da: xr.DataArray) -> xr.DataArray:
    """
    Validate IEEGGridTimeSeries schema.

    Canonical dims: ("time", "ML", "AP").
    Returns da unchanged so it can be used inline.
    """
    _check_type(da, xr.DataArray, "IEEGGridTimeSeries")
    _check_dims(
        da,
        ("time", "ML", "AP"),
        "IEEGGridTimeSeries",
        hint="sig.transpose('time', 'ML', 'AP')",
    )
    _check_coord_1d_increasing(da, "time", "IEEGGridTimeSeries")
    _check_coord_1d(da, "ML", "IEEGGridTimeSeries")
    _check_coord_1d(da, "AP", "IEEGGridTimeSeries")
    _warn_missing_attr(da, "fs", "IEEGGridTimeSeries")
    return da


def validate_multichannel(da: xr.DataArray) -> xr.DataArray:
    """
    Validate MultichannelTimeSeries schema.

    Canonical dims: ("channel", "time").
    Returns da unchanged so it can be used inline.
    """
    _check_type(da, xr.DataArray, "MultichannelTimeSeries")
    _check_dims(
        da,
        ("channel", "time"),
        "MultichannelTimeSeries",
        hint="sig.transpose('channel', 'time')",
    )
    _check_coord_1d_increasing(da, "time", "MultichannelTimeSeries")
    _warn_missing_attr(da, "fs", "MultichannelTimeSeries")
    return da


def validate_spectrogram4d(da: xr.DataArray) -> xr.DataArray:
    """
    Validate GridSpectrogram4D schema (orthoslicer-friendly).

    Canonical dims: ("ml", "ap", "time", "freq").
    Returns da unchanged so it can be used inline.
    """
    _check_type(da, xr.DataArray, "GridSpectrogram4D")
    _check_dims(da, ("ml", "ap", "time", "freq"), "GridSpectrogram4D")
    _check_coord_1d_increasing(da, "time", "GridSpectrogram4D")
    _check_coord_1d_increasing(da, "freq", "GridSpectrogram4D")
    return da


def validate_burst_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate BurstPeaksTable schema.

    Required columns: burst_id, x, y, t, z, value.
    Returns df unchanged so it can be used inline.
    """
    _check_type(df, pd.DataFrame, "BurstPeaksTable")
    required = {"burst_id", "x", "y", "t", "z", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"BurstPeaksTable missing columns: {sorted(missing)}\n"
            f"  Got: {sorted(df.columns)}"
        )
    return df


@dataclass(frozen=True)
class AtlasImageOverlay:
    """
    Bundle-safe atlas overlay: keep the image and its placement metadata together.
    """

    image: np.ndarray  # (H, W, 3) or (H, W, 4) uint8
    ap_extent: tuple[float, float]  # (ap_min, ap_max) mm bregma-relative
    ml_extent: tuple[float, float]  # (ml_min, ml_max) mm bregma-relative
    bl_distance: float = 7.5  # mm, rat default

    def __post_init__(self) -> None:
        img = np.asarray(self.image)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(
                "AtlasImageOverlay.image must be (H, W, 3) or (H, W, 4) uint8, "
                f"got shape {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"AtlasImageOverlay.image must be uint8, got {img.dtype}")


# ── Private helpers ───────────────────────────────────────────────────────────


def _check_type(obj, expected, entity: str) -> None:
    if not isinstance(obj, expected):
        raise TypeError(f"{entity} must be {expected.__name__}, got {type(obj).__name__}")


def _check_dims(da: xr.DataArray, expected: tuple[str, ...], entity: str, hint: str | None = None) -> None:
    got = tuple(da.dims)
    if got != tuple(expected):
        msg = f"{entity} must have dims {tuple(expected)}, got {got}."
        if hint:
            msg += f"\n  Hint: {hint}"
        raise ValueError(msg)


def _check_coord_1d(da: xr.DataArray, coord: str, entity: str) -> None:
    if coord not in da.coords:
        raise ValueError(f"{entity} missing coordinate {coord!r}")
    if da[coord].ndim != 1:
        raise ValueError(f"{entity} coordinate {coord!r} must be 1D")


def _check_coord_1d_increasing(da: xr.DataArray, coord: str, entity: str) -> None:
    _check_coord_1d(da, coord, entity)
    vals = np.asarray(da[coord].values)
    if vals.ndim != 1 or len(vals) < 2:
        raise ValueError(f"{entity} coordinate {coord!r} must have length >= 2")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{entity} coordinate {coord!r} contains non-finite values")
    if not np.all(np.diff(vals) > 0):
        raise ValueError(f"{entity} coordinate {coord!r} must be strictly increasing")


def _warn_missing_attr(da: xr.DataArray, attr: str, entity: str) -> None:
    if attr not in da.attrs:
        warnings.warn(f"{entity}: recommended attr {attr!r} is missing", stacklevel=3)
