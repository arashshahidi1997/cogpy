from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "DIMS_IEEG_GRID",
    "DIMS_MULTICHANNEL",
    "DIMS_SPECTROGRAM4D",
    "DIMS_IEEG_TIME_CHANNEL",
    "validate_ieeg_grid",
    "validate_multichannel",
    "validate_spectrogram4d",
    "validate_burst_peaks",
    "validate_ieeg_time_channel",
    "coerce_ieeg_grid",
    "coerce_multichannel",
    "coerce_spectrogram4d",
    "coerce_ieeg_time_channel",
    "assert_attrs_survive",
    "AtlasImageOverlay",
]

# Canonical dim orders (single source of truth for validators and docs).
DIMS_IEEG_GRID = ("time", "ML", "AP")
DIMS_MULTICHANNEL = ("channel", "time")
# Keep orthoslicer-friendly order consistent with existing code.
DIMS_SPECTROGRAM4D = ("ml", "ap", "time", "freq")
DIMS_IEEG_TIME_CHANNEL = ("time", "channel")


def validate_ieeg_grid(da: xr.DataArray) -> xr.DataArray:
    """
    Validate IEEGGridTimeSeries schema.

    Canonical dims: ("time", "ML", "AP").
    Returns da unchanged so it can be used inline.
    """
    _check_type(da, xr.DataArray, "IEEGGridTimeSeries")
    _check_dims(
        da,
        DIMS_IEEG_GRID,
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
        DIMS_MULTICHANNEL,
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
    _check_dims(da, DIMS_SPECTROGRAM4D, "GridSpectrogram4D")
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


def validate_ieeg_time_channel(da: xr.DataArray) -> xr.DataArray:
    """
    Validate an iEEG time×channel view intended for stacked-trace viewers.

    Canonical dims: ("time","channel").

    Supported channel coordinate forms
    -------------------------------
    1) **Recommended** (reset-index): `channel` is a flat integer coordinate and
       `AP` and `ML` are 1D coords aligned to `channel` (as produced by
       `stack(...).reset_index("channel")`).
    2) **Allowed** (MultiIndex): `channel` is a pandas.MultiIndex with level
       names including "AP" and "ML".
    """
    _check_type(da, xr.DataArray, "IEEGTimeChannel")
    _check_dims(
        da,
        DIMS_IEEG_TIME_CHANNEL,
        "IEEGTimeChannel",
        hint="sig.transpose('time','AP','ML').stack(channel=('AP','ML')).reset_index('channel')",
    )
    _check_coord_1d_increasing(da, "time", "IEEGTimeChannel")

    if "channel" in da.coords and da["channel"].ndim != 1:
        raise ValueError("IEEGTimeChannel coordinate 'channel' must be 1D when present")

    # Recommended form: AP and ML present as per-channel coords after reset_index.
    if "AP" in da.coords and "ML" in da.coords and da["AP"].dims == ("channel",) and da["ML"].dims == ("channel",):
        return da

    # Allowed fallback: MultiIndex channel.
    try:
        if "channel" not in da.coords:
            raise ValueError("no channel coordinate")
        idx = da["channel"].to_index()
        names = set(getattr(idx, "names", []) or [])
        if {"AP", "ML"}.issubset(names):
            return da
    except Exception:  # noqa: BLE001
        pass

    raise ValueError(
        "IEEGTimeChannel channel coordinate must either:\n"
        "  (a) have coords AP and ML aligned to channel (recommended), or\n"
        "  (b) be a MultiIndex with levels named AP and ML.\n"
        "  Hint: sig.stack(channel=('AP','ML')).reset_index('channel')"
    )


def coerce_ieeg_grid(
    da: xr.DataArray,
    *,
    fs: float | None = None,
    ap_coords: np.ndarray | None = None,
    ml_coords: np.ndarray | None = None,
    time_coords: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Best-effort coercion to IEEGGridTimeSeries schema.

    - Transposes to ("time","ML","AP") if dims are a permutation
    - Injects fs into attrs if provided and missing
    - Raises if dims are not a permutation of the canonical set
    """
    _check_type(da, xr.DataArray, "IEEGGridTimeSeries")
    if set(da.dims) != set(DIMS_IEEG_GRID):
        raise ValueError(
            f"IEEGGridTimeSeries must have dims {DIMS_IEEG_GRID} (permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != DIMS_IEEG_GRID:
        da = da.transpose(*DIMS_IEEG_GRID)

    if fs is None and "fs" not in da.attrs:
        fs = _maybe_extract_fs(da.attrs)
    if fs is not None and "fs" not in da.attrs:
        da = da.assign_attrs(fs=float(fs))

    # Inject missing coords if needed.
    n_time = int(da.sizes["time"])
    n_ml = int(da.sizes["ML"])
    n_ap = int(da.sizes["AP"])

    if "time" not in da.coords:
        if time_coords is not None:
            t = np.asarray(time_coords, dtype=float)
            if t.shape != (n_time,):
                raise ValueError(f"time_coords must have shape ({n_time},), got {t.shape}")
        else:
            fs_use = da.attrs.get("fs", None)
            if fs_use is not None:
                t = np.arange(n_time, dtype=float) / float(fs_use)
            else:
                t = np.arange(n_time, dtype=float)
        da = da.assign_coords(time=t)

    if "ML" not in da.coords:
        if ml_coords is not None:
            ml = np.asarray(ml_coords, dtype=float)
            if ml.shape != (n_ml,):
                raise ValueError(f"ml_coords must have shape ({n_ml},), got {ml.shape}")
        else:
            ml = np.arange(n_ml, dtype=float)
        da = da.assign_coords(ML=ml)

    if "AP" not in da.coords:
        if ap_coords is not None:
            ap = np.asarray(ap_coords, dtype=float)
            if ap.shape != (n_ap,):
                raise ValueError(f"ap_coords must have shape ({n_ap},), got {ap.shape}")
        else:
            ap = np.arange(n_ap, dtype=float)
        da = da.assign_coords(AP=ap)

    return validate_ieeg_grid(da)


def coerce_multichannel(
    da: xr.DataArray,
    *,
    fs: float | None = None,
    time_coords: np.ndarray | None = None,
    channel_coords: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Best-effort coercion to MultichannelTimeSeries schema.

    - Transposes to ("channel","time") if dims are a permutation
    - Injects fs into attrs if provided and missing
    """
    _check_type(da, xr.DataArray, "MultichannelTimeSeries")
    if set(da.dims) != set(DIMS_MULTICHANNEL):
        raise ValueError(
            f"MultichannelTimeSeries must have dims {DIMS_MULTICHANNEL} (permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != DIMS_MULTICHANNEL:
        da = da.transpose(*DIMS_MULTICHANNEL)

    if fs is None and "fs" not in da.attrs:
        fs = _maybe_extract_fs(da.attrs)
    if fs is not None and "fs" not in da.attrs:
        da = da.assign_attrs(fs=float(fs))

    n_ch = int(da.sizes["channel"])
    n_time = int(da.sizes["time"])

    if "time" not in da.coords:
        if time_coords is not None:
            t = np.asarray(time_coords, dtype=float)
            if t.shape != (n_time,):
                raise ValueError(f"time_coords must have shape ({n_time},), got {t.shape}")
        else:
            fs_use = da.attrs.get("fs", None)
            if fs_use is not None:
                t = np.arange(n_time, dtype=float) / float(fs_use)
            else:
                t = np.arange(n_time, dtype=float)
        da = da.assign_coords(time=t)

    if "channel" not in da.coords:
        if channel_coords is not None:
            ch = np.asarray(channel_coords)
            if ch.shape != (n_ch,):
                raise ValueError(f"channel_coords must have shape ({n_ch},), got {ch.shape}")
        else:
            ch = np.arange(n_ch)
        da = da.assign_coords(channel=ch)

    return validate_multichannel(da)


def coerce_spectrogram4d(da: xr.DataArray) -> xr.DataArray:
    """
    Best-effort coercion to GridSpectrogram4D schema.

    - Transposes to ("ml","ap","time","freq") if dims are a permutation
    """
    _check_type(da, xr.DataArray, "GridSpectrogram4D")
    if set(da.dims) != set(DIMS_SPECTROGRAM4D):
        raise ValueError(
            f"GridSpectrogram4D must have dims {DIMS_SPECTROGRAM4D} (permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != DIMS_SPECTROGRAM4D:
        da = da.transpose(*DIMS_SPECTROGRAM4D)
    return validate_spectrogram4d(da)


def coerce_ieeg_time_channel(da: xr.DataArray, *, fs: float | None = None) -> xr.DataArray:
    """
    Best-effort coercion to IEEGTimeChannel schema.

    - Transposes to ("time","channel") if dims are a permutation
    - If channel is a MultiIndex, resets it to a flat integer channel coord with
      `AP` and `ML` as auxiliary coords (preferred canonical form).
    - Injects fs into attrs if provided and missing
    """
    _check_type(da, xr.DataArray, "IEEGTimeChannel")
    if set(da.dims) != set(DIMS_IEEG_TIME_CHANNEL):
        raise ValueError(
            f"IEEGTimeChannel must have dims {DIMS_IEEG_TIME_CHANNEL} (permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != DIMS_IEEG_TIME_CHANNEL:
        da = da.transpose(*DIMS_IEEG_TIME_CHANNEL)

    # Prefer reset-index representation for stability across xarray versions.
    try:
        idx = da["channel"].to_index()
        if getattr(idx, "names", None) and len(getattr(idx, "names")) > 1:
            da = da.reset_index("channel")
    except Exception:  # noqa: BLE001
        pass

    if fs is None and "fs" not in da.attrs:
        fs = _maybe_extract_fs(da.attrs)
    if fs is not None and "fs" not in da.attrs:
        da = da.assign_attrs(fs=float(fs))

    return validate_ieeg_time_channel(da)


def assert_attrs_survive(da: xr.DataArray, required: list[str]) -> None:
    """
    Use in tests to catch operations that silently drop attrs.
    """
    missing = [k for k in required if k not in da.attrs]
    if missing:
        raise AssertionError(
            f"Attrs lost after operation: {missing}\n"
            f"  Remaining: {list(da.attrs.keys())}"
        )


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


def _maybe_extract_fs(attrs: dict) -> float | None:
    """
    Best-effort extraction of sampling rate from common attribute layouts.
    """
    candidates = ("fs", "Fs", "sampling_rate", "SamplingRate", "sampling_frequency", "SamplingFrequency")
    for k in candidates:
        if k in attrs:
            try:
                return float(attrs[k])
            except Exception:  # noqa: BLE001
                continue

    for container_key in ("meta", "metadata"):
        meta = attrs.get(container_key, None)
        if isinstance(meta, dict):
            for k in candidates:
                if k in meta:
                    try:
                        return float(meta[k])
                    except Exception:  # noqa: BLE001
                        continue

    return None
