from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "DIMS_IEEG_GRID",
    "DIMS_IEEG_GRID_WINDOW",
    "DIMS_MULTICHANNEL",
    "DIMS_MULTICHANNEL_WINDOW",
    "DIMS_SPECTROGRAM4D",
    "DIMS_IEEG_TIME_CHANNEL",
    "DIMS_CHANNEL_FEATURE_MAP",
    "DIMS_GRID_FEATURE_MAP",
    "DIMS_CHANNEL_SPECTRUM",
    "DIMS_GRID_SPECTRUM",
    "DIMS_CHANNEL_WINDOWED_SPECTRUM",
    "DIMS_GRID_WINDOWED_SPECTRUM",
    "DIMS_PAIRWISE_FEATURE_MATRIX",
    "DIMS_PAIRWISE_SPECTRUM",
    "DIMS_COMODULOGRAM",
    "DIMS_SPATIAL_COHERENCE_PROFILE",
    "validate_ieeg_grid",
    "validate_ieeg_grid_windowed",
    "validate_multichannel",
    "validate_multichannel_windowed",
    "validate_spectrogram4d",
    "validate_burst_peaks",
    "validate_ieeg_time_channel",
    "coerce_ieeg_grid",
    "coerce_ieeg_grid_windowed",
    "coerce_multichannel",
    "coerce_multichannel_windowed",
    "coerce_spectrogram4d",
    "coerce_ieeg_time_channel",
    "assert_attrs_survive",
    "AtlasImageOverlay",
]

# Canonical dim orders (single source of truth for validators and docs).
DIMS_IEEG_GRID = ("time", "ML", "AP")
DIMS_IEEG_GRID_WINDOW = ("time_win", "ML", "AP")
DIMS_MULTICHANNEL = ("channel", "time")
DIMS_MULTICHANNEL_WINDOW = ("time_win", "channel")
# Keep orthoslicer-friendly order consistent with existing code.
DIMS_SPECTROGRAM4D = ("ml", "ap", "time", "freq")
DIMS_IEEG_TIME_CHANNEL = ("time", "channel")

# Feature output schemas (validate/coerce added on first pipeline use)
DIMS_CHANNEL_FEATURE_MAP = ("channel",)
DIMS_GRID_FEATURE_MAP = ("AP", "ML")
DIMS_CHANNEL_SPECTRUM = ("channel", "freq")
DIMS_GRID_SPECTRUM = ("AP", "ML", "freq")
DIMS_CHANNEL_WINDOWED_SPECTRUM = ("time_win", "channel", "freq")
DIMS_GRID_WINDOWED_SPECTRUM = ("time_win", "AP", "ML", "freq")
DIMS_PAIRWISE_FEATURE_MATRIX = ("channel_i", "channel_j")
DIMS_PAIRWISE_SPECTRUM = ("channel_i", "channel_j", "freq")
DIMS_COMODULOGRAM = ("channel", "freq_phase", "freq_amp")
DIMS_SPATIAL_COHERENCE_PROFILE = ("distance_bin", "freq")


def validate_ieeg_grid(da: xr.DataArray, *, required_attrs: tuple[str, ...] = ()) -> xr.DataArray:
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
    _check_required_attrs(da, required_attrs, "IEEGGridTimeSeries")
    return da


def validate_ieeg_grid_windowed(
    da: xr.DataArray,
    *,
    win_dim: str = "time_win",
    required_attrs: tuple[str, ...] = (),
) -> xr.DataArray:
    """
    Validate IEEGGridWindowed schema (sliding-window grid features).

    Canonical dims: (win_dim, "ML", "AP") with default win_dim="time_win".
    Returns da unchanged so it can be used inline.

    Notes
    -----
    This schema is intended for per-window feature maps derived from an
    ``IEEGGridTimeSeries`` input. Typical attrs include:
      - fs (recommended)
      - window_size, window_step (recommended)
      - center_method, run_dim (optional/recommended depending on producer)
    """
    entity = "IEEGGridWindowed"
    _check_type(da, xr.DataArray, entity)
    expected = (str(win_dim), "ML", "AP")
    got = tuple(da.dims)
    if got != expected:
        raise ValueError(f"{entity} must have dims {expected}, got {got}.")
    _check_coord_1d_increasing(da, str(win_dim), entity)
    _check_coord_1d(da, "ML", entity)
    _check_coord_1d(da, "AP", entity)
    _warn_missing_attr(da, "fs", entity)
    _warn_missing_attr(da, "window_size", entity)
    _warn_missing_attr(da, "window_step", entity)
    _check_required_attrs(da, required_attrs, entity)
    return da


def validate_multichannel(da: xr.DataArray, *, required_attrs: tuple[str, ...] = ()) -> xr.DataArray:
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
    _check_required_attrs(da, required_attrs, "MultichannelTimeSeries")
    return da


def validate_multichannel_windowed(
    da: xr.DataArray,
    *,
    win_dim: str = "time_win",
    required_attrs: tuple[str, ...] = (),
) -> xr.DataArray:
    """
    Validate MultichannelWindowed schema (sliding-window channel features).

    Canonical dims: (win_dim, "channel") with default win_dim="time_win".
    Returns da unchanged so it can be used inline.
    """
    entity = "MultichannelWindowed"
    _check_type(da, xr.DataArray, entity)
    expected = (str(win_dim), "channel")
    got = tuple(da.dims)
    if got != expected:
        raise ValueError(f"{entity} must have dims {expected}, got {got}.")
    _check_coord_1d_increasing(da, str(win_dim), entity)
    _check_coord_1d(da, "channel", entity)
    _warn_missing_attr(da, "fs", entity)
    _warn_missing_attr(da, "window_size", entity)
    _warn_missing_attr(da, "window_step", entity)
    _check_required_attrs(da, required_attrs, entity)
    return da


def validate_spectrogram4d(da: xr.DataArray, *, required_attrs: tuple[str, ...] = ()) -> xr.DataArray:
    """
    Validate GridSpectrogram4D schema (orthoslicer-friendly).

    Canonical dims: ("ml", "ap", "time", "freq").
    Returns da unchanged so it can be used inline.
    """
    _check_type(da, xr.DataArray, "GridSpectrogram4D")
    _check_dims(da, DIMS_SPECTROGRAM4D, "GridSpectrogram4D")
    _check_coord_1d_increasing(da, "time", "GridSpectrogram4D")
    _check_coord_1d_increasing(da, "freq", "GridSpectrogram4D")
    _check_required_attrs(da, required_attrs, "GridSpectrogram4D")
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


def validate_ieeg_time_channel(da: xr.DataArray, *, required_attrs: tuple[str, ...] = ()) -> xr.DataArray:
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
    3) **Allowed** (flat channel): `channel` is a simple 1D coordinate without
       spatial metadata.
    """
    _check_type(da, xr.DataArray, "IEEGTimeChannel")
    _check_dims(
        da,
        DIMS_IEEG_TIME_CHANNEL,
        "IEEGTimeChannel",
        hint="sig.transpose('time','AP','ML').stack(channel=('AP','ML')).reset_index('channel')",
    )
    _check_coord_1d_increasing(da, "time", "IEEGTimeChannel")
    _warn_missing_attr(da, "fs", "IEEGTimeChannel")
    _check_required_attrs(da, required_attrs, "IEEGTimeChannel")

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

    # Allowed: plain 1D channel coordinate (no AP/ML metadata).
    if "channel" in da.coords and da["channel"].ndim == 1:
        warnings.warn(
            "IEEGTimeChannel: channel coordinate has no AP/ML metadata; accepting flat channel coordinate.",
            stacklevel=3,
        )
        return da

    raise ValueError(
        "IEEGTimeChannel channel coordinate must either:\n"
        "  (a) have coords AP and ML aligned to channel (recommended), or\n"
        "  (b) be a MultiIndex with levels named AP and ML.\n"
        "  (c) be a flat 1D channel coordinate (allowed).\n"
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


def coerce_ieeg_grid_windowed(
    da: xr.DataArray,
    *,
    win_dim: str = "time_win",
    win_coords: np.ndarray | None = None,
    ap_coords: np.ndarray | None = None,
    ml_coords: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Best-effort coercion to IEEGGridWindowed schema.

    - Transposes to (win_dim,"ML","AP") if dims are a permutation
    - Injects missing coords (win_dim/ML/AP) if provided, else uses arange
    """
    entity = "IEEGGridWindowed"
    _check_type(da, xr.DataArray, entity)
    win_dim = str(win_dim)
    expected_set = {win_dim, "ML", "AP"}
    if set(da.dims) != expected_set:
        raise ValueError(
            f"{entity} must have dims ({win_dim!r}, 'ML', 'AP') (permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != (win_dim, "ML", "AP"):
        da = da.transpose(win_dim, "ML", "AP")

    n_win = int(da.sizes[win_dim])
    n_ml = int(da.sizes["ML"])
    n_ap = int(da.sizes["AP"])

    if win_dim not in da.coords:
        if win_coords is not None:
            w = np.asarray(win_coords, dtype=float)
            if w.shape != (n_win,):
                raise ValueError(f"win_coords must have shape ({n_win},), got {w.shape}")
        else:
            w = np.arange(n_win, dtype=float)
        da = da.assign_coords(**{win_dim: w})

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

    return validate_ieeg_grid_windowed(da, win_dim=win_dim)


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


def coerce_multichannel_windowed(
    da: xr.DataArray,
    *,
    win_dim: str = "time_win",
    win_coords: np.ndarray | None = None,
    channel_coords: np.ndarray | None = None,
) -> xr.DataArray:
    """
    Best-effort coercion to MultichannelWindowed schema.

    - Transposes to (win_dim,"channel") if dims are a permutation
    - Injects missing coords (win_dim/channel) if provided, else uses arange
    """
    entity = "MultichannelWindowed"
    _check_type(da, xr.DataArray, entity)
    win_dim = str(win_dim)
    expected_set = {win_dim, "channel"}
    if set(da.dims) != expected_set:
        raise ValueError(
            f"{entity} must have dims ({win_dim!r}, 'channel') (permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != (win_dim, "channel"):
        da = da.transpose(win_dim, "channel")

    n_win = int(da.sizes[win_dim])
    n_ch = int(da.sizes["channel"])

    if win_dim not in da.coords:
        if win_coords is not None:
            w = np.asarray(win_coords, dtype=float)
            if w.shape != (n_win,):
                raise ValueError(f"win_coords must have shape ({n_win},), got {w.shape}")
        else:
            w = np.arange(n_win, dtype=float)
        da = da.assign_coords(**{win_dim: w})

    if "channel" not in da.coords:
        if channel_coords is not None:
            ch = np.asarray(channel_coords)
            if ch.shape != (n_ch,):
                raise ValueError(f"channel_coords must have shape ({n_ch},), got {ch.shape}")
        else:
            ch = np.arange(n_ch)
        da = da.assign_coords(channel=ch)

    return validate_multichannel_windowed(da, win_dim=win_dim)


def coerce_spectrogram4d(da: xr.DataArray) -> xr.DataArray:
    """
    Best-effort coercion to GridSpectrogram4D schema.

    - Transposes to ("ml","ap","time","freq") if dims are a permutation
    """
    _check_type(da, xr.DataArray, "GridSpectrogram4D")
    # Accept uppercase variants commonly used for grid data and coerce to canonical lowercase.
    if set(da.dims) == {"ML", "AP", "time", "freq"}:
        da = da.rename({"ML": "ml", "AP": "ap"})
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
    except (KeyError, AttributeError, TypeError, ValueError):
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

    Notes
    -----
    The `image` is defensively copied and marked read-only in `__post_init__` so
    callers cannot mutate it in-place while it's being shared through GUI bundles.
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
        img = np.array(img, copy=True)
        img.setflags(write=False)
        object.__setattr__(self, "image", img)


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
    if vals.ndim != 1 or len(vals) < 1:
        raise ValueError(f"{entity} coordinate {coord!r} must have length >= 1")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{entity} coordinate {coord!r} contains non-finite values")
    if len(vals) > 1 and not np.all(np.diff(vals) > 0):
        raise ValueError(f"{entity} coordinate {coord!r} must be strictly increasing")


def _warn_missing_attr(da: xr.DataArray, attr: str, entity: str) -> None:
    if attr not in da.attrs:
        warnings.warn(f"{entity}: recommended attr {attr!r} is missing", stacklevel=3)


def _check_required_attrs(da: xr.DataArray, required: tuple[str, ...], entity: str) -> None:
    missing = [k for k in required if k not in da.attrs]
    if missing:
        raise ValueError(f"{entity} missing required attrs: {missing}")


def _maybe_extract_fs(attrs: dict) -> float | None:
    """
    Best-effort extraction of sampling rate from common attribute layouts.
    """
    candidates = ("fs", "Fs", "sampling_rate", "SamplingRate", "sampling_frequency", "SamplingFrequency")
    for k in candidates:
        if k in attrs:
            try:
                return float(attrs[k])
            except (TypeError, ValueError):
                continue

    for container_key in ("meta", "metadata"):
        meta = attrs.get(container_key, None)
        if isinstance(meta, dict):
            for k in candidates:
                if k in meta:
                    try:
                        return float(meta[k])
                    except (TypeError, ValueError):
                        continue

    return None
