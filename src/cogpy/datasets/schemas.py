from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "DIMS_EVENT_CATALOG",
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
    "DIMS_EVENTS_TABLE",
    "DIMS_INTERVALS_TABLE",
    "DIMS_PERIEVENT",
    "EventCatalog",
    "validate_event_catalog",
    "coerce_event_catalog",
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
    "validate_grid_windowed_spectrum",
    "coerce_grid_windowed_spectrum",
    "coerce_ieeg_time_channel",
    "assert_attrs_survive",
    "AtlasImageOverlay",
    "Intervals",
    "Events",
]

# Canonical dim orders (single source of truth for validators and docs).
DIMS_EVENT_CATALOG = ("event",)
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
DIMS_EVENTS_TABLE = ("time",)
DIMS_INTERVALS_TABLE = ("t_start", "t_end")
DIMS_PERIEVENT = ("event", "channel", "time")


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


def validate_grid_windowed_spectrum(
    da: xr.DataArray, *, required_attrs: tuple[str, ...] = ()
) -> xr.DataArray:
    """
    Validate GridWindowedSpectrum schema (compute-oriented, uppercase).

    Canonical dims: ("time_win", "AP", "ML", "freq").
    Returns da unchanged so it can be used inline.
    """
    _check_type(da, xr.DataArray, "GridWindowedSpectrum")
    _check_dims(
        da,
        DIMS_GRID_WINDOWED_SPECTRUM,
        "GridWindowedSpectrum",
        hint="coerce_grid_windowed_spectrum(da)",
    )
    _check_coord_1d_increasing(da, "time_win", "GridWindowedSpectrum")
    _check_coord_1d_increasing(da, "freq", "GridWindowedSpectrum")
    _check_coord_1d(da, "AP", "GridWindowedSpectrum")
    _check_coord_1d(da, "ML", "GridWindowedSpectrum")
    _check_required_attrs(da, required_attrs, "GridWindowedSpectrum")
    return da


def coerce_grid_windowed_spectrum(da: xr.DataArray) -> xr.DataArray:
    """
    Best-effort coercion to GridWindowedSpectrum schema.

    Canonical dims: ("time_win", "AP", "ML", "freq").

    Accepts common input forms:
    - ``spectrogramx()`` output ``("ML", "AP", "freq", "time")`` — renames
      ``time`` → ``time_win`` and transposes.
    - Lowercase ``ml``/``ap`` variants — uppercased automatically.
    - Any permutation of the four canonical dims.
    """
    _check_type(da, xr.DataArray, "GridWindowedSpectrum")

    # Rename lowercase spatial dims to uppercase.
    renames = {}
    if "ml" in da.dims and "ML" not in da.dims:
        renames["ml"] = "ML"
    if "ap" in da.dims and "AP" not in da.dims:
        renames["ap"] = "AP"
    # Rename "time" → "time_win" when it carries window-center semantics.
    if "time" in da.dims and "time_win" not in da.dims:
        renames["time"] = "time_win"
    if renames:
        da = da.rename(renames)

    if set(da.dims) != set(DIMS_GRID_WINDOWED_SPECTRUM):
        raise ValueError(
            f"GridWindowedSpectrum must have dims {DIMS_GRID_WINDOWED_SPECTRUM} "
            f"(permutation allowed), got {tuple(da.dims)}"
        )
    if tuple(da.dims) != DIMS_GRID_WINDOWED_SPECTRUM:
        da = da.transpose(*DIMS_GRID_WINDOWED_SPECTRUM)
    return validate_grid_windowed_spectrum(da)


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


@dataclass
class Intervals:
    """
    Named set of time intervals for iEEG analysis.

    Lightweight alternative to pynapple.IntervalSet and MNE Annotations
    that stays within cogpy's numpy-first convention. Optional conversions
    to external objects available via to_pynapple() and to_mne_annotations().

    Parameters
    ----------
    starts : (n,) array-like — interval start times in seconds
    ends   : (n,) array-like — interval end times in seconds
    name   : str — label for this interval set (e.g. "PerSWS", "spindles")

    Validation
    ----------
    - ends must be strictly greater than starts (element-wise)
    - starts and ends must have the same length
    - all values must be finite

    Notes
    -----
    Intervals are open-ended on the right: [t0, t1).
    Consistent with cogpy.brainstates.intervals convention.
    """

    starts: np.ndarray
    ends: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.starts = np.asarray(self.starts, dtype=float)
        self.ends = np.asarray(self.ends, dtype=float)
        if self.starts.ndim != 1 or self.ends.ndim != 1:
            raise ValueError("Intervals.starts and ends must be 1D arrays")
        if len(self.starts) != len(self.ends):
            raise ValueError(
                f"Intervals.starts and ends must have same length, "
                f"got {len(self.starts)} and {len(self.ends)}"
            )
        if not np.all(np.isfinite(self.starts)) or not np.all(np.isfinite(self.ends)):
            raise ValueError("Intervals.starts and ends must be finite")
        if not np.all(self.ends > self.starts):
            raise ValueError("Intervals: all ends must be strictly greater than starts")

    def __len__(self) -> int:
        return len(self.starts)

    def __repr__(self) -> str:
        if len(self) == 0:
            return f"Intervals(name={self.name!r}, n=0)"
        return (
            f"Intervals(name={self.name!r}, n={len(self)}, "
            f"span=[{self.starts.min():.3f}, {self.ends.max():.3f}]s)"
        )

    def to_array(self) -> np.ndarray:
        """Return (n, 2) array of [[t0, t1], ...]."""
        return np.stack([self.starts, self.ends], axis=1)

    def total_duration(self) -> float:
        """Total duration covered by all intervals in seconds."""
        return float(np.sum(self.ends - self.starts))

    @classmethod
    def from_array(cls, arr: np.ndarray, name: str = "") -> "Intervals":
        """
        Construct from (n, 2) array of [[t0, t1], ...].

        Parameters
        ----------
        arr  : (n, 2) array-like
        name : str label
        """
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Intervals.from_array expects (n, 2) array, got {arr.shape}")
        return cls(arr[:, 0], arr[:, 1], name=name)

    @classmethod
    def from_state_dict(cls, states: dict, state_name: str) -> "Intervals":
        """
        Construct from cogpy brainstates state dict.

        Parameters
        ----------
        states     : dict of {state_name: [[t0, t1], ...]}
        state_name : key to extract
        """
        if state_name not in states:
            raise KeyError(
                f"State {state_name!r} not found in states dict. "
                f"Available: {list(states.keys())}"
            )
        arr = np.array(states[state_name], dtype=float)
        return cls.from_array(arr, name=state_name)

    def to_pynapple(self):
        """
        Convert to pynapple.IntervalSet.

        Requires pynapple to be installed.
        """
        try:
            import pynapple as nap
        except ImportError:
            raise ImportError(
                "pynapple is required for Intervals.to_pynapple(). Install via: pip install pynapple"
            )
        return nap.IntervalSet(start=self.starts, end=self.ends)

    def to_mne_annotations(self, orig_time=None):
        """
        Convert to mne.Annotations.

        Requires mne to be installed.

        Parameters
        ----------
        orig_time : float or None — origin time passed to mne.Annotations
        """
        try:
            import mne
        except ImportError:
            raise ImportError("mne is required for Intervals.to_mne_annotations(). Install via: pip install mne")
        durations = self.ends - self.starts
        return mne.Annotations(
            onset=self.starts,
            duration=durations,
            description=self.name,
            orig_time=orig_time,
        )


@dataclass
class Events:
    """
    Timestamped point events with optional labels.

    Represents discrete events (spindle peaks, ripple troughs,
    stimulus onsets) as a typed container. Replaces ad-hoc
    numpy arrays of timestamps in cogpy pipelines.

    Parameters
    ----------
    times  : (n,) array-like — event times in seconds
    labels : (n,) array-like of str, or empty array — per-event labels
    name   : str — label for this event set (e.g. "spindles", "ripples")

    Notes
    -----
    If labels is empty or not provided, an empty string array is stored.
    Use to_intervals() to expand point events to epoch windows.
    """

    times: np.ndarray
    labels: np.ndarray = None
    name: str = ""

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        if self.times.ndim != 1:
            raise ValueError("Events.times must be 1D")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("Events.times must be finite")
        if self.labels is None or (hasattr(self.labels, "__len__") and len(self.labels) == 0):
            self.labels = np.array([""] * len(self.times), dtype=str)
        else:
            self.labels = np.asarray(self.labels, dtype=str)
            if len(self.labels) != len(self.times):
                raise ValueError(
                    f"Events.labels must have same length as times, "
                    f"got {len(self.labels)} and {len(self.times)}"
                )
        sort_idx = np.argsort(self.times)
        self.times = self.times[sort_idx]
        self.labels = self.labels[sort_idx]

    def __len__(self) -> int:
        return len(self.times)

    def __repr__(self) -> str:
        return (
            f"Events(name={self.name!r}, n={len(self)}, "
            f"span=[{self.times.min():.3f}, {self.times.max():.3f}]s)"
            if len(self) > 0
            else f"Events(name={self.name!r}, n=0)"
        )

    def to_intervals(self, pre: float, post: float) -> Intervals:
        """
        Expand point events to symmetric windows → Intervals.

        Parameters
        ----------
        pre  : float — seconds before each event (positive = before)
        post : float — seconds after each event

        Returns
        -------
        Intervals with starts = times - pre, ends = times + post
        """
        if pre < 0 or post < 0:
            raise ValueError("pre and post must be non-negative")
        return Intervals(
            starts=self.times - pre,
            ends=self.times + post,
            name=self.name,
        )

    def to_pynapple(self):
        """
        Convert to pynapple.Ts (timestamps only, labels dropped).

        Requires pynapple to be installed.
        """
        try:
            import pynapple as nap
        except ImportError:
            raise ImportError("pynapple is required for Events.to_pynapple(). Install via: pip install pynapple")
        return nap.Ts(t=self.times)

    def restrict(self, intervals: "Intervals") -> "Events":
        """
        Return only events that fall within the given intervals.

        Parameters
        ----------
        intervals : Intervals

        Returns
        -------
        Events — subset of events within any interval
        """
        mask = np.zeros(len(self.times), dtype=bool)
        for t0, t1 in zip(intervals.starts, intervals.ends):
            mask |= (self.times >= t0) & (self.times < t1)
        return Events(self.times[mask], self.labels[mask], name=self.name)


@dataclass
class EventCatalog:
    """
    Canonical detector output contract: a per-event table + provenance + converters.

    Required table columns
    ----------------------
    - event_id   : str | int
    - t          : float  (peak/center time, seconds)
    - t0         : float  (start time, seconds)
    - t1         : float  (end time, seconds; must satisfy t1 > t0)
    - duration   : float  (t1 - t0, seconds)
    - label      : str
    - score      : float  (detector-specific units)

    Optional table columns (not enforced)
    -------------------------------------
    channel, AP, ML, f0, f1, f_peak, n_channels, ch_min, ch_max, source
    """

    family: str
    table: pd.DataFrame
    meta: dict
    memberships: pd.DataFrame | None = None

    def to_events(self) -> Events:
        return Events(
            times=self.table["t"].to_numpy(),
            labels=self.table["label"].to_numpy(),
            name=self.family,
        )

    def to_intervals(self) -> Intervals:
        return Intervals(
            starts=self.table["t0"].to_numpy(),
            ends=self.table["t1"].to_numpy(),
            name=self.family,
        )

    def to_event_stream(self, *, style=None):
        from cogpy.events import EventStream

        return EventStream(
            name=self.family,
            df=self.table,
            time_col="t",
            id_col="event_id",
            style=style,
        )

    def to_array(self) -> np.ndarray:
        t0 = np.asarray(self.table["t0"].to_numpy(), dtype=float)
        t1 = np.asarray(self.table["t1"].to_numpy(), dtype=float)
        return np.stack([t0, t1], axis=1)

    @classmethod
    def from_burst_dict(
        cls,
        bursts,
        *,
        meta: dict,
        memberships=None,
    ) -> "EventCatalog":
        """
        Convert `aggregate_bursts` output to the canonical EventCatalog schema.

        Field mapping
        -------------
        burst_id   → event_id
        t0_s       → t0
        t1_s       → t1
        t_peak_s   → t
        f0_hz      → f0
        f1_hz      → f1
        f_peak_hz  → f_peak
        n_channels, ch_min, ch_max → passthrough as optional cols
        score: use an available score/amplitude field when present; otherwise
               default to 1.0 and set meta["params"]["score_source"] = "not_available"
        """

        rows: list[dict] = []
        score_source = "not_available"
        score_key_candidates = (
            "score",
            "amp",
            "amplitude",
            "power",
            "peak_amp",
            "peak_power",
        )

        for b in list(bursts or []):
            if not isinstance(b, dict):
                raise ValueError("from_burst_dict expects bursts as a list of dicts")

            row: dict = {
                "event_id": b.get("burst_id"),
                "t0": b.get("t0_s"),
                "t1": b.get("t1_s"),
                "t": b.get("t_peak_s"),
                "f0": b.get("f0_hz"),
                "f1": b.get("f1_hz"),
                "f_peak": b.get("f_peak_hz"),
                "n_channels": b.get("n_channels"),
                "ch_min": b.get("ch_min"),
                "ch_max": b.get("ch_max"),
                "label": "burst",
            }

            score_val = None
            for k in score_key_candidates:
                if k in b and b.get(k) is not None:
                    score_val = b.get(k)
                    score_source = str(k)
                    break
            if score_val is None:
                score_val = 1.0
            row["score"] = score_val

            try:
                row["duration"] = float(row["t1"]) - float(row["t0"])
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Invalid t0/t1 for burst row: {e}") from e

            rows.append(row)

        df = pd.DataFrame.from_records(rows)

        # Normalize memberships to a DataFrame if provided as list of tuples.
        mem_df = None
        if memberships is not None:
            if isinstance(memberships, pd.DataFrame):
                mem_df = memberships
            else:
                try:
                    mem_df = pd.DataFrame(list(memberships), columns=["event_id", "channel"])
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"Invalid memberships; expected DataFrame or list of (event_id, channel): {e}") from e

        meta_out = dict(meta or {})
        params = meta_out.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("meta['params'] must be a dict")
        if score_source == "not_available":
            params = dict(params)
            params["score_source"] = "not_available"
            meta_out["params"] = params

        # Ensure minimum required meta keys for downstream validation/coercion workflows.
        meta_out.setdefault("n_events", int(len(df)))
        meta_out.setdefault("cogpy_version", "unknown")

        cat = cls(family="burst", table=df, meta=meta_out, memberships=mem_df)
        validate_event_catalog(cat)
        return cat


def validate_event_catalog(catalog: EventCatalog) -> None:
    allowed = {"burst", "ripple", "spindle", "wave", "generic"}

    if str(getattr(catalog, "family", "")) not in allowed:
        raise ValueError(f"EventCatalog.family must be one of {sorted(allowed)}, got {getattr(catalog, 'family', None)!r}")

    table = getattr(catalog, "table", None)
    if not isinstance(table, pd.DataFrame):
        raise ValueError("EventCatalog.table must be a pandas DataFrame")

    required_cols = {"event_id", "t", "t0", "t1", "duration", "label", "score"}
    missing = required_cols - set(table.columns)
    if missing:
        raise ValueError(f"EventCatalog.table missing required columns: {sorted(missing)}")

    # event_id uniqueness (compare as strings).
    ids = table["event_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("EventCatalog.table event_id values must be unique")

    # Numeric columns must be finite.
    for col in ("t0", "t1", "t", "duration", "score"):
        vals = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float, copy=False)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"EventCatalog.table column {col!r} must be finite (no NaN/inf)")

    t0 = pd.to_numeric(table["t0"], errors="coerce").to_numpy(dtype=float, copy=False)
    t1 = pd.to_numeric(table["t1"], errors="coerce").to_numpy(dtype=float, copy=False)
    t = pd.to_numeric(table["t"], errors="coerce").to_numpy(dtype=float, copy=False)
    duration = pd.to_numeric(table["duration"], errors="coerce").to_numpy(dtype=float, copy=False)

    if not np.all(t1 > t0):
        raise ValueError("EventCatalog.table must satisfy t1 > t0 for all events")

    if not np.all((t >= t0) & (t <= t1)):
        raise ValueError("EventCatalog.table must satisfy t0 <= t <= t1 for all events")

    if not np.all(np.abs(duration - (t1 - t0)) < 1e-6):
        raise ValueError("EventCatalog.table must satisfy abs(duration - (t1 - t0)) < 1e-6 for all events")

    meta = getattr(catalog, "meta", None)
    if not isinstance(meta, dict):
        raise ValueError("EventCatalog.meta must be a dict")

    required_meta = {"detector", "params", "fs", "n_events", "cogpy_version"}
    missing_meta = required_meta - set(meta.keys())
    if missing_meta:
        raise ValueError(f"EventCatalog.meta missing required keys: {sorted(missing_meta)}")

    if not isinstance(meta.get("params"), dict):
        raise ValueError("EventCatalog.meta['params'] must be a dict")

    try:
        n_events = int(meta["n_events"])
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"EventCatalog.meta['n_events'] must be an int: {e}") from e
    if n_events != len(table):
        raise ValueError("EventCatalog.meta['n_events'] must match len(EventCatalog.table)")

    memberships = getattr(catalog, "memberships", None)
    if memberships is not None:
        if not isinstance(memberships, pd.DataFrame):
            raise ValueError("EventCatalog.memberships must be a pandas DataFrame when provided")
        need = {"event_id", "channel"}
        missing_m = need - set(memberships.columns)
        if missing_m:
            raise ValueError(f"EventCatalog.memberships missing required columns: {sorted(missing_m)}")
        mem_ids = memberships["event_id"].astype(str).unique()
        table_id_set = set(ids.to_numpy())
        bad = [eid for eid in mem_ids if eid not in table_id_set]
        if bad:
            raise ValueError("EventCatalog.memberships contains event_id values not present in table['event_id']")


def coerce_event_catalog(
    obj,
    *,
    family: str | None = None,
    meta: dict | None = None,
    memberships=None,
) -> EventCatalog:
    if isinstance(obj, EventCatalog):
        validate_event_catalog(obj)
        return obj

    if isinstance(obj, pd.DataFrame):
        if family is None:
            raise ValueError("family is required when coercing from a DataFrame")
        if meta is None or not isinstance(meta, dict):
            raise ValueError("meta (dict) is required when coercing from a DataFrame")

        df = obj.copy()
        if "duration" not in df.columns:
            df["duration"] = pd.to_numeric(df["t1"], errors="coerce") - pd.to_numeric(df["t0"], errors="coerce")
        if "label" not in df.columns:
            df["label"] = "event"

        meta_out = dict(meta)
        meta_out["n_events"] = int(len(df))
        meta_out.setdefault("cogpy_version", "unknown")

        cat = EventCatalog(family=str(family), table=df, meta=meta_out, memberships=memberships)
        validate_event_catalog(cat)
        return cat

    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        if meta is None or not isinstance(meta, dict):
            raise ValueError("meta (dict) is required when coercing from a list[dict]")
        return EventCatalog.from_burst_dict(obj, meta=meta, memberships=memberships)

    raise ValueError(f"Unsupported input type for coerce_event_catalog: {type(obj).__name__}")


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
