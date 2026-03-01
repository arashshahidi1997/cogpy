"""Raw temporal channel features (no spatial context).

All functions accept arrays shaped like `(..., time)` and reduce over the time
axis (default: last axis).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import warnings
import numpy as np
import scipy.ndimage as nd
from scipy import signal
import xarray as xr

from cogpy.datasets import schemas as sch
from cogpy.core.utils.sliding_core import running_blockwise_xr, running_reduce_xr

EPS = 1e-12


# Temporal measures moved to cogpy.core.measures.temporal.
# Re-exported here for backward compatibility.
from cogpy.core.measures.temporal import (
    relative_variance,
    deviation,
    standard_deviation,
    amplitude,
    time_derivative,
    hurst_exponent,
    kurtosis,
)


def noise_to_signal(
    arr: np.ndarray,
    fs: float,
    *,
    axis: int = -1,
    low_freq: tuple[float, float] = (0.1, 30.0),
    high_freq: tuple[float, float] = (30.0, 80.0),
    nperseg: int = 256,
) -> np.ndarray:
    """High-band to low-band power ratio using Welch PSD.

    Parameters
    ----------
    arr
        Array shaped ``(..., time)``.
    fs
        Sampling rate in Hz.
    axis
        Time axis.
    low_freq, high_freq
        Frequency bands in Hz used for denominator and numerator.
    nperseg
        Welch segment length.
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.size == 0:
        raise ValueError("noise_to_signal received an empty array.")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")

    freq, psd = signal.welch(x, fs=float(fs), axis=axis, nperseg=int(nperseg))
    low_mask = (freq >= float(low_freq[0])) & (freq <= float(low_freq[1]))
    high_mask = (freq >= float(high_freq[0])) & (freq <= float(high_freq[1]))
    if not np.any(low_mask):
        raise ValueError(f"low_freq band {low_freq} is outside Welch frequency grid.")
    if not np.any(high_mask):
        raise ValueError(f"high_freq band {high_freq} is outside Welch frequency grid.")

    axis_psd = axis if axis >= 0 else (psd.ndim + axis)
    low = np.nanmean(np.take(psd, np.where(low_mask)[0], axis=axis_psd), axis=axis_psd)
    high = np.nanmean(np.take(psd, np.where(high_mask)[0], axis=axis_psd), axis=axis_psd)
    return high / (low + EPS)


def snr(
    arr: np.ndarray,
    fs: float,
    *,
    axis: int = -1,
    low_freq: tuple[float, float] = (0.1, 30.0),
    high_freq: tuple[float, float] = (30.0, 80.0),
    nperseg: int = 256,
) -> np.ndarray:
    """Alias for `noise_to_signal`."""
    return noise_to_signal(
        arr,
        fs,
        axis=axis,
        low_freq=low_freq,
        high_freq=high_freq,
        nperseg=nperseg,
    )


def temporal_mean_laplacian(arr: np.ndarray) -> np.ndarray:
    """Legacy feature used by the current preprocess pipeline.

    Matches `cogpy.core.preprocess.channel_feature_functions.temporal_mean_laplacian`.
    Expects `arr` shaped (AP, ML, time) and returns (AP, ML).
    """
    laplacian_stencil = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype=np.float64)
    a_laplacian = nd.convolve(
        np.asarray(arr, dtype=np.float64),
        np.expand_dims(laplacian_stencil, axis=-1),
        mode="mirror",
    )
    return np.mean(np.abs(a_laplacian), axis=-1)


DEFAULT_FEATURES: tuple[str, ...] = (
    "variance",
    "mean",
    "amplitude",
    "time_derivative",
    "hurst_exponent",
    "kurtosis",
    'snr'
)

SPECTRAL_FEATURES: tuple[str, ...] = (
    "noise_to_signal",
    "snr",
)

_DEPRECATED_FEATURE_NAMES: dict[str, str] = {
    "relative_variance": "variance",
    "deviation": "mean",
}

_feature_func = {
    # canonical names
    "variance": relative_variance,
    "mean": deviation,
    'relative_variance': relative_variance,
    'deviation': deviation,
    'standard_deviation': standard_deviation,
    'amplitude': amplitude,
    'time_derivative': time_derivative,
    'hurst_exponent': hurst_exponent,
    'kurtosis': kurtosis,
    'temporal_mean_laplacian': temporal_mean_laplacian,
    'noise_to_signal': noise_to_signal,
    "snr": snr,
}


def _coerce_xsig_to_schema(xsig: xr.DataArray, *, time_dim: str) -> tuple[xr.DataArray, str]:
    """
    Coerce `xsig` into a canonical schemas.py representation.

    Returns
    -------
    xsig_coerced, kind
        kind is one of {"ieeg_grid","multichannel","ieeg_time_channel"}.
    """
    if not isinstance(xsig, xr.DataArray):
        raise TypeError("xsig must be an xarray.DataArray")
    if time_dim not in xsig.dims:
        raise ValueError(f"time_dim={time_dim!r} not in xsig.dims={tuple(xsig.dims)}")

    xs = xsig

    # Normalize dim naming toward schemas.
    if "channel" not in xs.dims and "ch" in xs.dims:
        xs = xs.rename({"ch": "channel"})

    # Enforce schema time dim name.
    if time_dim != "time":
        if "time" in xs.dims:
            raise ValueError(
                f"Cannot rename time_dim={time_dim!r} -> 'time' because xsig already has a 'time' dim "
                f"(dims={tuple(xs.dims)})"
            )
        xs = xs.rename({time_dim: "time"})

    if "AP" in xs.dims and "ML" in xs.dims:
        return sch.coerce_ieeg_grid(xs), "ieeg_grid"

    if set(xs.dims) == set(sch.DIMS_MULTICHANNEL):
        return sch.coerce_multichannel(xs), "multichannel"

    if set(xs.dims) == set(sch.DIMS_IEEG_TIME_CHANNEL):
        return sch.coerce_ieeg_time_channel(xs), "ieeg_time_channel"

    raise ValueError(
        "xsig does not match a supported schema.\n"
        f"  Got dims: {tuple(xsig.dims)}\n"
        "  Expected one of:\n"
        "    - IEEGGridTimeSeries: ('time','ML','AP') (permutation allowed)\n"
        "    - MultichannelTimeSeries: ('channel','time') (permutation allowed; also accepts ``'ch'`` → ``'channel'``)\n"
        "    - IEEGTimeChannel: ('time','channel') (permutation allowed)\n"
        "  Hint: rename dims and/or use cogpy.datasets.schemas.coerce_* helpers."
    )


def _require_fs(xsig: xr.DataArray) -> float:
    if "fs" not in xsig.attrs:
        raise ValueError("Sampling rate required: set xsig.attrs['fs'] (Hz) for noise_to_signal.")
    fs = float(xsig.attrs["fs"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"xsig.attrs['fs'] must be positive and finite, got {xsig.attrs['fs']!r}.")
    return fs


def _apply_reduce_func(a: np.ndarray, *, axis: int, func: Callable) -> np.ndarray:
    return func(a, axis=axis)


def extract_channel_features_xr(
    xsig: xr.DataArray,
    *,
    features: list[str] | tuple[str, ...] | None = None,
    time_dim: str = "time",
    window_size: int | None = None,
    window_step: int | None = None,
    out_dim: str = "time_win",
    center_method: str = "midpoint",
    progress: bool = True,
) -> xr.Dataset:
    """Extract per-channel temporal features into an ``xr.Dataset``.

    Parameters
    ----------
    xsig
        Input signal. Coerced to a canonical schema before feature extraction:
        - Grid: ``('time','ML','AP')`` (permutation allowed)
        - Multichannel: ``('channel','time')`` (permutation allowed; also accepts ``'ch'`` → ``'channel'``)
        - Time×channel: ``('time','channel')`` (permutation allowed)
        The feature functions reduce along ``time_dim`` (renamed to ``'time'`` in the output).
    features
        Names of features to compute. Defaults to ``DEFAULT_FEATURES``.
        Include ``"temporal_mean_laplacian"`` only for grid signals that include
        ``AP`` and ``ML`` dims.
    window_size, window_step
        If both are provided, computes *windowed* features over sliding windows,
        returning variables with ``out_dim`` instead of ``time_dim``.
        If omitted, computes one feature map over the full recording.
    out_dim
        Name of the window/time output dimension when windowing.
    center_method
        Passed to sliding_core window center helpers, default ``"midpoint"``.
    progress
        Progress bar toggle (only affects blockwise features when windowing).

    Notes
    -----
    The windowed path (``window_size`` / ``window_step``) calls ``xsig.data`` via
    ``sliding_core``. For Dask-backed inputs, this may trigger eager
    materialization of the full array.
    """
    xsig, kind = _coerce_xsig_to_schema(xsig, time_dim=time_dim)
    time_dim = "time"
    is_grid = kind == "ieeg_grid"

    feats = list(DEFAULT_FEATURES if features is None else features)
    if window_size is None and window_step is not None:
        raise ValueError("Provide both window_size and window_step, or neither.")
    if window_size is not None and window_step is None:
        raise ValueError("Provide both window_size and window_step, or neither.")
    for name in feats:
        if name in _DEPRECATED_FEATURE_NAMES:
            warnings.warn(
                f"Feature name {name!r} is deprecated and will be removed in a future version. "
                f"Use {_DEPRECATED_FEATURE_NAMES[name]!r} instead.",
                category=FutureWarning,
                stacklevel=3,
            )
        if name not in _feature_func:
            raise ValueError(f"Unknown feature {name!r}. Known: {sorted(_feature_func.keys())}")

    # Fail fast for attributes needed by requested features.
    fs = _require_fs(xsig) if any(f in SPECTRAL_FEATURES for f in feats) else None

    def _tml_ml_ap(arr_ml_ap_t: np.ndarray) -> np.ndarray:
        arr_ap_ml_t = np.transpose(arr_ml_ap_t, (1, 0, 2))
        out_ap_ml = temporal_mean_laplacian(arr_ap_ml_t)
        return np.transpose(out_ap_ml, (1, 0))

    ds_vars: dict[str, xr.DataArray] = {}
    win_dim = "time_win"
    requested_out_dim = str(out_dim)
    if window_size is not None and requested_out_dim != win_dim:
        warnings.warn(
            f"extract_channel_features_xr: out_dim={requested_out_dim!r} is deprecated; "
            f"windowed outputs always use out_dim={win_dim!r}.",
            category=FutureWarning,
            stacklevel=2,
        )
    for name in feats:
        func = _feature_func[name]

        if window_size is None:
            if name == "temporal_mean_laplacian":
                if not is_grid:
                    raise ValueError("temporal_mean_laplacian requires an IEEGGridTimeSeries input.")
                da = xr.apply_ufunc(
                    _tml_ml_ap,
                    xsig,
                    input_core_dims=[["ML", "AP", time_dim]],
                    output_core_dims=[["ML", "AP"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[np.float64],
                )
            elif name in SPECTRAL_FEATURES:
                assert fs is not None

                spectral_func = partial(func, fs=fs)

                da = xr.apply_ufunc(
                    spectral_func,
                    xsig,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[]],
                    vectorize=True,
                    kwargs={"axis": -1},
                    dask="parallelized",
                    output_dtypes=[np.float64],
                )
            else:
                _wrapped = partial(_apply_reduce_func, func=func)

                da = xr.apply_ufunc(
                    _wrapped,
                    xsig,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[]],
                    vectorize=True,
                    kwargs={"axis": -1},
                    dask="parallelized",
                    output_dtypes=[np.float64],
                )
            da.name = name
            ds_vars[name] = da
            continue

        # windowed features
        if name == "temporal_mean_laplacian":
            if not is_grid:
                raise ValueError("temporal_mean_laplacian requires an IEEGGridTimeSeries input.")
            da = running_blockwise_xr(
                xsig,
                int(window_size),
                int(window_step),
                _tml_ml_ap,
                run_dim=time_dim,
                out_dim=win_dim,
                center_method=center_method,
                progress=bool(progress),
            )
        elif name in SPECTRAL_FEATURES:
            assert fs is not None
            spectral_func = partial(func, fs=fs)
            da = running_blockwise_xr(
                xsig,
                int(window_size),
                int(window_step),
                spectral_func,
                run_dim=time_dim,
                out_dim=win_dim,
                center_method=center_method,
                progress=bool(progress),
            )
        else:
            reducer = partial(_apply_reduce_func, func=func)
            da = running_reduce_xr(
                xsig,
                int(window_size),
                int(window_step),
                reducer,
                run_dim=time_dim,
                out_dim=win_dim,
                center_method=center_method,
            )
        if is_grid:
            da = sch.coerce_ieeg_grid_windowed(da, win_dim=win_dim)
        else:
            da = sch.coerce_multichannel_windowed(da, win_dim=win_dim)
        da.name = name
        ds_vars[name] = da

    ds = xr.Dataset(ds_vars)
    ds.attrs.update(dict(xsig.attrs))
    ds.attrs.update(
        {
            "time_dim": str(time_dim),
            "features": list(feats),
            "schema_kind": str(kind),
        }
    )
    if window_size is not None:
        ds.attrs.update(
            {
                "window_size": int(window_size),
                "window_step": int(window_step),
                "out_dim": win_dim,
                "center_method": str(center_method),
            }
        )
        # Convenience window-rate metadata for downstream smoothing/aggregation.
        if "fs" in xsig.attrs and "fs_win" not in ds.attrs:
            try:
                fs_all = float(xsig.attrs["fs"])
                if np.isfinite(fs_all) and fs_all > 0:
                    ds.attrs.setdefault("window_size_s", float(window_size) / fs_all)
                    ds.attrs.setdefault("window_step_s", float(window_step) / fs_all)
                    ds.attrs.setdefault("fs_win", fs_all / float(window_step))
            except (TypeError, ValueError):
                pass
        if win_dim in ds.coords:
            ds.coords[win_dim].attrs.setdefault("long_name", "window center time")
    return ds
