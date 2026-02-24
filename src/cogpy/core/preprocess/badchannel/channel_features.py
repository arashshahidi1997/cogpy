"""Raw temporal channel features (no spatial context).

All functions accept arrays shaped like `(..., time)` and reduce over the time
axis (default: last axis).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as nd
from scipy import signal
import xarray as xr

from cogpy.core.utils.sliding_core import running_blockwise_xr, running_reduce_xr

EPS = 1e-12


def relative_variance(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.nanvar(arr, axis=axis)


def deviation(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.nanmean(arr, axis=axis)


def standard_deviation(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.nanstd(arr, axis=axis)


def amplitude(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)


def time_derivative(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    grad = np.gradient(arr, axis=axis)
    return np.nanmean(np.abs(grad), axis=axis)


def hurst_exponent(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    y = arr - np.nanmean(arr, axis=axis, keepdims=True)
    z = np.cumsum(y, axis=axis)
    r = np.nanmax(z, axis=axis) - np.nanmin(z, axis=axis)
    std = np.nanstd(arr, axis=axis)
    return 0.5 * np.log((r / (std + EPS)) + EPS)


def kurtosis(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    centered = x - mean
    m2 = np.nanmean(centered**2, axis=axis)
    m4 = np.nanmean(centered**4, axis=axis)
    return m4 / ((m2**2) + EPS)


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
    "relative_variance",
    "deviation",
    "amplitude",
    "time_derivative",
    "hurst_exponent",
    "kurtosis",
)


def _feature_func(name: str):
    if name == "relative_variance":
        return relative_variance
    if name == "deviation":
        return deviation
    if name == "standard_deviation":
        return standard_deviation
    if name == "amplitude":
        return amplitude
    if name == "time_derivative":
        return time_derivative
    if name == "hurst_exponent":
        return hurst_exponent
    if name == "kurtosis":
        return kurtosis
    if name == "temporal_mean_laplacian":
        return temporal_mean_laplacian
    raise KeyError(f"Unknown feature: {name}")


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
        Input signal. Common ECoG schemas are supported, e.g. dims:
        - ``(time, ch)``
        - ``(time, AP, ML)``
        The feature functions reduce along ``time_dim``.
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
    """
    if not isinstance(xsig, xr.DataArray):
        raise TypeError("extract_channel_features_xr expects an xarray.DataArray")
    if time_dim not in xsig.dims:
        raise ValueError(f"time_dim={time_dim!r} not in xsig.dims={tuple(xsig.dims)}")

    feats = list(DEFAULT_FEATURES if features is None else features)
    if window_size is None and window_step is not None:
        raise ValueError("Provide both window_size and window_step, or neither.")
    if window_size is not None and window_step is None:
        raise ValueError("Provide both window_size and window_step, or neither.")

    ds_vars: dict[str, xr.DataArray] = {}
    for name in feats:
        func = _feature_func(name)

        if window_size is None:
            if name == "temporal_mean_laplacian":
                if "AP" not in xsig.dims or "ML" not in xsig.dims:
                    raise ValueError("temporal_mean_laplacian requires dims ('AP','ML',time).")
                da = xr.apply_ufunc(
                    temporal_mean_laplacian,
                    xsig,
                    input_core_dims=[["AP", "ML", time_dim]],
                    output_core_dims=[["AP", "ML"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[np.float64],
                )
            else:
                axis = xsig.get_axis_num(time_dim)

                def _wrapped(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
                    return func(x, axis=axis)

                da = xr.apply_ufunc(
                    _wrapped,
                    xsig,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[]],
                    vectorize=True,
                    kwargs={"axis": axis},
                    dask="parallelized",
                    output_dtypes=[np.float64],
                )
            da.name = name
            ds_vars[name] = da
            continue

        # windowed features
        if name == "temporal_mean_laplacian":
            da = running_blockwise_xr(
                xsig,
                int(window_size),
                int(window_step),
                temporal_mean_laplacian,
                run_dim=time_dim,
                out_dim=out_dim,
                center_method=center_method,
                progress=bool(progress),
            )
        else:
            da = running_reduce_xr(
                xsig,
                int(window_size),
                int(window_step),
                lambda a, axis=-1: func(a, axis=axis),
                run_dim=time_dim,
                out_dim=out_dim,
                center_method=center_method,
            )
        da.name = name
        ds_vars[name] = da

    ds = xr.Dataset(ds_vars)
    ds.attrs.update(dict(xsig.attrs))
    ds.attrs.update(
        {
            "time_dim": str(time_dim),
            "features": list(feats),
        }
    )
    if window_size is not None:
        ds.attrs.update(
            {
                "window_size": int(window_size),
                "window_step": int(window_step),
                "out_dim": str(out_dim),
                "center_method": str(center_method),
            }
        )
    return ds
