"""PSD utilities for TensorScope (v2.8.0)."""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["compute_psd_window", "psd_to_db", "stack_spatial_dims"]


def compute_psd_window(
    signal: xr.DataArray,
    *,
    t_center: float,
    window_size: float,
    nperseg: int = 256,
    method: str = "welch",
    axis: str = "time",
    bandwidth: float = 4.0,
    noverlap: int | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> xr.DataArray:
    """
    Compute PSD in a time window around `t_center`.

    Parameters
    ----------
    signal
        Input signal with a `time` dimension and `fs` in attrs/coord.
    t_center
        Center time in seconds.
    window_size
        Total window size in seconds.
    nperseg, noverlap, method, bandwidth, axis, fmin, fmax
        Passed to `cogpy.spectral.specx.psdx`.
    """
    from cogpy.spectral.specx import psdx

    t_center = float(t_center)
    half = float(window_size) / 2.0
    t0 = t_center - half
    t1 = t_center + half

    win = signal.sel({str(axis): slice(t0, t1)})

    return psdx(
        win,
        axis=str(axis),
        method=str(method),  # type: ignore[arg-type]
        bandwidth=float(bandwidth),
        nperseg=int(nperseg),
        noverlap=None if noverlap is None else int(noverlap),
        fmin=float(fmin),
        fmax=None if fmax is None else float(fmax),
    )


def psd_to_db(psd: xr.DataArray, *, ref: float = 1.0) -> xr.DataArray:
    """Convert PSD to dB: 10 * log10(psd / ref)."""
    ref = float(ref)
    vals = np.asarray(psd.data, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(vals / ref)
    da = xr.DataArray(out, dims=psd.dims, coords=psd.coords, attrs=dict(psd.attrs), name=psd.name)
    da.attrs["units"] = "dB"
    return da


def stack_spatial_dims(data: xr.DataArray) -> xr.DataArray:
    """
    Stack (AP, ML) into `channel` if present. Otherwise returns input unchanged.

    For (AP, ML) grids, the resulting `channel` coordinate is a MultiIndex.
    """
    if ("AP" in data.dims) and ("ML" in data.dims):
        return data.stack(channel=("AP", "ML"))
    if "channel" in data.dims:
        return data
    return data.expand_dims(channel=[0])

