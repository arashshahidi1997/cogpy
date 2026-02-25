from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

__all__ = ["top_n_variance", "top_n_correlation"]


def _time_slice_indices(time_vals: np.ndarray, t0: float | None, t1: float | None) -> tuple[int, int]:
    t = np.asarray(time_vals, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("time_vals must be 1D")
    lo = float(t[0]) if t0 is None else float(t0)
    hi = float(t[-1]) if t1 is None else float(t1)
    if hi < lo:
        lo, hi = hi, lo
    i0 = int(np.searchsorted(t, lo, side="left"))
    i1 = int(np.searchsorted(t, hi, side="right"))
    i0 = max(i0, 0)
    i1 = min(i1, len(t))
    if i1 <= i0:
        i0, i1 = 0, len(t)
    return i0, i1


def _as_tc(sig_tc: xr.DataArray) -> xr.DataArray:
    if not isinstance(sig_tc, xr.DataArray):
        raise TypeError("sig_tc must be an xarray.DataArray")
    if "time" not in sig_tc.dims or "channel" not in sig_tc.dims:
        raise ValueError(f"sig_tc must have dims ('time','channel'); got dims={tuple(sig_tc.dims)}")
    return sig_tc.transpose("time", "channel")


def top_n_variance(
    sig_tc: xr.DataArray,
    *,
    n: int,
    t0: float | None = None,
    t1: float | None = None,
) -> list[int]:
    """
    Return channel indices (positional) for the top-N variance channels
    within the requested time window.
    """
    sig_tc = _as_tc(sig_tc)
    n_ch = int(sig_tc.sizes["channel"])
    n = int(n)
    if n <= 0:
        return []
    n = min(n, n_ch)

    i0, i1 = _time_slice_indices(sig_tc["time"].values, t0, t1)
    arr = np.asarray(sig_tc.isel(time=slice(i0, i1)).values)
    # arr: (time, channel)
    v = np.var(arr, axis=0)
    order = np.argsort(v)[::-1]
    return [int(i) for i in order[:n]]


def top_n_correlation(
    sig_tc: xr.DataArray,
    *,
    seed_channel: int,
    n: int,
    t0: float | None = None,
    t1: float | None = None,
    method: Literal["pearson"] = "pearson",
) -> list[int]:
    """
    Return channel indices (positional) for the top-N channels most correlated
    with a seed channel in the requested time window.
    """
    if method != "pearson":
        raise ValueError("only method='pearson' is supported")

    sig_tc = _as_tc(sig_tc)
    n_ch = int(sig_tc.sizes["channel"])
    seed_channel = int(seed_channel)
    if not (0 <= seed_channel < n_ch):
        raise ValueError(f"seed_channel out of range: {seed_channel} (n_ch={n_ch})")

    n = int(n)
    if n <= 0:
        return []
    n = min(n, n_ch)

    i0, i1 = _time_slice_indices(sig_tc["time"].values, t0, t1)
    arr = np.asarray(sig_tc.isel(time=slice(i0, i1)).values)  # (time, channel)
    x = arr[:, seed_channel]
    x = x - x.mean()
    x_sd = np.sqrt(np.mean(x * x)) + 1e-12

    y = arr - arr.mean(axis=0, keepdims=True)
    y_sd = np.sqrt(np.mean(y * y, axis=0)) + 1e-12

    corr = (x[:, None] * y).mean(axis=0) / (x_sd * y_sd)
    order = np.argsort(corr)[::-1]
    return [int(i) for i in order[:n]]

