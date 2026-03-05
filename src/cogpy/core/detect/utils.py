"""
Detection utilities (v2.6.4).

Note: keep heavy deps (SciPy) as local imports so this module can be imported
under minimal interpreters. The supported runtime is the `cogpy` conda env.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "bandpass_filter",
    "hilbert_envelope",
    "zscore_1d",
    "find_true_runs",
    "merge_intervals",
    "dual_threshold_events_1d",
]


def bandpass_filter(
    data: xr.DataArray,
    low: float,
    high: float,
    *,
    order: int = 4,
    time_dim: str = "time",
) -> xr.DataArray:
    """Bandpass filter using `cogpy.core.preprocess.filtx.bandpassx`."""
    from cogpy.core.preprocess.filtx import bandpassx

    return bandpassx(data, float(low), float(high), int(order), axis=str(time_dim))


def hilbert_envelope(data: xr.DataArray, *, time_dim: str = "time") -> xr.DataArray:
    """Compute Hilbert envelope (magnitude of analytic signal)."""
    from scipy.signal import hilbert

    if time_dim not in data.dims:
        raise ValueError(f"Expected time_dim={time_dim!r} in data.dims={tuple(data.dims)}")

    # Ensure numpy-backed array for SciPy.
    values = data.data
    try:
        values = values.compute()
    except Exception:  # noqa: BLE001
        pass
    arr = np.asarray(values, dtype=float)

    axis = int(data.get_axis_num(time_dim))
    analytic = hilbert(arr, axis=axis)
    env = np.abs(analytic)
    return xr.DataArray(env, dims=data.dims, coords=data.coords, attrs=dict(data.attrs), name=data.name)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    """Z-score a 1D array with safe std handling."""
    xv = np.asarray(x, dtype=float)
    mu = float(np.nanmean(xv)) if xv.size else 0.0
    sd = float(np.nanstd(xv)) if xv.size else 1.0
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (xv - mu) / sd


def find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous True runs in a 1D boolean mask.

    Returns list of (i0, i1) inclusive indices.
    """
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []

    # Find edges with padding.
    x = np.concatenate([[False], m, [False]])
    d = np.diff(x.astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def merge_intervals(intervals: list[tuple[int, int]], *, gap: int) -> list[tuple[int, int]]:
    """Merge 1D intervals whose gap is <= `gap` samples."""
    if not intervals:
        return []
    ints = sorted((int(a), int(b)) for a, b in intervals)
    out: list[tuple[int, int]] = []
    cur0, cur1 = ints[0]
    for a, b in ints[1:]:
        if a <= cur1 + int(gap) + 1:
            cur1 = max(cur1, b)
        else:
            out.append((cur0, cur1))
            cur0, cur1 = a, b
    out.append((cur0, cur1))
    return out


def dual_threshold_events_1d(
    x: np.ndarray,
    t: np.ndarray,
    *,
    low: float,
    high: float,
    direction: str = "positive",
) -> list[dict[str, Any]]:
    """
    Dual-threshold detection on a 1D array.

    Events are regions crossing `low` and containing at least one sample above `high`.
    """
    xv = np.asarray(x, dtype=float)
    tv = np.asarray(t, dtype=float)
    if xv.ndim != 1 or tv.ndim != 1 or xv.size != tv.size:
        raise ValueError("x and t must be 1D arrays of same length")

    direction = str(direction)
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'")

    if direction == "negative":
        xv = -xv
        low = float(low)
        high = float(high)

    above_low = xv >= float(low)
    above_high = xv >= float(high)

    events: list[dict[str, Any]] = []
    for i0, i1 in find_true_runs(above_low):
        if not np.any(above_high[i0 : i1 + 1]):
            continue
        seg = xv[i0 : i1 + 1]
        k = int(np.argmax(seg))
        ip = i0 + k
        events.append(
            {
                "t0": float(tv[i0]),
                "t": float(tv[ip]),
                "t1": float(tv[i1]),
                "value": float(xv[ip]),
                "duration": float(tv[i1] - tv[i0]),
            }
        )
    return events

