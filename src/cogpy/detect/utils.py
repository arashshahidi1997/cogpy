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
    "score_to_bouts",
    "bout_occupancy",
    "bout_duration_summary",
]


def bandpass_filter(
    data: xr.DataArray,
    low: float,
    high: float,
    *,
    order: int = 4,
    time_dim: str = "time",
) -> xr.DataArray:
    """Bandpass filter using `cogpy.preprocess.filtx.bandpassx`."""
    from cogpy.preprocess.filtx import bandpassx

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


def score_to_bouts(
    score: np.ndarray,
    times: np.ndarray,
    *,
    low: float,
    high: float,
    min_duration: float = 0.0,
    merge_gap: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Convert a 1D score time series to event bouts via dual threshold.

    Composes :func:`dual_threshold_events_1d` with gap merging and
    minimum-duration filtering.  Useful for converting any continuous
    noise/artifact score into discrete event intervals.

    Parameters
    ----------
    score : (N,) — 1D score array
    times : (N,) — corresponding time stamps (seconds)
    low : float — lower threshold (event boundary)
    high : float — upper threshold (event must contain at least one
        sample above this value)
    min_duration : float — discard events shorter than this (seconds,
        default 0.0)
    merge_gap : float — merge events separated by less than this
        (seconds, default 0.0)

    Returns
    -------
    events : list[dict]
        Each dict has keys ``t0``, ``t1``, ``t``, ``value``, ``duration``.
    """
    score = np.asarray(score, dtype=float)
    times = np.asarray(times, dtype=float)

    events = dual_threshold_events_1d(score, times, low=low, high=high)

    if merge_gap > 0 and len(events) > 1:
        # Convert time-domain events to sample-index intervals for merging.
        # find_true_runs / merge_intervals work on integer sample indices.
        dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        gap_samples = max(0, int(round(float(merge_gap) / dt)))

        above_low = score >= float(low)
        above_high = score >= float(high)

        # Re-detect with merged intervals
        runs = find_true_runs(above_low)
        merged = merge_intervals(runs, gap=gap_samples)

        events = []
        for i0, i1 in merged:
            seg = score[i0 : i1 + 1]
            if not np.any(above_high[i0 : i1 + 1]):
                continue
            k = int(np.argmax(seg))
            ip = i0 + k
            events.append(
                {
                    "t0": float(times[i0]),
                    "t": float(times[ip]),
                    "t1": float(times[i1]),
                    "value": float(score[ip]),
                    "duration": float(times[i1] - times[i0]),
                }
            )

    if min_duration > 0:
        events = [e for e in events if e["duration"] >= float(min_duration)]

    return events


def bout_occupancy(bouts: list[dict[str, Any]], total_duration: float) -> float:
    """
    Fraction of *total_duration* occupied by bouts.

    Parameters
    ----------
    bouts : list[dict]
        Each dict must have a ``"duration"`` key (as returned by
        :func:`score_to_bouts`).
    total_duration : float
        Total recording or epoch duration (seconds).

    Returns
    -------
    float
        Occupancy in ``[0, 1]``.  Returns ``0.0`` when *bouts* is empty.
    """
    if total_duration <= 0:
        raise ValueError(f"total_duration must be > 0, got {total_duration}")
    if not bouts:
        return 0.0
    return float(min(sum(b["duration"] for b in bouts) / total_duration, 1.0))


def bout_duration_summary(bouts: list[dict[str, Any]]) -> dict[str, float]:
    """
    Summary statistics of bout durations.

    Parameters
    ----------
    bouts : list[dict]
        Each dict must have a ``"duration"`` key.

    Returns
    -------
    dict
        Keys: ``count``, ``mean``, ``median``, ``std``, ``p5``, ``p95``.
        All values are ``0.0`` when *bouts* is empty (``count`` is ``0``).
    """
    if not bouts:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p5": 0.0, "p95": 0.0}
    durations = np.array([b["duration"] for b in bouts], dtype=float)
    return {
        "count": len(durations),
        "mean": float(np.mean(durations)),
        "median": float(np.median(durations)),
        "std": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
        "p5": float(np.percentile(durations, 5)),
        "p95": float(np.percentile(durations, 95)),
    }

