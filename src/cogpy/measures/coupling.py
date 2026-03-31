"""
Event-train and spectral coupling measures.

STATUS: ACTIVE
Reason: Cross-correlogram, PETH, and spectral power cross-correlation for
        pixecog coupling analysis (H1, H2, H4, H5, H6).
Superseded by: n/a
Safe to remove: no

All functions are pure — array/xarray in, array/xarray out. No file I/O.
EventCatalog timestamps or raw numpy arrays are accepted interchangeably.

References
----------
Sirota et al. (2003), Wierzynski et al. (2009), Peyrache et al. (2011),
Siapas & Wilson (1998).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cogpy.events.catalog import EventCatalog

__all__ = [
    "cross_correlogram",
    "peri_event_histogram",
    "spectral_power_xcorr",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_timestamps(events) -> np.ndarray:
    """Accept EventCatalog, DataFrame, or ndarray; return 1-D float array of times."""
    # EventCatalog
    if hasattr(events, "df"):
        return np.asarray(events.df["t"], dtype=float)
    # pandas DataFrame
    if hasattr(events, "iloc"):
        if "t" in events.columns:
            return np.asarray(events["t"], dtype=float)
        raise ValueError("DataFrame must have a 't' column")
    return np.asarray(events, dtype=float).ravel()


def _circular_shift(times: np.ndarray, rng: np.random.Generator, t_min: float, t_max: float) -> np.ndarray:
    """Circularly shift timestamps by a random offset within [t_min, t_max]."""
    span = t_max - t_min
    shift = rng.uniform(0.1 * span, 0.9 * span)
    shifted = (times - t_min + shift) % span + t_min
    return shifted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cross_correlogram(
    events_a,
    events_b,
    *,
    bin_size: float,
    window: float,
    normalize: bool = True,
    n_surrogates: int = 0,
    seed: int | None = None,
) -> dict:
    """
    Cross-correlogram of two event trains.

    For each event in ``events_a``, counts events in ``events_b`` within
    ``±window`` seconds, binned at ``bin_size`` resolution.

    Parameters
    ----------
    events_a : array-like or EventCatalog
        Reference event timestamps (seconds).
    events_b : array-like or EventCatalog
        Target event timestamps (seconds).
    bin_size : float
        Bin width in seconds (e.g. 0.005 for 5 ms, 0.1 for 100 ms).
    window : float
        Half-window in seconds (e.g. 0.5 or 5.0).
    normalize : bool
        If True, divide by ``n_events_a * bin_size`` to obtain rate (Hz).
        If False, return raw counts.
    n_surrogates : int
        Number of circular-shift surrogates to compute.  0 = no surrogates.
    seed : int or None
        Random seed for surrogate generation.

    Returns
    -------
    result : dict with keys
        ``lags``      — (n_bins,) bin-center lag values in seconds
        ``ccg``       — (n_bins,) observed cross-correlogram (counts or Hz)
        ``surrogates``— (n_surrogates, n_bins) surrogate CCGs (only if n_surrogates > 0)

    Notes
    -----
    Multi-timescale usage:

    - Fine:   ``bin_size=0.005, window=0.5``  → ±500 ms at 5 ms bins
    - Coarse: ``bin_size=0.1,   window=5.0``  → ±5 s at 100 ms bins

    For auto-correlogram pass the same event train as both arguments and
    set the zero-lag bin to NaN after the call.
    """
    ta = np.sort(_to_timestamps(events_a))
    tb = np.sort(_to_timestamps(events_b))

    bins = np.arange(-window, window + bin_size, bin_size)
    n_bins = len(bins) - 1

    def _compute_ccg(ta_: np.ndarray, tb_: np.ndarray) -> np.ndarray:
        counts = np.zeros(n_bins, dtype=np.float64)
        for t in ta_:
            lo = t - window
            hi = t + window
            # fast slice via searchsorted
            i0 = np.searchsorted(tb_, lo)
            i1 = np.searchsorted(tb_, hi, side="right")
            lags = tb_[i0:i1] - t
            c, _ = np.histogram(lags, bins=bins)
            counts += c
        if normalize and len(ta_) > 0:
            counts = counts / (len(ta_) * bin_size)
        return counts

    ccg = _compute_ccg(ta, tb)
    lags = 0.5 * (bins[:-1] + bins[1:])

    result: dict = {"lags": lags, "ccg": ccg}

    if n_surrogates > 0:
        rng = np.random.default_rng(seed)
        t_min = float(min(ta.min(), tb.min()))
        t_max = float(max(ta.max(), tb.max()))
        surr = np.empty((n_surrogates, n_bins), dtype=np.float64)
        for k in range(n_surrogates):
            tb_s = _circular_shift(tb, rng, t_min, t_max)
            surr[k] = _compute_ccg(ta, np.sort(tb_s))
        result["surrogates"] = surr

    return result


def peri_event_histogram(
    spikes,
    events,
    *,
    bin_size: float,
    window: float,
    baseline: tuple[float, float] | None = None,
    n_surrogates: int = 0,
    seed: int | None = None,
) -> dict:
    """
    Peri-event time histogram (PETH) of spike times aligned to event timestamps.

    Parameters
    ----------
    spikes : array-like or EventCatalog
        Spike (or event) timestamps in seconds.
    events : array-like or EventCatalog
        Trigger event timestamps in seconds.
    bin_size : float
        Bin width in seconds.
    window : float
        Half-window in seconds; output spans ``[-window, +window]``.
    baseline : (t_start, t_end) or None
        If given, normalize each bin by the mean rate in this lag range.
        Both values are relative to trigger (e.g. ``(-0.5, -0.1)``).
    n_surrogates : int
        Number of circular-shift surrogates.  0 = no surrogates.
    seed : int or None
        Random seed for surrogate generation.

    Returns
    -------
    result : dict with keys
        ``lags``       — (n_bins,) bin-center lag values in seconds
        ``peth``       — (n_bins,) mean firing rate across events (Hz)
        ``peth_sem``   — (n_bins,) standard error of the mean
        ``n_events``   — int, number of trigger events used
        ``surrogates`` — (n_surrogates, n_bins) surrogate PETHs (if requested)

    Notes
    -----
    Spike counts are converted to instantaneous firing rate (spikes/s) by
    dividing by ``bin_size``.  For multi-unit activity or LFP event trains
    pass any timestamp array.
    """
    ts = np.sort(_to_timestamps(spikes))
    te = np.sort(_to_timestamps(events))

    bins = np.arange(-window, window + bin_size, bin_size)
    n_bins = len(bins) - 1

    def _compute_peth(ts_: np.ndarray, te_: np.ndarray):
        """Return (n_events, n_bins) count matrix."""
        mat = np.zeros((len(te_), n_bins), dtype=np.float64)
        for i, t in enumerate(te_):
            lo = t - window
            hi = t + window
            i0 = np.searchsorted(ts_, lo)
            i1 = np.searchsorted(ts_, hi, side="right")
            lags = ts_[i0:i1] - t
            c, _ = np.histogram(lags, bins=bins)
            mat[i] = c
        return mat

    mat = _compute_peth(ts, te)
    rate = mat / bin_size  # convert to Hz
    peth = rate.mean(axis=0)
    peth_sem = rate.std(axis=0) / np.sqrt(max(len(te), 1))

    if baseline is not None:
        b0, b1 = float(baseline[0]), float(baseline[1])
        lags_c = 0.5 * (bins[:-1] + bins[1:])
        bl_mask = (lags_c >= b0) & (lags_c <= b1)
        bl_mean = peth[bl_mask].mean() if bl_mask.any() else 1.0
        if bl_mean > 0:
            peth = peth / bl_mean
            peth_sem = peth_sem / bl_mean

    lags = 0.5 * (bins[:-1] + bins[1:])
    result: dict = {"lags": lags, "peth": peth, "peth_sem": peth_sem, "n_events": len(te)}

    if n_surrogates > 0:
        rng = np.random.default_rng(seed)
        t_min = float(min(ts.min(), te.min()))
        t_max = float(max(ts.max(), te.max()))
        surr = np.empty((n_surrogates, n_bins), dtype=np.float64)
        for k in range(n_surrogates):
            ts_s = np.sort(_circular_shift(ts, rng, t_min, t_max))
            mat_s = _compute_peth(ts_s, te)
            surr[k] = mat_s.mean(axis=0) / bin_size
        result["surrogates"] = surr

    return result


def spectral_power_xcorr(
    power_a,
    power_b,
    *,
    max_lag: int,
    log_transform: bool = True,
    normalize: bool = True,
) -> dict:
    """
    Cross-correlation of two spectral power time series.

    Implements the Sirota et al. (2003) spectral power cross-correlation
    approach for quantifying coupling between oscillatory events (e.g.
    ripple power vs. spindle power).

    Parameters
    ----------
    power_a : array-like, shape (time,) or (freq, time)
        First spectral power time series.  If 2-D, correlates along
        the last (time) axis and returns a 2-D output.
    power_b : array-like, shape (time,) or (freq, time)
        Second spectral power time series.  Must have the same time
        length as ``power_a``.
    max_lag : int
        Maximum lag in **samples** (both positive and negative).
    log_transform : bool
        If True (default), apply ``log10(x + 1)`` before correlating.
        Reduces the effect of large transient bursts.
    normalize : bool
        If True (default), normalize to Pearson correlation coefficient
        (range ±1).  If False, return raw cross-correlation sums.

    Returns
    -------
    result : dict with keys
        ``lags`` — (2 * max_lag + 1,) integer lag indices
        ``xcorr``— (2 * max_lag + 1,) or (freq, 2 * max_lag + 1)
                   cross-correlation at each lag

    Notes
    -----
    Lag units are in **samples**; convert to seconds by dividing by
    the sampling rate of the power envelope (e.g. the spectrogram time step).

    Lag convention (standard cross-correlation):
    ``xcorr[l] = Σ_t  power_a[t] * power_b[t + l]``

    - Negative lag (l < 0): ``power_b`` leads ``power_a``
      (b's peak appears earlier in time than a's)
    - Positive lag (l > 0): ``power_a`` leads ``power_b``
    """
    a = np.asarray(power_a, dtype=np.float64)
    b = np.asarray(power_b, dtype=np.float64)

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"power_a and power_b must have the same time length; "
            f"got {a.shape[-1]} vs {b.shape[-1]}"
        )

    if log_transform:
        a = np.log10(np.clip(a, 0.0, None) + 1.0)
        b = np.log10(np.clip(b, 0.0, None) + 1.0)

    # Demean along time axis
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)

    n = a.shape[-1]
    lags = np.arange(-max_lag, max_lag + 1)

    def _xcorr_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty(len(lags), dtype=np.float64)
        for i, lag in enumerate(lags):
            if lag == 0:
                overlap_x, overlap_y = x, y
            elif lag > 0:
                # y leads x: align y[lag:] with x[:-lag]
                overlap_x = x[:-lag]
                overlap_y = y[lag:]
            else:
                # x leads y: align x[-lag:] with y[:lag]
                overlap_x = x[-lag:]
                overlap_y = y[:lag]
            out[i] = np.dot(overlap_x, overlap_y)
        if normalize:
            norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
            out = out / (norm + 1e-12)
        return out

    if a.ndim == 1:
        xcorr = _xcorr_1d(a, b)
    else:
        # Broadcast over leading dimensions (e.g. freq axis)
        shape_out = a.shape[:-1] + (len(lags),)
        xcorr = np.empty(shape_out, dtype=np.float64)
        for idx in np.ndindex(a.shape[:-1]):
            xcorr[idx] = _xcorr_1d(a[idx], b[idx])

    return {"lags": lags, "xcorr": xcorr}
