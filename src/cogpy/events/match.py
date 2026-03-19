"""
Event matching and lag estimation primitives.

Pure functions for aligning and comparing discrete event trains
across signals or streams.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "match_nearest",
    "match_nearest_symmetric",
    "event_lag_histogram",
    "estimate_lag",
    "estimate_drift",
]


def match_nearest(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    max_lag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match each event in *times_a* to its nearest event in *times_b*.

    For each event in *times_a*, finds the closest event in *times_b*
    within ``±max_lag``.  Events in *times_b* may be matched to multiple
    events in *times_a* (greedy, not bijective).

    Parameters
    ----------
    times_a : (N,) float
        Sorted event times for signal A (seconds).
    times_b : (M,) float
        Sorted event times for signal B (seconds).
    max_lag : float
        Maximum allowed absolute lag (seconds).  Pairs with
        ``|t_a - t_b| > max_lag`` are excluded.

    Returns
    -------
    idx_a : (K,) int
        Indices into *times_a* that were matched.
    idx_b : (K,) int
        Corresponding indices into *times_b*.
    lags : (K,) float
        Signed lags ``times_b[idx_b] - times_a[idx_a]`` (seconds).
        Positive means B follows A.
    """
    a = np.asarray(times_a, dtype=float).ravel()
    b = np.asarray(times_b, dtype=float).ravel()
    max_lag = float(max_lag)
    if max_lag <= 0:
        raise ValueError(f"max_lag must be positive, got {max_lag}")

    if a.size == 0 or b.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty.copy(), np.array([], dtype=float)

    # Vectorized nearest-neighbor via searchsorted + two-candidate check.
    j_right = np.searchsorted(b, a, side="left")  # index of first b >= a
    j_right = np.clip(j_right, 0, len(b) - 1)
    j_left = np.clip(j_right - 1, 0, len(b) - 1)

    dist_right = np.abs(b[j_right] - a)
    dist_left = np.abs(b[j_left] - a)

    use_left = dist_left < dist_right  # prefer right on tie
    best_idx_b = np.where(use_left, j_left, j_right)
    best_lag = b[best_idx_b] - a

    mask = np.abs(best_lag) <= max_lag
    idx_a = np.where(mask)[0]
    return idx_a, best_idx_b[mask], best_lag[mask]


def match_nearest_symmetric(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    max_lag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bijective nearest-neighbor matching between two event trains.

    Like :func:`match_nearest`, but enforces one-to-one matching:
    each event in B is matched to at most one event in A (the closest).
    Ties are broken by smallest absolute lag.

    Parameters
    ----------
    times_a, times_b : (N,) and (M,) float
        Sorted event times (seconds).
    max_lag : float
        Maximum allowed absolute lag (seconds).

    Returns
    -------
    idx_a : (K,) int
        Indices into *times_a*.
    idx_b : (K,) int
        Corresponding indices into *times_b*.
    lags : (K,) float
        Signed lags ``times_b[idx_b] - times_a[idx_a]``.
    """
    idx_a_fwd, idx_b_fwd, lags_fwd = match_nearest(times_a, times_b, max_lag=max_lag)
    if idx_a_fwd.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty.copy(), np.array([], dtype=float)

    # Resolve duplicates in idx_b: keep the pair with smallest |lag|
    seen_b: dict[int, int] = {}  # idx_b → position in output arrays
    for pos in range(len(idx_a_fwd)):
        ib = int(idx_b_fwd[pos])
        if ib not in seen_b or abs(lags_fwd[pos]) < abs(lags_fwd[seen_b[ib]]):
            seen_b[ib] = pos
    keep = sorted(seen_b.values())
    return idx_a_fwd[keep], idx_b_fwd[keep], lags_fwd[keep]


def event_lag_histogram(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    max_lag: float,
    bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlogram (lag histogram) between two event trains.

    For every pair ``(a_i, b_j)`` with ``|a_i - b_j| <= max_lag``,
    accumulates the signed lag ``b_j - a_i`` into a histogram.

    Parameters
    ----------
    times_a, times_b : (N,) and (M,) float
        Sorted event times (seconds).
    max_lag : float
        Maximum absolute lag to consider (seconds).
    bin_width : float
        Histogram bin width (seconds).

    Returns
    -------
    counts : (n_bins,) int
        Histogram counts.
    bin_edges : (n_bins + 1,) float
        Bin edges in seconds.
    """
    a = np.asarray(times_a, dtype=float).ravel()
    b = np.asarray(times_b, dtype=float).ravel()
    max_lag = float(max_lag)
    bin_width = float(bin_width)

    n_bins = int(np.ceil(2 * max_lag / bin_width))
    bin_edges = np.linspace(-max_lag, max_lag, n_bins + 1)

    # Collect all pairwise lags within max_lag using binary search.
    lag_chunks: list[np.ndarray] = []
    for ta in a:
        j_lo = np.searchsorted(b, ta - max_lag, side="left")
        j_hi = np.searchsorted(b, ta + max_lag, side="right")
        if j_lo < j_hi:
            lag_chunks.append(b[j_lo:j_hi] - ta)

    if lag_chunks:
        lags_all = np.concatenate(lag_chunks)
        counts, _ = np.histogram(lags_all, bins=bin_edges)
    else:
        counts = np.zeros(n_bins, dtype=int)

    return counts, bin_edges


def estimate_lag(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    max_lag: float,
    method: str = "median",
) -> float:
    """
    Estimate constant temporal lag between two event trains.

    Parameters
    ----------
    times_a, times_b : (N,) and (M,) float
        Sorted event times (seconds).
    max_lag : float
        Maximum absolute lag for matching (seconds).
    method : {"median", "mode", "mean"}
        How to aggregate matched lags.

        - ``"median"``: robust to outliers (default).
        - ``"mode"``: peak of lag histogram (bin width = max_lag / 50).
        - ``"mean"``: simple average.

    Returns
    -------
    float
        Estimated lag (seconds).  Positive means B follows A.
        Returns ``nan`` if no matches found.
    """
    _, _, lags = match_nearest(times_a, times_b, max_lag=max_lag)
    if lags.size == 0:
        return float("nan")

    method = str(method)
    if method == "median":
        return float(np.median(lags))
    elif method == "mean":
        return float(np.mean(lags))
    elif method == "mode":
        bin_width = max_lag / 50
        counts, edges = np.histogram(lags, bins=int(np.ceil(2 * max_lag / bin_width)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return float(centers[np.argmax(counts)])
    else:
        raise ValueError(f"Unknown method {method!r}; use 'median', 'mode', or 'mean'")


def estimate_drift(
    times_a: np.ndarray,
    times_b: np.ndarray,
    *,
    max_lag: float,
    degree: int = 1,
) -> np.ndarray:
    """
    Estimate time-varying drift between two event trains.

    Fits a polynomial to the matched lags as a function of time,
    capturing slow drift (e.g. clock skew between acquisition systems).

    Parameters
    ----------
    times_a, times_b : (N,) and (M,) float
        Sorted event times (seconds).
    max_lag : float
        Maximum absolute lag for matching (seconds).
    degree : int
        Polynomial degree (1 = linear drift, 2 = quadratic, etc.).

    Returns
    -------
    coeffs : (degree + 1,) float
        Polynomial coefficients, highest degree first (numpy polyfit convention).
        ``np.polyval(coeffs, t)`` gives estimated lag at time ``t``.
        Returns array of ``nan`` if fewer than ``degree + 1`` matches.
    """
    idx_a, _, lags = match_nearest(times_a, times_b, max_lag=max_lag)
    a = np.asarray(times_a, dtype=float).ravel()

    if len(idx_a) < degree + 1:
        return np.full(degree + 1, np.nan)

    matched_times = a[idx_a]
    coeffs = np.polyfit(matched_times, lags, deg=degree)
    return coeffs
