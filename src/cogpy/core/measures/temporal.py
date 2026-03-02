"""
Temporal signal measures.

Status
------
STATUS: ACTIVE
Reason: General-purpose temporal characterization measures for iEEG signals.
Superseded by: n/a
Safe to remove: no

All functions accept arr: (..., time) and reduce over axis (default -1).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew as _scipy_skew

EPS = 1e-12

__all__ = [
    "relative_variance",
    "deviation",
    "standard_deviation",
    "amplitude",
    "time_derivative",
    "hurst_exponent",
    "dfa_exponent",
    "sample_entropy",
    "lempel_ziv",
    "kurtosis",
    "skewness",
    "hjorth_mobility",
    "hjorth_complexity",
    "jump_index",
    "zero_crossing_rate",
    "saturation_fraction",
]


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
    """
    Hurst exponent via R/S analysis.

    H > 0.5: persistent long-range correlations (typical healthy LFP).
    H < 0.5: anti-persistent.
    H ~ 0.5: white noise.

    Wraps nolds.hurst_rs. More reliable than pure numpy R/S estimator
    for short time series (N < 5000).

    Parameters
    ----------
    arr : (..., time)
    axis : int — time axis (default -1)

    Returns
    -------
    H : same shape as arr with axis removed.

    Notes
    -----
    nolds.hurst_rs is biased for very short series (N < 500).
    For non-stationary signals prefer dfa_exponent.
    """
    try:
        import nolds  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = next(
            p for p in Path(__file__).resolve().parents if (p / "code" / "lib").is_dir()
        )
        sys.path.insert(0, str(repo_root / "code" / "lib" / "nolds"))
        import nolds  # type: ignore[import-not-found]

    arr = np.moveaxis(arr, axis, -1)
    out = np.apply_along_axis(nolds.hurst_rs, -1, arr)
    return out


def dfa_exponent(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Detrended Fluctuation Analysis (DFA) scaling exponent.

    More robust than Hurst R/S for non-stationary signals.
    Alpha ~ 0.5: uncorrelated (white noise).
    Alpha ~ 1.0: 1/f noise (typical LFP).
    Alpha > 1.0: non-stationary, strong long-range correlations.

    Wraps nolds.dfa.

    Parameters
    ----------
    arr : (..., time)
    axis : int — time axis (default -1)

    Returns
    -------
    alpha : same shape as arr with axis removed.
    """
    try:
        import nolds  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = next(
            p for p in Path(__file__).resolve().parents if (p / "code" / "lib").is_dir()
        )
        sys.path.insert(0, str(repo_root / "code" / "lib" / "nolds"))
        import nolds  # type: ignore[import-not-found]

    arr = np.moveaxis(arr, axis, -1)
    out = np.apply_along_axis(nolds.dfa, -1, arr)
    return out


def sample_entropy(
    arr: np.ndarray, *, axis: int = -1, order: int = 2, metric: str = "chebyshev"
) -> np.ndarray:
    """
    Sample Entropy (SampEn).

    Probability that patterns of m samples that match will still
    match at m+1 samples. Lower = more regular. Higher = more complex.

    Wraps antropy.sample_entropy.

    Parameters
    ----------
    arr : (..., time)
    axis : int — time axis (default -1)
    order : int — template length m (default 2)
    metric : str — distance metric (default 'chebyshev')

    Returns
    -------
    entropy : same shape as arr with axis removed.

    Notes
    -----
    Tolerance r is set automatically by antropy as 0.2 * std(x).
    """
    try:
        import antropy  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = next(
            p for p in Path(__file__).resolve().parents if (p / "code" / "lib").is_dir()
        )
        sys.path.insert(0, str(repo_root / "code" / "lib" / "antropy" / "src"))
        import antropy  # type: ignore[import-not-found]

    arr = np.moveaxis(arr, axis, -1)

    def fn(x):
        return antropy.sample_entropy(x, order=order, metric=metric)

    out = np.apply_along_axis(fn, -1, arr)
    return out


def lempel_ziv(arr: np.ndarray, *, axis: int = -1, normalize: bool = True) -> np.ndarray:
    """
    Lempel-Ziv complexity on binarized signal.

    Signal binarized at median before LZ76 parsing.
    Lower = more repetitive/structured.
    Higher = more random/complex.

    Wraps antropy.lziv_complexity.

    Parameters
    ----------
    arr : (..., time)
    axis : int — time axis (default -1)
    normalize : bool — normalize by theoretical maximum (default True)

    Returns
    -------
    lzc : same shape as arr with axis removed.

    Notes
    -----
    Binarization: x > median(x). Sensitive to binarization threshold —
    median is robust but consider mean for symmetric distributions.
    """
    try:
        import antropy  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        repo_root = next(
            p for p in Path(__file__).resolve().parents if (p / "code" / "lib").is_dir()
        )
        sys.path.insert(0, str(repo_root / "code" / "lib" / "antropy" / "src"))
        import antropy  # type: ignore[import-not-found]

    arr = np.moveaxis(arr, axis, -1)

    def _lzc(x):
        binary = (x > np.median(x)).astype(int)
        val = antropy.lziv_complexity(binary, normalize=normalize)
        if normalize:
            return float(np.clip(val, 0.0, 1.0))
        return val

    out = np.apply_along_axis(_lzc, -1, arr)
    return out


def kurtosis(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    mean = np.nanmean(x, axis=axis, keepdims=True)
    centered = x - mean
    m2 = np.nanmean(centered**2, axis=axis)
    m4 = np.nanmean(centered**4, axis=axis)
    return m4 / ((m2**2) + EPS)


def skewness(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Asymmetry of amplitude distribution via scipy.stats.skew.
    Non-zero skewness indicates spike contamination or asymmetric noise.
    nan_policy="omit" used for robustness.
    """
    return _scipy_skew(arr, axis=axis, nan_policy="omit", bias=True)


def hjorth_mobility(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Normalized mean frequency: sqrt(var(dx/dt) / var(x)).
    Elevated mobility relative to neighbors indicates broadband noise.
    Uses np.gradient for derivative.
    """
    dx = np.gradient(arr, axis=axis)
    return np.sqrt(np.nanvar(dx, axis=axis) / (np.nanvar(arr, axis=axis) + EPS))


def hjorth_complexity(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Ratio of mobility of derivative to mobility of signal.
    Pure sine = 1.0. White noise >> 1.
    """
    dx = np.gradient(arr, axis=axis)
    ddx = np.gradient(dx, axis=axis)
    mob_x = np.sqrt(np.nanvar(dx, axis=axis) / (np.nanvar(arr, axis=axis) + EPS))
    mob_dx = np.sqrt(np.nanvar(ddx, axis=axis) / (np.nanvar(dx, axis=axis) + EPS))
    return mob_dx / (mob_x + EPS)


def jump_index(arr: np.ndarray, *, axis: int = -1, robust: bool = True) -> np.ndarray:
    """
    max(|diff(x)|) / scale(x).
    High jump + low kurtosis = mechanical/cable artifact.
    robust=True uses MAD*1.4826 as scale; False uses std.
    """
    dx = np.diff(arr, axis=axis)
    peak = np.nanmax(np.abs(dx), axis=axis)
    if robust:
        med = np.nanmedian(arr, axis=axis, keepdims=True)
        scale = np.nanmedian(np.abs(arr - med), axis=axis) * 1.4826
    else:
        scale = np.nanstd(arr, axis=axis)
    return peak / (scale + EPS)


def zero_crossing_rate(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Fraction of samples where signal changes sign.
    zcr = mean(sign(x[1:]) != sign(x[:-1])).
    High ZCR = high-frequency noise. Low ZCR = DC drift.
    Output in [0, 1].
    """
    crossings = np.diff(np.sign(arr), axis=axis) != 0
    return np.nanmean(crossings.astype(float), axis=axis)


def saturation_fraction(
    arr: np.ndarray, *, axis: int = -1, adc_max: float, eps: float = 2.0
) -> np.ndarray:
    """
    Fraction of samples within eps of ADC clipping limit.
    sat = mean(|x| > adc_max - eps).
    Clipped channels corrupt coherence and introduce harmonics.
    adc_max has no default — caller must provide it explicitly.
    """
    return np.nanmean(np.abs(arr) > (adc_max - eps), axis=axis)
