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

