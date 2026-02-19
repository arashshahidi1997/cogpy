"""Raw temporal channel features (no spatial context).

All functions accept arrays shaped like `(..., time)` and reduce over the time
axis (default: last axis).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as nd

EPS = 1e-12


def relative_variance(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.nanvar(arr, axis=axis)


def deviation(arr: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.nanmean(arr, axis=axis)


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
