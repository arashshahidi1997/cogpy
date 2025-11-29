"""Sliding window helpers that support arbitrary axes using NumPy only."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import as_strided


def _validate_window_params(x: np.ndarray, window_size: int, window_step: int) -> Tuple[int, int]:
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_step <= 0:
        raise ValueError("window_step must be a positive integer")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension for the time axis")
    n_samples = x.shape[-1]
    if n_samples < window_size:
        raise ValueError("window_size cannot exceed the length of the time axis")
    n_windows = 1 + (n_samples - window_size) // window_step
    if n_windows <= 0:
        raise ValueError("No windows can be formed with the given parameters")
    return n_samples, n_windows


def _window_view_last_axis(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
    """Construct the sliding window view assuming the last axis is time."""

    _, n_windows = _validate_window_params(x, window_size, window_step)
    last_stride = x.strides[-1]
    new_shape = x.shape[:-1] + (n_windows, window_size)
    new_strides = x.strides[:-1] + (window_step * last_stride, last_stride)
    view = as_strided(x, shape=new_shape, strides=new_strides)
    view.setflags(write=False)
    return view


def _naive_windows_last_axis(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
    """Slow reference implementation that copies windows along the last axis."""

    _, n_windows = _validate_window_params(x, window_size, window_step)
    out_shape = x.shape[:-1] + (n_windows, window_size)
    out = np.empty(out_shape, dtype=x.dtype)
    batch_shape = x.shape[:-1]

    if not batch_shape:
        for i in range(n_windows):
            start = i * window_step
            out[i] = x[start : start + window_size]
        return out

    for batch_idx in np.ndindex(batch_shape):
        series = x[batch_idx]
        target = out[batch_idx]
        for i in range(n_windows):
            start = i * window_step
            target[i] = series[start : start + window_size]
    return out


def sliding_window_da(
    x: np.ndarray, window_size: int, window_step: int = 1, axis: int = -1
) -> np.ndarray:
    """Return a non-copying view of sliding windows using NumPy's as_strided."""

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    moved = np.moveaxis(x, axis, -1)
    view = _window_view_last_axis(moved, window_size, window_step)
    return np.moveaxis(view, -2, axis)


def sliding_window_naive(
    x: np.ndarray, window_size: int, window_step: int = 1, axis: int = -1
) -> np.ndarray:
    """Slow reference implementation that copies each sliding window explicitly."""

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    moved = np.moveaxis(x, axis, -1)
    out = _naive_windows_last_axis(moved, window_size, window_step)
    return np.moveaxis(out, -2, axis)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cases = [
        (rng.normal(size=(64, 1_000_000)), 256, 128, -1),
        (rng.normal(size=(1_000_000, 8)), 256, 65_536, 0),
    ]

    for arr, window_size, window_step, axis in cases:
        w_view = sliding_window_da(arr, window_size, window_step, axis=axis)
        w_naive = sliding_window_naive(arr, window_size, window_step, axis=axis)
        assert np.allclose(w_view, w_naive)

        start = time.perf_counter()
        w_view = sliding_window_da(arr, window_size, window_step, axis=axis)
        t_view = time.perf_counter() - start

        start = time.perf_counter()
        w_naive = sliding_window_naive(arr, window_size, window_step, axis=axis)
        t_naive = time.perf_counter() - start

        print(
            f"shape={arr.shape}, axis={axis}: da={t_view:.3f}s, naive={t_naive:.3f}s, mean={w_view.mean():.6f}"
        )
