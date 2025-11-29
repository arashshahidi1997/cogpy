"""Minimal sliding window helpers built only on NumPy."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
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


def sliding_window_da(x: np.ndarray, window_size: int, window_step: int = 1) -> np.ndarray:
    """Return a non-copying view of sliding windows using NumPy stride tricks."""

    x = np.asarray(x)
    _, n_windows = _validate_window_params(x, window_size, window_step)
    last_stride = x.strides[-1]
    new_shape = x.shape[:-1] + (n_windows, window_size)
    new_strides = x.strides[:-1] + (window_step * last_stride, last_stride)
    view = as_strided(x, shape=new_shape, strides=new_strides)
    view.setflags(write=False)
    return view


def sliding_window_naive(x: np.ndarray, window_size: int, window_step: int = 1) -> np.ndarray:
    """Slow reference implementation that copies each sliding window explicitly."""

    x = np.asarray(x)
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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 1_000_000))
    window_size = 256
    window_step = 64

    w_view = sliding_window_da(x, window_size, window_step)
    w_naive = sliding_window_naive(x, window_size, window_step)
    assert np.allclose(w_view, w_naive)

    start = time.perf_counter()
    w_view = sliding_window_da(x, window_size, window_step)
    t_view = time.perf_counter() - start

    start = time.perf_counter()
    w_naive = sliding_window_naive(x, window_size, window_step)
    t_naive = time.perf_counter() - start

    print(f"sliding_window_da: {t_view:.3f}s, mean={w_view.mean():.6f}")
    print(f"sliding_window_naive: {t_naive:.3f}s, mean={w_naive.mean():.6f}")
