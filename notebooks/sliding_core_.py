"""
Core sliding-window utilities for NumPy arrays.

This module provides:
- a low-level, non-copying sliding window view constructor (`sliding_window`)
- a slow reference implementation (`sliding_window_naive`) for testing
- mid-level helpers for applying functions over windows:
    * `running_reduce`  – reduces over window samples
    * `running_blockwise` – applies a function to the full core block per window

All functions are NumPy-only; higher-level xarray wrappers live elsewhere.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import as_strided

__all__ = [
    "sliding_window",
    "sliding_window_naive",
    "running_reduce",
    "running_blockwise",
    "benchmark_sliding",
]


def _validate_window_params(x: np.ndarray, window_size: int, window_step: int) -> Tuple[int, int]:
    """
    Validate sliding-window parameters for the last axis and return (N, n_windows).
    """

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if window_step <= 0:
        raise ValueError("window_step must be a positive integer")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension containing samples")
    n_samples = x.shape[-1]
    if n_samples < window_size:
        raise ValueError("window_size cannot exceed the length of the time axis")
    n_windows = 1 + (n_samples - window_size) // window_step
    if n_windows <= 0:
        raise ValueError("No windows can be formed with the given parameters")
    return n_samples, n_windows


def _window_view_last_axis(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
    """
    Return a non-copying sliding-window view along the last axis of `x`.
    """

    _, n_windows = _validate_window_params(x, window_size, window_step)
    last_stride = x.strides[-1]
    new_shape = x.shape[:-1] + (n_windows, window_size)
    new_strides = x.strides[:-1] + (window_step * last_stride, last_stride)
    view = as_strided(x, shape=new_shape, strides=new_strides)
    view.setflags(write=False)
    return view


def _naive_windows_last_axis(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
    """
    Slow, copy-based reference implementation of sliding windows along the last axis.
    """

    _, n_windows = _validate_window_params(x, window_size, window_step)
    out = np.empty(x.shape[:-1] + (n_windows, window_size), dtype=x.dtype)
    batch_shape = x.shape[:-1]

    if not batch_shape:
        for i in range(n_windows):
            start = i * window_step
            out[i, :] = x[start : start + window_size]
        return out

    for batch_idx in np.ndindex(batch_shape):
        series = x[batch_idx]
        target = out[batch_idx]
        for i in range(n_windows):
            start = i * window_step
            target[i, :] = series[start : start + window_size]
    return out


def sliding_window(
    x: np.ndarray,
    window_size: int,
    window_step: int = 1,
    axis: int = -1,
) -> np.ndarray:
    """
    Construct a non-copying sliding-window view along `axis`.

    The output shape follows the convention:
        shape_before_axis + (n_windows,) + shape_after_axis + (window_size,)
    where the final axis enumerates samples within each window. Internally this
    relies on NumPy's `as_strided`, so the returned view is read-only.
    """

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    moved = np.moveaxis(x, axis, -1)
    view = _window_view_last_axis(moved, window_size, window_step)
    return np.moveaxis(view, -2, axis)


def sliding_window_naive(
    x: np.ndarray,
    window_size: int,
    window_step: int = 1,
    axis: int = -1,
) -> np.ndarray:
    """
    Copy-based reference implementation of `sliding_window` for validation/testing.
    """

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    moved = np.moveaxis(x, axis, -1)
    out = _naive_windows_last_axis(moved, window_size, window_step)
    return np.moveaxis(out, -2, axis)


def running_reduce(
    x: np.ndarray,
    window_size: int,
    window_step: int,
    reducer: Callable,
    axis: int = -1,
    reducer_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    Apply `reducer` independently to each sliding window along `axis`.

    `reducer` must accept an `axis` keyword and reduce the final dimension,
    so the result typically has shape:
        shape_before_axis + (n_windows,) + shape_after_axis
    """

    x = np.asarray(x)
    reducer_kwargs = {} if reducer_kwargs is None else reducer_kwargs
    windows = sliding_window(x, window_size, window_step, axis=axis)
    return reducer(windows, axis=-1, **reducer_kwargs)


def running_blockwise(
    x: np.ndarray,
    window_size: int,
    window_step: int,
    func: Callable[[np.ndarray], np.ndarray],
    axis: int = -1,
) -> np.ndarray:
    """
    Apply `func` to the full core block of each window (gufunc-like helper).

    All axes except the per-window index are treated as the core block provided
    to `func`. The outputs from each window are stacked into an array of shape
    `(n_windows, *feature_shape)` where `feature_shape` is the shape returned by
    `func` for a single window. Useful for heavy per-window ops (PCA, SVD, etc.).
    """

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    windows = sliding_window(x, window_size, window_step, axis=axis)
    win_axis = axis
    windows_moved = np.moveaxis(windows, win_axis, 0)
    n_windows = windows_moved.shape[0]
    example_out = np.asarray(func(windows_moved[0]))
    feature_shape = example_out.shape
    out = np.empty((n_windows,) + feature_shape, dtype=example_out.dtype)
    out[0] = example_out
    for i in range(1, n_windows):
        out[i] = func(windows_moved[i])
    return out


def benchmark_sliding() -> None:
    """
    Simple timing and correctness checks for sliding_window vs sliding_window_naive.
    """

    import time

    rng = np.random.default_rng(123)
    cases = [
        (rng.standard_normal((32, 4096)), 128, 32, -1),
        (rng.standard_normal((2048, 4)), 3, 1, 0),
    ]

    for arr, win, step, axis in cases:
        start = time.perf_counter()
        ref = sliding_window_naive(arr, win, step, axis=axis)
        t_ref = time.perf_counter() - start

        start = time.perf_counter()
        view = sliding_window(arr, win, step, axis=axis)
        t_view = time.perf_counter() - start

        assert np.allclose(ref, view)
        print(
            f"shape={arr.shape}, axis={axis}, windows={view.shape}, "
            f"naive={t_ref*1e3:6.2f} ms, view={t_view*1e3:6.2f} ms"
        )


if __name__ == "__main__":
    benchmark_sliding()
