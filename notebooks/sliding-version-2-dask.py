"""Version 2: sliding window helper with shared NumPy/Dask API building view-based windows."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import as_strided

try:
    import dask.array as da
    from dask.array.lib.stride_tricks import sliding_window_view as da_sliding_window_view
except Exception:  # pragma: no cover - optional dependency
    da = None
    da_sliding_window_view = None


def _validate_window_params(x, window_size: int, window_step: int) -> Tuple[int, int]:
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


def _window_view_last_axis_np(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
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


def _is_dask_array(x) -> bool:
    return da is not None and isinstance(x, da.Array)


def sliding_window_da(
    x, window_size: int, window_step: int = 1, axis: int = -1
):
    """Return sliding windows as a view (NumPy) or lazy graph (Dask) without copying."""

    if _is_dask_array(x):
        axis = normalize_axis_index(axis, x.ndim)
        moved = da.moveaxis(x, axis, -1)
        _validate_window_params(moved, window_size, window_step)
        windows = da_sliding_window_view(moved, window_size, axis=-1)
        windows = windows[..., ::window_step, :]
        return da.moveaxis(windows, -2, axis)

    x_np = np.asarray(x)
    axis = normalize_axis_index(axis, x_np.ndim)
    moved = np.moveaxis(x_np, axis, -1)
    view = _window_view_last_axis_np(moved, window_size, window_step)
    return np.moveaxis(view, -2, axis)


def sliding_window_naive(
    x: np.ndarray, window_size: int, window_step: int = 1, axis: int = -1
) -> np.ndarray:
    """Slow NumPy-only reference implementation that copies each sliding window explicitly."""

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

    if da is not None:
        x_np = rng.normal(size=(64, 1_000_000))
        x_da = da.from_array(x_np, chunks=(16, 100_000))
        window_size = 256
        window_step = 128

        w_np = sliding_window_da(x_np, window_size, window_step, axis=-1)
        start = time.perf_counter()
        w_da = sliding_window_da(x_da, window_size, window_step, axis=-1).compute()
        t_compute = time.perf_counter() - start
        assert np.allclose(w_da, w_np)
        print(
            f"Dask compute: shape={w_da.shape}, chunks={x_da.chunks}, compute_time={t_compute:.3f}s, mean={w_da.mean():.6f}"
        )
    else:
        print("Dask is not available; skipping Dask benchmark.")
