"""Core sliding-window utilities for NumPy arrays.
TAG: v1.0.0

This module provides:
- a low-level, non-copying sliding window view constructor (`sliding_window`)
- a slow reference implementation (`sliding_window_naive`) for testing
- mid-level helpers for applying functions over windows:
    * `running_reduce`  – reduces over window samples
    * `running_blockwise` – applies a function to the full core block per window

All functions are NumPy-only; higher-level xarray wrappers live elsewhere.

Examples
--------
>>> import numpy as np
>>> from cogpy.core.utils.sliding_core import sliding_window, running_reduce
>>> x = np.arange(100)
>>> xwin = sliding_window(x, window_size=10, window_step=5)
>>> xwin.shape
(19, 10)
>>> xmulti = np.arange(16*100).reshape(16, 100)
>>> xmulti_win = sliding_window(xmulti, window_size=10, window_step=5, axis=1)
>>> xmulti_win.shape
(16, 19, 10)
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
from tqdm.auto import trange as tqdrange

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import as_strided

__all__ = [
    "sliding_window",
    "sliding_window_naive",
    "window_onsets",
    "window_ends",
    "window_centers_idx",
    "window_centers_time",
    "running_reduce",
    "running_blockwise",
    "running_reduce_xr",
    "running_blockwise_xr",
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


def window_onsets(n_samples: int, window_size: int, window_step: int) -> np.ndarray:
    """Return window onset indices for a 1D axis of length ``n_samples``."""
    n_samples = int(n_samples)
    window_size = int(window_size)
    window_step = int(window_step)
    if n_samples < window_size:
        raise ValueError("window_size cannot exceed the length of the time axis")
    n_windows = 1 + (n_samples - window_size) // window_step
    if n_windows <= 0:
        raise ValueError("No windows can be formed with the given parameters")
    return np.arange(n_windows, dtype=int) * window_step


def window_ends(n_samples: int, window_size: int, window_step: int) -> np.ndarray:
    """Return window end indices (inclusive) for a 1D axis of length ``n_samples``."""
    on = window_onsets(n_samples, window_size, window_step)
    return on + (int(window_size) - 1)


def window_centers_idx(
    n_samples: int,
    window_size: int,
    window_step: int,
    *,
    method: str = "midpoint",
) -> np.ndarray:
    """Return window center indices as floats.

    Parameters
    ----------
    method
        ``"midpoint"`` (default) returns ``onset + (window_size-1)/2`` which yields
        half-sample centers for even window sizes.
        ``"lower"`` returns integer centers via ``onset + floor((window_size-1)/2)``.
        ``"upper"`` returns integer centers via ``onset + ceil((window_size-1)/2)``.
    """
    on = window_onsets(n_samples, window_size, window_step).astype(float)
    w = float(int(window_size) - 1)
    m = str(method).lower()
    if m == "midpoint":
        return on + w / 2.0
    if m == "lower":
        return on + np.floor(w / 2.0)
    if m == "upper":
        return on + np.ceil(w / 2.0)
    raise ValueError("method must be one of {'midpoint','lower','upper'}")


def window_centers_time(
    t: np.ndarray,
    window_size: int,
    window_step: int,
    *,
    method: str = "midpoint",
) -> np.ndarray:
    """Return window center times from a 1D time vector ``t``.

    For ``method="midpoint"`` (default) this computes the midpoint between
    the start and end sample times of each window:
        ``0.5*(t[onset] + t[end])``
    This is robust to small irregularities in sampling.

    For ``method="lower"``/``"upper"``, it indexes the corresponding center sample.
    """
    t = np.asarray(t)
    if t.ndim != 1:
        raise ValueError(f"t must be 1D, got shape {t.shape}")
    on = window_onsets(t.size, window_size, window_step)
    en = on + (int(window_size) - 1)
    m = str(method).lower()
    if m == "midpoint":
        return 0.5 * (t[on] + t[en])
    if m == "lower":
        idx = window_centers_idx(t.size, window_size, window_step, method="lower").astype(int)
        return t[idx]
    if m == "upper":
        idx = window_centers_idx(t.size, window_size, window_step, method="upper").astype(int)
        return t[idx]
    raise ValueError("method must be one of {'midpoint','lower','upper'}")


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
    *,
    return_centers: bool = False,
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
    y = reducer(windows, axis=-1, **reducer_kwargs)
    if not return_centers:
        return y
    axis = normalize_axis_index(axis, x.ndim)
    centers = window_centers_idx(x.shape[axis], window_size, window_step)
    return y, centers


def running_blockwise(
    x: np.ndarray,
    window_size: int,
    window_step: int,
    func: Callable[[np.ndarray], np.ndarray],
    axis: int = -1,
    *,
    return_centers: bool = False,
    progress: bool = True,
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
    it = tqdrange(n_windows) if progress else range(n_windows)
    for i in it:
        if i == 0:
            out[i] = example_out
        else:
            out[i] = func(windows_moved[i])
    if not return_centers:
        return out
    centers = window_centers_idx(x.shape[axis], window_size, window_step)
    return out, centers


def _infer_feature_dims(
    x_dims: tuple[str, ...],
    *,
    run_dim: str,
    feature_shape: tuple[int, ...],
) -> tuple[str, ...]:
    """
    Infer feature dims for a running window operation.

    If the function output shape matches the input dims excluding `run_dim`,
    reuse those dim names; otherwise create generic feature dim names.
    """
    remaining_dims = tuple(d for d in x_dims if d != run_dim)
    # Only infer when the shapes are compatible; for all other cases, fall back.
    # NOTE: this matches by *count* only. It assumes `func` returns outputs whose
    # axes follow the same order as `remaining_dims`. If your function reorders
    # axes (e.g. swaps spatial dims), pass `feature_dims` explicitly.
    if len(feature_shape) == len(remaining_dims):
        return remaining_dims
    return tuple(f"feat{i}" for i in range(len(feature_shape)))


def running_blockwise_xr(
    xsig,
    window_size: int,
    window_step: int,
    func: Callable[[np.ndarray], np.ndarray],
    *,
    run_dim: str = "time",
    out_dim: str = "time_win",
    center_method: str = "midpoint",
    feature_dims: Optional[tuple[str, ...]] = None,
    return_centers: bool = False,
    progress: bool = True,
):
    """xarray wrapper for :func:`running_blockwise`.

    Returns an ``xr.DataArray`` with a windowed coordinate on ``out_dim``.
    By default the coordinate is computed from the midpoint of each window in
    ``xsig[run_dim]`` (if present), otherwise it is returned as sample indices.

    Notes
    -----
    `func` receives a NumPy array with the per-window samples on the *last axis*.
    The remaining axes (excluding `run_dim`) follow the input dim order of `xsig`
    with `run_dim` removed. If your function expects a specific axis order,
    transpose inside `func` or call ``xsig.transpose(...)`` before passing it in.
    """
    import xarray as xr

    if not isinstance(xsig, xr.DataArray):
        raise TypeError("running_blockwise_xr expects an xarray.DataArray")
    if run_dim not in xsig.dims:
        raise ValueError(f"run_dim={run_dim!r} not in xsig.dims={tuple(xsig.dims)}")

    axis = xsig.get_axis_num(run_dim)
    out, centers_idx = running_blockwise(
        xsig.data,
        window_size=int(window_size),
        window_step=int(window_step),
        func=func,
        axis=int(axis),
        return_centers=True,
        progress=bool(progress),
    )

    feature_shape = tuple(np.asarray(out).shape[1:])
    if feature_dims is None:
        feature_dims = _infer_feature_dims(tuple(xsig.dims), run_dim=run_dim, feature_shape=feature_shape)
    if len(feature_dims) != len(feature_shape):
        raise ValueError(
            f"feature_dims has length {len(feature_dims)} but func output has {len(feature_shape)} dims"
        )

    coords = {}
    # window coordinate
    if run_dim in xsig.coords:
        coords[out_dim] = window_centers_time(
            xsig.coords[run_dim].values, int(window_size), int(window_step), method=str(center_method)
        )
    else:
        coords[out_dim] = centers_idx

    # pass through coords for any inferred feature dims (when possible)
    for d in feature_dims:
        if d in xsig.coords:
            coords[d] = xsig.coords[d].values

    da = xr.DataArray(
        out,
        dims=(out_dim,) + tuple(feature_dims),
        coords=coords,
        attrs=dict(xsig.attrs),
        name=xsig.name,
    )
    da.attrs.update({"window_size": int(window_size), "window_step": int(window_step), "run_dim": str(run_dim)})
    if return_centers:
        return da, centers_idx
    return da


def running_reduce_xr(
    xsig,
    window_size: int,
    window_step: int,
    reducer: Callable,
    *,
    run_dim: str = "time",
    out_dim: str = "time_win",
    center_method: str = "midpoint",
    reducer_kwargs: Optional[dict] = None,
    return_centers: bool = False,
):
    """xarray wrapper for :func:`running_reduce`.

    The reducer is expected to reduce the per-window sample axis, so the output
    feature dims match ``xsig.dims`` with ``run_dim`` replaced by ``out_dim``.
    """
    import xarray as xr

    if not isinstance(xsig, xr.DataArray):
        raise TypeError("running_reduce_xr expects an xarray.DataArray")
    if run_dim not in xsig.dims:
        raise ValueError(f"run_dim={run_dim!r} not in xsig.dims={tuple(xsig.dims)}")

    axis = xsig.get_axis_num(run_dim)
    y, centers_idx = running_reduce(
        xsig.data,
        window_size=int(window_size),
        window_step=int(window_step),
        reducer=reducer,
        axis=int(axis),
        reducer_kwargs=reducer_kwargs,
        return_centers=True,
    )

    dims_out = tuple(out_dim if d == run_dim else d for d in xsig.dims)
    coords = {}
    for d in dims_out:
        if d == out_dim:
            continue
        if d in xsig.coords:
            coords[d] = xsig.coords[d].values
    if run_dim in xsig.coords:
        coords[out_dim] = window_centers_time(
            xsig.coords[run_dim].values, int(window_size), int(window_step), method=str(center_method)
        )
    else:
        coords[out_dim] = centers_idx

    da = xr.DataArray(
        y,
        dims=dims_out,
        coords=coords,
        attrs=dict(xsig.attrs),
        name=xsig.name,
    )
    da.attrs.update({"window_size": int(window_size), "window_step": int(window_step), "run_dim": str(run_dim)})
    if return_centers:
        return da, centers_idx
    return da


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
