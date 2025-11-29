"""Unified sliding window utilities (versions 0-3) for NumPy, Dask, and xarray."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from numpy.core.multiarray import normalize_axis_index
from numpy.lib.stride_tricks import as_strided

try:
    import dask.array as da
    from dask.array.lib.stride_tricks import sliding_window_view as da_sliding_window_view
except Exception:  # pragma: no cover - optional dependency
    da = None
    da_sliding_window_view = None

__all__ = [
    "sliding_window_da_last",
    "sliding_window_naive_last",
    "sliding_window_da_numpy",
    "sliding_window_naive_numpy",
    "sliding_window_da",
    "sliding_window_xr",
    "compare_all_variants",
]


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
    _, n_windows = _validate_window_params(x, window_size, window_step)
    last_stride = x.strides[-1]
    new_shape = x.shape[:-1] + (n_windows, window_size)
    new_strides = x.strides[:-1] + (window_step * last_stride, last_stride)
    view = as_strided(x, shape=new_shape, strides=new_strides)
    view.setflags(write=False)
    return view


def _naive_windows_last_axis(x: np.ndarray, window_size: int, window_step: int) -> np.ndarray:
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


def sliding_window_da_last(x: np.ndarray, window_size: int, window_step: int = 1) -> np.ndarray:
    """Version 0: assume the last axis is time and return a non-copying NumPy view."""

    x = np.asarray(x)
    return _window_view_last_axis_np(x, window_size, window_step)


def sliding_window_naive_last(x: np.ndarray, window_size: int, window_step: int = 1) -> np.ndarray:
    """Version 0 reference: slow copy-based implementation along the last axis."""

    x = np.asarray(x)
    return _naive_windows_last_axis(x, window_size, window_step)


def sliding_window_da_numpy(
    x: np.ndarray, window_size: int, window_step: int = 1, axis: int = -1
) -> np.ndarray:
    """Version 1: NumPy view that supports any axis via moveaxis."""

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    moved = np.moveaxis(x, axis, -1)
    view = _window_view_last_axis_np(moved, window_size, window_step)
    return np.moveaxis(view, -2, axis)


def sliding_window_naive_numpy(
    x: np.ndarray, window_size: int, window_step: int = 1, axis: int = -1
) -> np.ndarray:
    """Version 1 reference: slow copy implementation supporting arbitrary axes."""

    x = np.asarray(x)
    axis = normalize_axis_index(axis, x.ndim)
    moved = np.moveaxis(x, axis, -1)
    out = _naive_windows_last_axis(moved, window_size, window_step)
    return np.moveaxis(out, -2, axis)


def sliding_window_da(
    x, window_size: int, window_step: int = 1, axis: int = -1
):
    """Version 2: type-aware helper (NumPy view or Dask graph) supporting any axis."""

    if _is_dask_array(x):
        axis = normalize_axis_index(axis, x.ndim)
        moved = da.moveaxis(x, axis, -1)
        _validate_window_params(moved, window_size, window_step)
        windows = da_sliding_window_view(moved, window_size, axis=-1)
        windows = windows[..., ::window_step, :]
        return da.moveaxis(windows, -2, axis)

    return sliding_window_da_numpy(np.asarray(x), window_size, window_step, axis=axis)


def sliding_window_xr(
    xsig: xr.DataArray,
    window_size: int,
    window_step: int = 1,
    dim: str = "time",
    window_dim: str = "window",
    sample_dim: str = "window_sample",
) -> xr.DataArray:
    """Version 3: thin xarray wrapper that delegates window building to sliding_window_da."""

    if dim not in xsig.dims:
        raise ValueError(f"Dimension '{dim}' not present in the input DataArray")

    axis = xsig.get_axis_num(dim)
    data = sliding_window_da(xsig.data, window_size, window_step, axis=axis)
    dims = list(xsig.dims)
    dims[axis] = window_dim
    dims.append(sample_dim)

    coords = {}
    for name in xsig.dims:
        if name == dim:
            continue
        if name in xsig.coords:
            coords[name] = xsig.coords[name]
        else:
            coords[name] = (name, np.arange(xsig.sizes[name]))

    if dim in xsig.coords:
        base = np.asarray(xsig.coords[dim].values)
    else:
        base = np.arange(xsig.sizes[dim])

    n_windows = data.shape[axis]
    starts = np.arange(n_windows) * window_step
    ends = starts + window_size - 1
    if np.issubdtype(base.dtype, np.datetime64) or np.issubdtype(base.dtype, np.timedelta64):
        centers = base[starts] + (base[ends] - base[starts]) / 2
    elif np.issubdtype(base.dtype, np.number):
        centers = (base[starts] + base[ends]) / 2
    else:
        centers = base[starts]

    coords[window_dim] = (window_dim, centers)
    coords[sample_dim] = (sample_dim, np.arange(window_size))

    return xr.DataArray(data, dims=dims, coords=coords, attrs=xsig.attrs)


def compare_all_variants() -> None:
    """Run the cross-version timing/validation harness from sliding-version-all-test."""

    def _time_call(func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    def _print_case(case_name: str, timings: List[Tuple[str, float]]):
        print(f"\nCase: {case_name}")
        for label, elapsed in timings:
            print(f"  {label:<30s} {elapsed*1e3:8.2f} ms")

    rng = np.random.default_rng(123)
    cases: Dict[str, Dict[str, object]] = {
        "batch_last": {
            "array": rng.normal(size=(32, 4096)),
            "window_size": 128,
            "window_step": 32,
            "axis": -1,
        },
        "batch_first": {
            "array": rng.normal(size=(2048, 4)),
            "window_size": 128,
            "window_step": 64,
            "axis": 0,
        },
        # NEW realistic-ish case
        "realistic_256x1e6": {
            # close to your shape (256 channels, 10^6 samples),
            # use float32 to keep memory down
            "array": rng.normal(size=(256, 1_000_000)).astype(np.float32),
            "window_size": 1024,
            "window_step": 256,
            "axis": -1,  # time is last axis
        },
    }

    for name, cfg in cases.items():
        array = cfg["array"]  # type: ignore[assignment]
        window_size = cfg["window_size"]  # type: ignore[assignment]
        window_step = cfg["window_step"]  # type: ignore[assignment]
        axis = cfg["axis"]  # type: ignore[assignment]

        timings: List[Tuple[str, float]] = []

        ref, t_ref = _time_call(
            sliding_window_naive_numpy, array, window_size, window_step, axis=axis
        )
        timings.append(("v1.naive (reference)", t_ref))

        if axis == array.ndim - 1:
            v0_naive, t_naive0 = _time_call(
                sliding_window_naive_last, array, window_size, window_step
            )
            assert np.allclose(v0_naive, ref)
            timings.append(("v0.naive", t_naive0))

            view0, t_view0 = _time_call(sliding_window_da_last, array, window_size, window_step)
            assert np.allclose(view0, ref)
            timings.append(("v0.as_strided", t_view0))

        v1_view, t_v1 = _time_call(
            sliding_window_da_numpy, array, window_size, window_step, axis=axis
        )
        assert np.allclose(v1_view, ref)
        timings.append(("v1.as_strided", t_v1))

        v2_view, t_v2 = _time_call(sliding_window_da, array, window_size, window_step, axis=axis)
        assert np.allclose(v2_view, ref)
        timings.append(("v2.NumPy", t_v2))

        if da is not None:
            x_da = da.from_array(array, chunks=tuple(max(1, s // 4) for s in array.shape))
            windows_da, t_graph = _time_call(
                sliding_window_da, x_da, window_size, window_step, axis=axis
            )
            start = time.perf_counter()
            computed = windows_da.compute()
            t_compute = time.perf_counter() - start
            assert np.allclose(computed, ref)
            timings.append(("v2.Dask graph", t_graph))
            timings.append(("v2.Dask compute", t_compute))
        else:
            timings.append(("v2.Dask graph", float("nan")))

        dims = [f"dim_{i}" for i in range(array.ndim)]
        xr_input = xr.DataArray(array, dims=dims)
        xr_view, t_xr = _time_call(
            sliding_window_xr,
            xr_input,
            window_size,
            window_step,
            dim=dims[axis],
            window_dim="window",
            sample_dim="sample",
        )
        assert np.allclose(xr_view.data, ref)
        timings.append(("v3.xarray", t_xr))

        _print_case(name, timings)


if __name__ == "__main__":
    compare_all_variants()
