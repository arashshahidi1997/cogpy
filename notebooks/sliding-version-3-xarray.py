"""Version 3: sliding window utilities with NumPy/Dask core and a thin xarray wrapper."""

from __future__ import annotations

from typing import Tuple

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


def sliding_window_xr(
    xsig: xr.DataArray,
    window_size: int,
    window_step: int = 1,
    dim: str = "time",
    window_dim: str = "window",
    sample_dim: str = "window_sample",
) -> xr.DataArray:
    """Construct sliding windows while preserving xarray dims/coords via a thin wrapper."""

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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xsig = xr.DataArray(
        rng.normal(size=(4, 128)),
        dims=("channel", "time"),
        coords={"channel": np.arange(4), "time": np.linspace(0.0, 1.0, 128)},
        attrs={"units": "mV"},
    )

    window_size = 16
    window_step = 4
    xr_windows = sliding_window_xr(
        xsig, window_size, window_step, dim="time", window_dim="window", sample_dim="sample"
    )

    print(
        f"xarray input dims={xsig.dims}, shape={xsig.shape}; output dims={xr_windows.dims}, shape={xr_windows.shape}"
    )

    axis = xsig.get_axis_num("time")
    np_windows = sliding_window_da(xsig.data, window_size, window_step, axis=axis)
    assert np.allclose(xr_windows.data, np_windows)
    print("sliding_window_xr matches sliding_window_da on raw data")
