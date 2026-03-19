"""Sliding window operations

Status
------
WIP 

Metadata
--------
Author : Arash Shahidi <A.Shahidi@campus.lmu.de>
Last Updated : 2025-08-26

Utitlies for sliding window operations on xarray DataArrays and dask Arrays as well as NumPy arrays.

Examples
--------
>>> from cogpy.utils.sliding import rolling_win
"""

import numpy as np
import xarray as xr
import math
from typing import Callable, Optional, Dict, Any, Sequence


def rolling_win(
    xsig: xr.DataArray,
    window_size: int,
    window_step: int,
    dim: str = "time",
    window_dim: str = "window",
    min_periods: int = None,
):
    """
    rolling window for xarray along a given dimension.

    Parameters
    ----------
    xsig : xr.DataArray
            Input xarray DataArray.
    window_size : int
            Size of the rolling window.
    window_step : int
            Step size between windows.
    dim : str
            Dimension along which to apply the rolling window. Default is "time".
    window_dim : str
            Name of the new dimension created for the rolling windows. Default is "window".
    min_periods : int
            Minimum number of observations in window required to have a value; otherwise, the result is NaN.
            Default is None, which means the window size.

    Returns
    -------
    xr.DataArray
            DataArray with an added dimension for the rolling windows.
    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from cogpy.utils.sliding import rolling_win
    >>> time = np.arange(1000)
    >>> signal = np.sin(2 * np.pi * 0.01 * time)
    >>> xsig = xr.DataArray(signal, dims=['time'], coords={'time': time})
    >>> xwin = rolling_win(xsig, window_size=100, window_step=50, dim='time', window_dim='window')
    """
    assert dim in xsig.dims, f"Dimension '{dim}' not found in input xarray."
    assert (
        dim in xsig.coords
    ), f"Dimension '{dim}' does not have coordinates in input xarray."
    assert (
        window_dim not in xsig.dims
    ), f"Dimension '{window_dim}' already exists in input xarray."

    # 1) rolling + construct with stride
    xroll = xsig.rolling({dim: window_size}, center=True, min_periods=min_periods)
    xwin = xroll.construct(window_dim, stride=window_step)  # <-- strided centers
    wstart_idx, wend_idx = roll_win_window_start_end(
        xsig.sizes[dim], window_size, window_step
    )
    return xwin.isel({dim: slice(wstart_idx, wend_idx)})


def roll_win_window_start_end(N, window_size, window_step):
    nstart = (window_size // 2 - 1) // window_step + 1
    nend = (N - 1 - (window_size - 1) // 2) // window_step + 1
    return nstart, nend


def _is_dask_array(a) -> bool:
    try:
        import dask.array as da

        return isinstance(getattr(a, "data", a), da.Array)
    except Exception:
        return hasattr(getattr(a, "data", a), "chunks")


def running_measure(
    measure: Callable,
    xsig: xr.DataArray,
    fs: float = None,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    slider_kwargs: Optional[Dict[str, Any]] = None,
    measure_input_core_dims: Sequence[str] = (),
    measure_output_core_dims: Sequence[str] = (),
    measure_output_sizes: Dict[str, int] = None,
    run_dim: str = "time",
    window_dim: str = "window",
    output_dtype: Optional[Any] = None,
    name: Optional[str] = None,
) -> xr.DataArray:
    """
    Apply `measure` over rolling windows along `run_dim`, preserving Dask laziness.

    Parameters
    ----------
    measure : callable
            Function applied per window. Signature should accept a 1D window along `window_dim`
            plus any kwargs in `measure_kwargs`. It must return something with the same length
            as the input window if you keep `output_core_dims=[[window_dim]]` (default below).
            If it reduces the window to a scalar, change `output_core_dims=[[]]`.
    xsig : xr.DataArray
            Signal (NumPy- or Dask-backed works).
    fs : float
            Sampling rate (Hz). Used for time coords if missing and for window time labels.
    measure_kwargs : dict
            Extra kwargs forwarded to `measure`.
    slider_kwargs : dict
            Should include: {"window_size": int, "window_step": int}. Extras ignored.
    measure_input_core_dims : sequence of str
            Any additional core dims that `measure` expects *besides* `window_dim`.
    measure_output_core_dims : sequence of str
            Core dims that `measure` returns (default: same as `window_dim`).
    measure_output_sizes : dict
            Sizes of the output core dims (required in case xsig is passed as Dask.DataArray).
    run_dim : str
            Dimension along which rolling windows are taken (default "time").
    window_dim : str
            Name of the created window core dimension (default "window").
    output_dtype : dtype or None
            Output dtype hint for xarray/dask. If None, inferred from input dtype.
    name : str or None
            Name for the output DataArray.

    Returns
    -------
    xr.DataArray
            Result with dims: (other dims, window_dim, [plus any dims `measure` returns])
    """
    measure_kwargs = {} if measure_kwargs is None else measure_kwargs
    slider_kwargs = {} if slider_kwargs is None else slider_kwargs
    window_size = int(slider_kwargs.get("window_size", 256))
    window_step = int(slider_kwargs.get("window_step", 64))

    if fs is None:
        try:
            fs = xsig.fs
        except AttributeError:
            raise ValueError(
                "Signal does not have an attribute 'fs'. You may provide it explicitly via fs=<value> or set it as an attribute on the input xarray."
            )

    # Normalize input: ensure DataArray, `run_dim` exists, and has a time coord
    if not isinstance(xsig, xr.DataArray):
        xsig = xr.DataArray(xsig, dims=[run_dim])
    if run_dim not in xsig.dims:
        raise ValueError(f"`run_dim='{run_dim}'` not found in xsig.dims={xsig.dims}")

    if run_dim not in xsig.coords:
        xsig = xsig.assign_coords({run_dim: np.arange(xsig.sizes[run_dim]) / fs})

    assert (
        window_dim in measure_input_core_dims[0]
    ), f"Expected '{window_dim}' in measure_input_core_dims={measure_input_core_dims}"

    # Rolling windows (your xut.rolling_win should append `window_dim`)
    x_roll = rolling_win(
        xsig,
        window_size=window_size,
        window_step=window_step,
        dim=run_dim,
        window_dim=window_dim,
    )

    # Decide if we should use the Dask path
    is_dask = _is_dask_array(x_roll)

    # Prepare a single kwargs dict to pass to xr.apply_ufunc
    if output_dtype is None:
        output_dtype = xsig.dtype if output_dtype is None else output_dtype

    # --- Chunking strategy ---
    core_dims = tuple(measure_input_core_dims[0])  # e.g. (row_dim, col_dim, 'window')
    # core dims: one chunk each
    core_chunk_dict = {d: -1 for d in core_dims if d in x_roll.dims}
    # mapped dims: everything else present in x_roll
    mapped_dims = tuple(d for d in x_roll.dims if d not in core_dims)
    # choose a reasonable chunk along mapped dims (tune this!)
    mapped_chunk_size = 32  # number of processed windows in memory, adjust to memory
    mapped_chunk_dict = {d: mapped_chunk_size for d in mapped_dims}

    # apply chunking to windows
    if is_dask:
        x_roll = x_roll.chunk({**core_chunk_dict, **mapped_chunk_dict})

    out = xroll_apply(
        measure,
        x_roll,
        measure_kwargs,
        measure_input_core_dims=measure_input_core_dims,
        measure_output_core_dims=measure_output_core_dims,
        measure_output_sizes=measure_output_sizes,
        output_dtype=output_dtype,
        name=name,
        dask="parallelized" if is_dask else None,
    )
    return out


def xroll_apply(
    measure: Callable,
    x_roll: xr.DataArray,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    measure_input_core_dims: Sequence[str] = (),
    measure_output_core_dims: Sequence[str] = (),
    measure_output_sizes: Dict[str, int] = None,
    output_dtype: Optional[Any] = None,
    name: Optional[str] = None,
    dask=None,
):
    apply_kwargs = dict(
        input_core_dims=measure_input_core_dims,
        output_core_dims=measure_output_core_dims,
        vectorize=True,
        output_dtypes=[output_dtype],
        kwargs=measure_kwargs,
    )

    if dask == "parallelized":
        assert (
            measure_output_sizes is not None
        ), "Dask requires measure_output_sizes, specify as Dict[str, int]"
        apply_kwargs.update(
            dask=dask, dask_gufunc_kwargs=dict(output_sizes=measure_output_sizes)
        )

    # Single call site
    out = xr.apply_ufunc(measure, x_roll, **apply_kwargs)

    # keep output chunking consistent (core dims unchunked, mapped dims chunked)
    if _is_dask_array(out):
        # flatten the declared output core dims (single-output gufunc assumed)
        out_core_dims = (
            tuple(measure_output_core_dims[0]) if measure_output_core_dims else ()
        )
        out_core_dims = tuple(
            d for d in out_core_dims if d in out.dims
        )  # keep only present dims

        # everything in the output that's not a core dim is a "mapped" dim
        out_mapped_dims = tuple(d for d in out.dims if d not in out_core_dims)

        # policy: core dims = one chunk, mapped dims = modest chunks
        mapped_chunk_size = 1024  # tune based on your memory/dtype
        out_core_chunk_dict = {d: -1 for d in out_core_dims}
        out_mapped_chunk_dict = {d: mapped_chunk_size for d in out_mapped_dims}

        out = out.chunk({**out_core_chunk_dict, **out_mapped_chunk_dict})

    # Name and coordinates for window positions (center time of each window)
    if name is None:
        name = f"sig_{getattr(measure, '__name__', 'measure')}"
    out = out.rename(name)
    return out


def running_measure_sane(
    measure: Callable,
    xsig: xr.DataArray,
    fs: float = None,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    slider_kwargs: Optional[Dict[str, Any]] = None,
    *,
    measure_input_core_dims: Sequence[Sequence[str]] = (),
    measure_output_core_dims: Sequence[Sequence[str]] = (),
    measure_output_sizes: Optional[Dict[str, int]] = None,
    run_dim: str = "time",
    window_dim: str = "window",
    output_dtype: Optional[Any] = None,
    name: Optional[str] = None,
    # sane-chunking knobs:
    min_windows_per_chunk: int = 4,
    min_samples_floor: int = 65536,
    max_chunk_samples: Optional[int] = None,
    other_dim_chunks: Optional[Dict[str, int]] = None,
):
    """
    Apply `measure` over strided, centered rolling windows along `run_dim`,
    using window-aware rechunking to keep memory sane.

    Returns
    -------
    xr.DataArray
        Result with dims: (mapped dims..., [dims returned by measure])
    """
    measure_kwargs = {} if measure_kwargs is None else measure_kwargs
    slider_kwargs = {} if slider_kwargs is None else slider_kwargs
    window_size = int(slider_kwargs.get("window_size", 256))
    window_step = int(slider_kwargs.get("window_step", 64))

    # fs handling
    if fs is None:
        fs = getattr(xsig, "fs", None)
        if fs is None:
            raise ValueError("Provide fs=<sampling_rate> or set xsig.fs")

    # normalize input
    if not isinstance(xsig, xr.DataArray):
        xsig = xr.DataArray(xsig, dims=[run_dim])
    if run_dim not in xsig.dims:
        raise ValueError(f"`run_dim='{run_dim}'` not in xsig.dims={tuple(xsig.dims)}")
    if run_dim not in xsig.coords:
        xsig = xsig.assign_coords({run_dim: np.arange(xsig.sizes[run_dim]) / fs})

    # validate core-dims signature
    if not (
        isinstance(measure_input_core_dims, (list, tuple))
        and len(measure_input_core_dims) >= 1
        and isinstance(measure_input_core_dims[0], (list, tuple))
    ):
        raise ValueError("measure_input_core_dims must be like [[..., 'window'], ...]")
    if window_dim not in measure_input_core_dims[0]:
        raise ValueError(
            f"'{window_dim}' must appear in measure_input_core_dims[0]={measure_input_core_dims[0]}"
        )

    # window-aware rechunk + construct (lazy)
    xsig = rechunk_for_rolling(
        xsig,
        dim=run_dim,
        window_size=window_size,
        window_step=window_step,
        min_windows_per_chunk=min_windows_per_chunk,
        min_samples_floor=min_samples_floor,
        max_chunk_samples=max_chunk_samples,
        other_dims=other_dim_chunks,
    )
    x_roll = rolling_win_sane(
        xsig,
        window_size=window_size,
        window_step=window_step,
        dim=run_dim,
        window_dim=window_dim,
        min_periods=slider_kwargs.get("min_periods", None),
        min_windows_per_chunk=min_windows_per_chunk,
        min_samples_floor=min_samples_floor,
        max_chunk_samples=max_chunk_samples,
        other_dim_chunks=other_dim_chunks,
    )

    # dask path?
    is_dask = hasattr(x_roll.data, "chunks")

    # dtype hint without computing
    if output_dtype is None:
        try:
            output_dtype = np.result_type(xsig.dtype, np.float32)
        except Exception:
            output_dtype = np.float32

    # chunking policy: core dims (including window) = single chunk; mapped dims = modest chunks
    core_dims_tuple = tuple(measure_input_core_dims[0])
    core_chunk_dict = {d: -1 for d in core_dims_tuple if d in x_roll.dims}
    mapped_dims = tuple(d for d in x_roll.dims if d not in core_dims_tuple)
    mapped_chunk_size = int(slider_kwargs.get("mapped_chunk_size", 1024))
    mapped_chunk_dict = {d: mapped_chunk_size for d in mapped_dims}

    if is_dask:
        x_roll = x_roll.chunk({**core_chunk_dict, **mapped_chunk_dict})

    apply_kwargs = dict(
        input_core_dims=measure_input_core_dims,
        output_core_dims=measure_output_core_dims,
        vectorize=True,
        output_dtypes=[output_dtype],
        kwargs=measure_kwargs,
    )
    if is_dask:
        if measure_output_sizes is None:
            raise ValueError(
                "With Dask, provide measure_output_sizes for any new/output core dims."
            )
        apply_kwargs.update(
            dask="parallelized",
            dask_gufunc_kwargs=dict(output_sizes=measure_output_sizes),
        )

    out = xr.apply_ufunc(measure, x_roll, **apply_kwargs)

    # keep output chunking consistent
    if hasattr(out.data, "chunks"):
        out_core_dims = (
            tuple(measure_output_core_dims[0]) if measure_output_core_dims else ()
        )
        out_core_dims = tuple(d for d in out_core_dims if d in out.dims)
        out_mapped_dims = tuple(d for d in out.dims if d not in out_core_dims)
        out = out.chunk(
            {
                **{d: -1 for d in out_core_dims},
                **{d: mapped_chunk_size for d in out_mapped_dims},
            }
        )

    # name result
    if name is None:
        base = xsig.name or "sig"
        name = f"{base}_{getattr(measure, '__name__', 'measure')}"
    return out.rename(name)


def compute_window_chunk_len(
    window_size: int,
    window_step: int,
    *,
    min_windows_per_chunk: int = 4,
    min_samples_floor: int = 65536,
    max_chunk_samples: Optional[int] = None,
) -> int:
    """
    Choose a chunk length for the rolling axis that:
      1) is >= min_windows_per_chunk * window_size,
      2) is >= min_samples_floor (helps amortize overhead on tiny windows),
      3) is an exact multiple of window_step,
      4) is clipped to max_chunk_samples if provided (and then rounded up to multiple of step).

    Returns
    -------
    int
        Recommended chunk length along the rolling dimension.
    """
    base = max(min_windows_per_chunk * window_size, min_samples_floor)
    if max_chunk_samples is not None:
        base = min(base, max_chunk_samples)
    # round up to nearest multiple of window_step
    m = int(math.ceil(base / window_step)) * int(window_step)
    # never smaller than the window itself
    return max(m, int(window_size))


def rechunk_for_rolling(
    xsig: xr.DataArray,
    *,
    dim: str = "time",
    window_size: int,
    window_step: int,
    min_windows_per_chunk: int = 4,
    min_samples_floor: int = 65536,
    max_chunk_samples: Optional[int] = None,
    other_dims: Optional[Dict[str, int]] = None,
) -> xr.DataArray:
    """
    Rechunk `xsig` so that chunks along `dim` are large enough for efficient rolling windows.
    Leaves other dims unchanged unless `other_dims` is passed.

    Notes
    -----
    - Purely lazy (no compute).
    - Works for NumPy- or Dask-backed; on NumPy it just returns the same array.

    Returns
    -------
    xr.DataArray
        Rechunked array (or same array if not Dask).
    """
    if dim not in xsig.dims:
        raise ValueError(f"Rolling dim '{dim}' not found; dims={xsig.dims}")

    # If not dask-backed, nothing to do
    if not hasattr(xsig.data, "chunks"):
        return xsig

    time_chunk = compute_window_chunk_len(
        window_size,
        window_step,
        min_windows_per_chunk=min_windows_per_chunk,
        min_samples_floor=min_samples_floor,
        max_chunk_samples=max_chunk_samples,
    )

    chunks = {dim: int(time_chunk)}
    if other_dims:
        chunks.update(other_dims)
    return xsig.chunk(chunks)


def rolling_win_sane(
    xsig: xr.DataArray,
    *,
    window_size: int,
    window_step: int,
    dim: str = "time",
    window_dim: str = "window",
    min_periods: Optional[int] = None,
    min_windows_per_chunk: int = 4,
    min_samples_floor: int = 65536,
    max_chunk_samples: Optional[int] = None,
    other_dim_chunks: Optional[Dict[str, int]] = None,
) -> xr.DataArray:
    """
    Convenience wrapper:
      1) rechunk `xsig` with window-aware chunks,
      2) build strided rolling windows (centered),
      3) trim edge centers.

    Returns
    -------
    xr.DataArray
        DataArray with dims: (..., dim (centers), window_dim)
    """
    xsig = rechunk_for_rolling(
        xsig,
        dim=dim,
        window_size=window_size,
        window_step=window_step,
        min_windows_per_chunk=min_windows_per_chunk,
        min_samples_floor=min_samples_floor,
        max_chunk_samples=max_chunk_samples,
        other_dims=other_dim_chunks,
    )

    # construct windows lazily; xarray/dask will handle halo overlap
    xroll = xsig.rolling({dim: int(window_size)}, center=True, min_periods=min_periods)
    xwin = xroll.construct(window_dim, stride=int(window_step))

    # trim to valid centered positions (same math as your helper)
    nstart = (window_size // 2 - 1) // window_step + 1
    nend = (xsig.sizes[dim] - 1 - (window_size - 1) // 2) // window_step + 1
    return xwin.isel({dim: slice(nstart, nend)})
