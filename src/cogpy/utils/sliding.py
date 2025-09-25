"""
Module: sliding
Status: WIP
Last Updated: 2025-08-26
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:
	Sliding window utilities

Functions:
	rolling_win: Applies a rolling window to an xarray DataArray.

Classes:

Constants:

Example:
"""

import numpy as np
import xarray as xr
from param import Callable
from typing import Optional, Dict, Any, Sequence

def rolling_win(
	xsig: xr.DataArray,
	window_size: int,
	window_step: int,
	dim: str = "time",
	window_dim: str = "window",
	min_periods: int = None,
):
	assert dim in xsig.dims, f"Dimension '{dim}' not found in input xarray."
	assert dim in xsig.coords, f"Dimension '{dim}' does not have coordinates in input xarray."
	assert window_dim not in xsig.dims, f"Dimension '{window_dim}' already exists in input xarray."

	# 1) rolling + construct with stride
	xroll = xsig.rolling({dim: window_size}, center=True, min_periods=min_periods)
	xwin = xroll.construct(window_dim, stride=window_step)  # <-- strided centers
	wstart_idx, wend_idx = roll_win_window_start_end(xsig.sizes[dim], window_size, window_step)
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
	*,
	measure_input_core_dims: Sequence[str] = (),
	measure_output_core_dims: Sequence[str] = (),
	measure_output_sizes: Dict[str, int] = None,
	run_dim: str = "time",
	window_dim: str = "window",
	output_dtype: Optional[Any] = None,
	name: Optional[str] = None,
):
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
			raise ValueError("Signal does not have an attribute 'fs'. You may provide it explicitly via fs=<value> or set it as an attribute on the input xarray.")

	# Normalize input: ensure DataArray, `run_dim` exists, and has a time coord
	if not isinstance(xsig, xr.DataArray):
		xsig = xr.DataArray(xsig, dims=[run_dim])
	if run_dim not in xsig.dims:
		raise ValueError(f"`run_dim='{run_dim}'` not found in xsig.dims={xsig.dims}")

	if run_dim not in xsig.coords:
		xsig = xsig.assign_coords({run_dim: np.arange(xsig.sizes[run_dim]) / fs})

	assert window_dim in measure_input_core_dims[0], f"Expected '{window_dim}' in measure_input_core_dims={measure_input_core_dims}"

	# Rolling windows (your xut.rolling_win should append `window_dim`)
	x_roll = rolling_win(
		xsig, window_size=window_size, window_step=window_step, dim=run_dim, window_dim=window_dim
	)

	# Decide if we should use the Dask path
	is_dask = _is_dask_array(x_roll)

	# Prepare a single kwargs dict to pass to xr.apply_ufunc
	if output_dtype is None:
		output_dtype = np.result_type(np.asarray(xsig.data).dtype, np.float64)

	# --- Chunking strategy ---
	core_dims = tuple(measure_input_core_dims[0])  # e.g. (row_dim, col_dim, 'window')
	# core dims: one chunk each
	core_chunk_dict = {d: -1 for d in core_dims if d in x_roll.dims}
	# mapped dims: everything else present in x_roll
	mapped_dims = tuple(d for d in x_roll.dims if d not in core_dims)
	# choose a reasonable chunk along mapped dims (tune this!)
	mapped_chunk_size = 1024  # number of processed windows in memory, adjust to memory
	mapped_chunk_dict = {d: mapped_chunk_size for d in mapped_dims}

    # apply chunking to windows
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
		assert measure_output_sizes is not None, "Dask requires measure_output_sizes, specify as Dict[str, int]"
		apply_kwargs.update(dask="parallelized", dask_gufunc_kwargs=dict(output_sizes=measure_output_sizes))

	# Single call site
	out = xr.apply_ufunc(measure, x_roll, **apply_kwargs)

    # keep output chunking consistent (core dims unchunked, mapped dims chunked)
	if _is_dask_array(out):
		# flatten the declared output core dims (single-output gufunc assumed)
		out_core_dims = tuple(measure_output_core_dims[0]) if measure_output_core_dims else ()
		out_core_dims = tuple(d for d in out_core_dims if d in out.dims)  # keep only present dims

		# everything in the output that's not a core dim is a "mapped" dim
		out_mapped_dims = tuple(d for d in out.dims if d not in out_core_dims)

		# policy: core dims = one chunk, mapped dims = modest chunks
		mapped_chunk_size = 1024  # tune based on your memory/dtype
		out_core_chunk_dict = {d: -1 for d in out_core_dims}
		out_mapped_chunk_dict = {d: mapped_chunk_size for d in out_mapped_dims}

		out = out.chunk({**out_core_chunk_dict, **out_mapped_chunk_dict})

	# Name and coordinates for window positions (center time of each window)
	if name is None:
		base = (xsig.name or "sig")
		name = f"{base}_{getattr(measure, '__name__', 'measure')}"
	out = out.rename(name)
	return out
