"""
Module: xarr
Status: REVIEW
Last Updated: 2025-08-26
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:
    This module provides utility functions for working with xarray objects.

Functions:
    spaced_sample_around_coord: Returns indices spaced around a center coordinate.
    drop_attrs: Removes attributes from an xarray DataArray except specified ones.
    save_xarray: Saves an xarray DataArray to disk with attributes in a JSON file.
    load_xarray: Loads an xarray DataArray from disk and restores attributes.
    reshape_dimension: Reshapes a dimension of an xarray DataArray into new dimensions.
    unstack: Unstacks a multi-indexed dimension in an xarray DataArray.
    coords_from_multitindex: Extracts coordinates from a multi-indexed xarray dimension.
    axis_dim_from_xarr: Maps between axis numbers and dimension names in xarray.
    roll_dim: Rolls the dimensions of an xarray DataArray.
    inject_fs: Decorator to inject the 'fs' attribute from xarray DataArray into a function.

Example:
"""

import xarray as xr
import numpy as np
import pandas as pd
from .reshape import reshape_axes
from functools import wraps

def dur_slice(xsig, dim, fraction, duration):
    """
    Returns a slice of the time series based on fraction and duration.

    Parameters
    ----------
    xsig : xr.DataArray
        Input DataArray with a time dimension.
    dim : str
        Name of the time dimension.
    fraction : float
        Fraction between 0 and 1 indicating where to center the slice in the time series.
    duration : float
        Duration of the time window in the same units as the time dimension.

    Returns
    -------
    slice
        Slice object with start and end time for indexing.
    """
    time_vec = xsig[dim].values
    return dim_dur_slice(time_vec, fraction, duration)

def dim_dur_slice(time_vec, time_fraction, duration):
    """
    Return a slice of the time series based on fraction and duration.

    Parameters
    ----------
    time_vec : array-like
        1D array of time values (must be sorted).
    time_fraction : float
        Fraction between 0 and 1 indicating where to center the slice in the time series.
    duration : float
        Duration of the time window in the same units as time_vec.

    Returns
    -------
    slice
        Slice object with start and end time for indexing.
    """

    # Ensure sorted time vector
    time_vec = np.array(time_vec)
    if not np.all(np.diff(time_vec) >= 0):
        raise ValueError("time_vec must be sorted in ascending order.")

    t_min, t_max = time_vec[0], time_vec[-1]
    total_span = t_max - t_min

    if duration > total_span:
        raise ValueError("Duration exceeds total time span.")

    # Find center time
    center_time = t_min + time_fraction * total_span

    # Calculate start and end times
    half_dur = duration / 2
    start_time = center_time - half_dur
    end_time = center_time + half_dur

    # Adjust bounds explicitly to ensure duration is as requested
    if start_time < t_min:
        start_time = t_min
        end_time = start_time + duration
        if end_time > t_max:
            end_time = t_max
            start_time = end_time - duration
    elif end_time > t_max:
        end_time = t_max
        start_time = end_time - duration
        if start_time < t_min:
            start_time = t_min
            end_time = start_time + duration
    return slice(start_time, end_time)

def _get_index(xsig: xr.DataArray, dim: str) -> pd.Index:
    """
    Returns the index object for a given dimension in an xarray DataArray.

    Parameters
	----------
	xsig : xr.DataArray
		The input data array.
    dim : str
		The name of the dimension to sample around.

    Returns
    -------
    coord_index : pandas.Index
		The index object from arr.get_index(dim).
    """
    return xsig.get_index(dim)

def spaced_sample_around_coord(xsig: xr.DataArray, dim: str, center, nsample: int, step=1, clip=True):
    """
    Returns indices spaced around a center coordinate.
    Example:
    A data array with dim='time'
    t_coord = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t_sample_around5 = spaced_sample_around_coord(t_coord, center=5, nsample=3, step=2)
    Output: [3, 5, 7]

	Parameters
	----------
	xsig : xr.DataArray
		The input data array.
    dim : str
		The name of the dimension to sample around.
	center : float or int
		The coordinate value to center the sampling.
	nsample : int
		number of samples
	step : int, optional
		Step size between samples.
	clip : bool, optional
		If True, clips indices to valid range.

	Returns
	-------
	np.ndarray
		Array of sampled indices.
    """
    coord_index = _get_index(xsig, dim)
    it = coord_index.get_indexer([center], method="nearest")[0]
    tsamples = slice_around(it, nsample, step)
    if clip:
        tsamples = tsamples[(tsamples >= 0) & (tsamples < len(coord_index))]
    return tsamples

def xdim_subsample_around(xsig: xr.DataArray, dim: str, center, nsample: int, step=1, clip=True):
    """
    Returns indices spaced around a center coordinate.
    Example:
    A data array with dim='time'
    t_coord = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t_sample_around5 = spaced_sample_around_coord(t_coord, center=5, nsample=3, step=2)
    Output: [3, 5, 7]

	Parameters
	----------
	xsig : xr.DataArray
		The input data array.
    dim : str
		The name of the dimension to sample around.
	center : float or int
		The coordinate value to center the sampling.
	nsample : int
		number of samples
	step : int, optional
		Step size between samples.
	clip : bool, optional
		If True, clips indices to valid range.

	Returns
	-------
	np.ndarray
		Array of sampled indices.
    """
    coord_index = _get_index(xsig, dim)
    it = coord_index.get_indexer([center], method="nearest")[0]
    dim_slice = slice_around(it, nsample, step)
    return xsig.coords[dim].isel({dim: dim_slice})

def slice_around(it, nsample, step=1):
    half_n = nsample // 2
    start = it - half_n * step
    stop = start + nsample * step
    return slice(start, stop, step)

def xarr_wrap(func_):
    """
    Decorator to wrap a function that takes numpy arrays as input and output
    and applies it to xarray.DataArray objects, retaining the original metadata.
    
    Parameters
    ----------
    func_: function that transforms numpy arrays along a specific axis.
    """
    @wraps(func_)
    def wrapper(x, *args, **kwargs):
        if isinstance(x, xr.DataArray):
            dim = kwargs.pop('dim', None)  # Get the dimension name if provided
            if dim:
                # Apply the function along the specified dimension
                out_data = xr.apply_ufunc(
                    func_,                   # Function to apply
                    x,                       # Input DataArray
                    input_core_dims=[[dim]],  # Dimension along which to apply
                    kwargs=kwargs,            # Additional arguments for the function
                    dask='parallelized',      # Enable Dask support for parallel computation
                    output_dtypes=[x.dtype]   # Output dtype specification
                )
            else:
                # Apply the function to the whole array if no dimension is specified
                out_data = xr.apply_ufunc(
                    func_,
                    x,
                    kwargs=kwargs,
                    dask='parallelized',
                    output_dtypes=[x.dtype]
                )

            # Return a DataArray, retaining the original metadata
            return xr.DataArray(out_data, dims=x.dims, coords=x.coords, attrs=x.attrs)
        else:
            # If the input is not an xarray.DataArray, apply the function directly
            return func_(x, *args, **kwargs)
    
    return wrapper

def reshape_dimension(dataarray, dim, new_shape, new_dims, new_coords=None):
    """
    A much quicker version of unstacking a DataArray than the built-in unstack method.
    Reshapes a specified dimension of an Xarray DataArray into a desired shape,
    replacing the dimension with new dimensions names at the correct position,
    using Xarray's capabilities to transpose and reshape.
    
    Parameters
    ----------
    dataarray : xr.DataArray
        The DataArray to reshape.

    dim : str
        The name of the dimension to reshape.

    new_shape : tuple
        The new shape of the dimension.

    new_dims : tuple
        The new dimension names to replace the reshaped dimension.

    new_coords : dict, optional
        New coordinates for the reshaped dimensions. If not provided, the coordinates
        will be replaced with a range of integers for each dimension.
    
    Returns
    -------
    reshaped_dataarray : xr.DataArray
        The reshaped DataArray.
    """
    if dim not in dataarray.dims:
        raise ValueError(f"Dimension '{dim}' not found in the DataArray.")
    
    if len(new_shape) != len(new_dims):
        raise ValueError("The length of new_shape must match the length of new_dims.")
    
    original_size = dataarray.sizes[dim]
    reshaped_size = np.prod(new_shape)
    if original_size != reshaped_size:
        raise ValueError("The product of new_shape does not match the size of the dimension to be reshaped.")
    
    dim_index = dataarray.dims.index(dim)
    new_data = reshape_axes(dataarray.data, dim_index, new_shape)
    new_dims_list = dataarray.dims[:dim_index] + new_dims + dataarray.dims[dim_index+1:]

    # Prepare new coordinates, adjusting for the new dimensions
    coords_ = {k: v for k, v in dataarray.coords.items() if k != dim}
    if new_coords is None:
        for i, new_dim in enumerate(new_dims):
            coords_[new_dim] = np.arange(new_shape[i])
    else:
        for new_dim, new_coord in new_coords.items():
            coords_[new_dim] = new_coord

    # Create the reshaped DataArray with the corrected dimensionality
    reshaped_dataarray = xr.DataArray(new_data, dims=new_dims_list, coords=coords_)
    
    # copy attributes
    reshaped_dataarray.attrs = dataarray.attrs
    return reshaped_dataarray

def unstack(dataarray, dim):
    new_coords = coords_from_multitindex(dataarray[dim])
    new_shape = tuple(len(new_coords[name]) for name in new_coords)
    new_dims = tuple(new_coords.keys())
    return reshape_dimension(dataarray, dim, new_shape, new_dims, new_coords)

def coords_from_multitindex(xarr_multiindex):
    coo_names = [c for c in xarr_multiindex.coords][1:]
    coo_df = pd.MultiIndex.from_tuples(xarr_multiindex.values, names=coo_names)
    coo_dict = {name: level.tolist() for name, level in zip(coo_df.names, coo_df.levels)}
    return coo_dict

def axis_dim_from_xarr(x, axis=-1, dim=None):
    """
    Handle axis and dim arguments for xarray DataArray.

    Parameters
    ----------
    x : xarray.DataArray
        Input DataArray.
    axis : int, optional
        Axis along which to operate (default is -1).
    dim : str, optional
        Name of the dimension along which to operate. If specified, this
        will override the axis argument.

    Returns
    -------
    axis : int
        The axis corresponding to the specified or default dimension.
    dim : str
        The dimension name corresponding to the axis.
    """
    if dim is not None:
        # Get the axis number for the given dimension
        axis = x.get_axis_num(dim)
    else:
        # Get the dimension name for the given axis
        dim = x.dims[axis]
    
    return axis, dim

def roll_dim(x, nroll):
    """
    Rolls the dimensions of an xarray.DataArray by nroll positions.

    Parameters
    ----------
    x : xr.DataArray
        Input DataArray.
    nroll : int
        Number of positions to roll the dimensions.

    Returns
    -------
    xr.DataArray
        DataArray with rolled dimensions.
    """
    rolled_dims = np.roll(list(x.dims), nroll)
    return x.transpose(*rolled_dims)

