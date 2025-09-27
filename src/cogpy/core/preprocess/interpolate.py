"""
Module: interpolate.py
Status: REVIEW
Last Updated: 2025-09-13
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:
	Provides functions for interpolating and extrapolating bad or missing data in 2D and 3D arrays, particularly useful for ECoG data processing.

Functions:
	interpolate_bads: Interpolates and optionally extrapolates bad channels in a 3D array based on a 2D mask.
	nan_mask: Creates a mask indicating the positions of NaN values in a 3D array.
	extrapolate_bads: Extrapolates to fill in NaN values based on the spatial neighborhood of each bad channel in a grid.
	_griddata: Helper function to perform grid data interpolation on 2D arrays.

"""
import numpy as np
from scipy.interpolate import griddata
from ..utils.grid_neighborhood import build_neighbor_masks, make_footprint

def _griddata(_arr, nans, method='linear', **kwargs):
	"""
	Interpolate values for positions marked as NaN in the input array.

	This function uses grid data interpolation over 2D arrays to fill in NaN values based on the specified interpolation method.

	Parameters
	----------
	_arr : ndarray
		The input array containing NaN values to interpolate. It should be a 2D array.
	nans : ndarray
		A boolean array of the same shape as `_arr` indicating the positions of NaNs in `_arr` to be interpolated.
	method : str, optional
		The method of interpolation (e.g., 'linear', 'nearest'). Default is 'linear'.
	**kwargs :
		Additional keyword arguments to be passed to `scipy.interpolate.griddata`.

	Returns
	-------
	ndarray
		A copy of `_arr` with NaN values interpolated based on the specified method.

	Notes
	-----
	This function does not perform extrapolation. NaN values that cannot be interpolated based on surrounding data will remain NaN.
	"""
	not_nans = np.invert(nans)
	x, y = np.where(not_nans) # reference coordinates
	z = _arr[x, y] # reference values
	ix, iy = np.where(nans) # bad channel coordinates
	iz = griddata((x,y), z, (ix, iy), method=method, **kwargs)
	_iarr = _arr.copy()
	_iarr[ix, iy] = iz
	return _iarr

def extrapolate_bads(x, footprint=None, median=False, gridshape=(16,16)):
	"""
	Extrapolates to fill in NaN values based on the spatial neighborhood of each bad channel in a grid.

	Parameters
	----------
	x : ndarray
		The input array with dimensions (grid, grid, [time]). NaN values indicate bad or missing data to be filled.
	median : bool, optional
		If True, uses the median of the spatial neighborhood for extrapolation instead of the mean. Default is False.
	gridshape : tuple, optional
		The shape of the grid used for extrapolation. Default is (16, 16).

	Returns
	-------
	ndarray
		The input array with NaN values extrapolated based on the specified method.

	Notes
	-----
	Extrapolation is performed by calculating the mean or median of the spatial neighborhood of each bad channel.
	The function requires a predefined `exclude_neighbor_mask_per_node` mapping or similar functionality to determine neighborhoods.
	"""
	x_ = np.copy(x)
	# find nans
	isnan = np.where(np.isnan(x_[:,:,0])) 

	if isnan[0].size == 0:
		return x_

	if footprint is None:
		footprint = make_footprint(rank=2, connectivity=1, niter=2)

	grid_shape = x.shape[:2]
	exclude_mask_per_node = build_neighbor_masks(footprint, grid_shape, exclude=True)
	# find bad channel indices where nans occur
	bad_chan = np.ravel_multi_index(isnan, gridshape) 
	# replace nan at each bad channel with the mean of spatial neighborhood of the channel
	statistic = np.nanmedian if median else np.nanmean 
	x_[isnan] = np.array([statistic(x_[np.where(exclude_mask_per_node[ich])], axis=0) for ich in bad_chan]) 
	return x_

def interpolate_bads(arr, skip, method='linear', extrapolate=True, neighbor_mask=None, **kwargs):
	"""
	Interpolates and optionally extrapolates to fill in bad or missing data in a 3D array based on a 2D mask of bad channels.

	Parameters
	----------
	arr : ndarray
		A 3D array containing values (x, y, t), where bad or missing data needs to be interpolated.
	skip : ndarray
		A 2D boolean array where True indicates bad channels in `arr` to be skipped (or interpolated).
	method : str, optional
		The interpolation method to use. Default is 'linear'.
	extrapolate : bool, optional
		If True, performs extrapolation on the interpolated array to fill in any remaining NaNs. Default is True.
	neighbor_mask : ndarray, optional
		A 2D boolean array indicating the neighbors to consider for extrapolation.
	**kwargs :
		Additional keyword arguments to be passed to the interpolation function.

	Returns
	-------
	ndarray
		A 2D array with interpolated (and possibly extrapolated) values at the positions of bad channels.

	Notes
	-----
	The function relies on `_griddata` for interpolation and `extrapolate_bads` for optional extrapolation.
	"""
	iarr_masked = bad2nan(arr, skip)

	# interpolate
	iarr = _griddata(iarr_masked, skip, method=method)

	# extrapolate
	if extrapolate:
		iarr = extrapolate_bads(iarr, gridshape=skip.shape)

	return iarr

def bad2nan(arr, bad):
	"""
	Converts bad channel indicators in a 2D boolean array to NaN values in a corresponding 3D data array.

	Parameters
	----------
	arr : ndarray
		A 3D array with dimensions (x, y, t) containing data values.
	bad : ndarray
		A 2D boolean array where True indicates bad channels in `arr` that should be set to NaN.

	Returns
	-------
	ndarray
		A copy of `arr` with values at positions indicated by `bad` set to NaN.

	Notes
	-----
	This function is useful for preparing data for interpolation by marking bad channels as NaN.
	"""
	assert arr.ndim == 3, "Input array must be 3D"
	assert bad.ndim == 2, "Bad channel mask must be 2D"
	assert arr.shape[:2] == bad.shape, "Spatial dimensions of arr and bad must match"

	iarr = arr.copy()
	iarr[bad, :] = np.nan
	return iarr

def infer_nan_mask(arr):
	"""
	Creates a mask indicating the positions of NaN values in the input array.

	Parameters
	----------
	arr : ndarray
		A 3D input array to check for NaN values.

	Returns
	-------
	ndarray
		A 2D boolean array where True indicates the presence of NaN in any position along the last dimension of `arr`.

	Raises
	------
	AssertionError
		If the mask for absolute NaN presence does not match the mask for occasional NaN presence.

	Notes
	-----
	This function differentiates between absolute and occasional NaN presence across the third dimension of `arr`.
	"""
	isnan = np.array(np.isnan(arr), dtype=bool)
	abs_bad_mask = np.all(isnan, axis=-1)
	occasional_bad_mask = np.any(isnan, axis=-1)
	assert np.all(abs_bad_mask == occasional_bad_mask), print(occasional_bad_mask)
	return abs_bad_mask
