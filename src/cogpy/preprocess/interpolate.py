"""Spatial interpolation for bad / missing channels.

Two families of functions are provided:

- **Grid-based** (``interpolate_bads``, ``extrapolate_bads``) — assumes a
  regular 2D ``(AP, ML)`` grid. Fast and simple when electrodes sit on a
  uniform lattice.

- **Coordinate-based** (``interpolate_bads_coords``, ``interpolate_bads_xarray``)
  — takes explicit ``(x, y)`` electrode coordinates in physical units. Handles
  non-uniform layouts such as graphene ECoG arrays with hemispheric gaps,
  checkerboard 32×16 patterns, and depth probes. The geometry lives in the
  data (BIDS ``_electrodes.tsv`` coordinates), not in the code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import griddata

from ..utils.grid_neighborhood import build_neighbor_masks, make_footprint

if TYPE_CHECKING:
    import xarray as xr


def _griddata(_arr, nans, method="linear", **kwargs):
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
    x, y = np.where(not_nans)  # reference coordinates
    z = _arr[x, y]  # reference values
    ix, iy = np.where(nans)  # bad channel coordinates
    iz = griddata((x, y), z, (ix, iy), method=method, **kwargs)
    _iarr = _arr.copy()
    _iarr[ix, iy] = iz
    return _iarr


def extrapolate_bads(x, footprint=None, median=False, gridshape=(16, 16)):
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
    isnan = np.where(np.isnan(x_[:, :, 0]))

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
    x_[isnan] = np.array(
        [
            statistic(x_[np.where(exclude_mask_per_node[ich])], axis=0)
            for ich in bad_chan
        ]
    )
    return x_


def interpolate_bads(
    arr, skip, method="linear", extrapolate=True, neighbor_mask=None, **kwargs
):
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


def interpolate_bads_coords(
    arr: np.ndarray,
    coords: np.ndarray,
    bad: np.ndarray,
    method: str = "linear",
    fill_method: str = "nearest",
) -> np.ndarray:
    """Interpolate bad channels using explicit electrode coordinates.

    Unlike :func:`interpolate_bads`, this function does not assume a regular
    grid. It accepts arbitrary 2D coordinates (``coords``) and interpolates
    bad channel values using Delaunay triangulation of the good channels.
    This is the correct behavior for non-uniform electrode layouts such as:

    - Graphene ECoG grids with an anatomical gap between hemispheres
    - 32×16 checkerboard-interleaved arrays
    - Linear depth probes with non-square aspect ratios
    - Mixed-modality recordings where electrodes have heterogeneous spacing

    Parameters
    ----------
    arr : ndarray, shape (n_channels, *extra_dims)
        Signal array with channels along axis 0. Any number of trailing
        dimensions (time, frequency, etc.) are preserved.
    coords : ndarray, shape (n_channels, 2)
        ``(x, y)`` electrode coordinates. Units are arbitrary but must be
        consistent across channels (e.g. all in mm).
    bad : ndarray, shape (n_channels,)
        Boolean mask — ``True`` where a channel is bad and should be
        interpolated.
    method : {"linear", "nearest", "cubic"}, default "linear"
        Interpolation method passed to :func:`scipy.interpolate.griddata`.
    fill_method : {"nearest", None}, default "nearest"
        Fallback for bad channels outside the convex hull of good channels
        (where ``method="linear"`` returns NaN). If ``"nearest"``, a second
        griddata pass with ``method="nearest"`` fills remaining gaps. If
        ``None``, NaNs are left in place.

    Returns
    -------
    out : ndarray
        Copy of ``arr`` with rows at ``bad`` positions replaced by
        interpolated values.

    Notes
    -----
    ``scipy.interpolate.griddata`` vectorizes over trailing dimensions of
    ``values``, so the full signal is interpolated in a single call per
    method pass — no per-timepoint Python loop.

    See Also
    --------
    interpolate_bads : Grid-based version for uniform 2D ``(AP, ML)`` layouts.
    interpolate_bads_xarray : Wrapper that reads coords from a DataArray.
    """
    arr = np.asarray(arr)
    coords = np.asarray(coords)
    bad = np.asarray(bad, dtype=bool)

    if arr.shape[0] != coords.shape[0] or arr.shape[0] != bad.shape[0]:
        raise ValueError(
            f"arr, coords, and bad must agree on n_channels along axis 0; "
            f"got arr.shape[0]={arr.shape[0]}, coords.shape[0]={coords.shape[0]}, "
            f"bad.shape[0]={bad.shape[0]}"
        )
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n_channels, 2); got {coords.shape}")

    if not bad.any():
        return arr.copy()

    good = ~bad
    out = arr.copy()

    # Primary interpolation — uses Delaunay triangulation of good points.
    # scipy.griddata accepts values of shape (n, *) and vectorizes.
    out[bad] = griddata(
        coords[good],
        arr[good],
        coords[bad],
        method=method,
    )

    # Fill bad channels outside the convex hull (linear returns NaN for those).
    if fill_method is not None:
        bad_nan = bad.copy()
        bad_nan &= np.isnan(out).reshape(out.shape[0], -1).any(axis=1)
        if bad_nan.any():
            out[bad_nan] = griddata(
                coords[good],
                arr[good],
                coords[bad_nan],
                method=fill_method,
            )

    return out


def interpolate_bads_xarray(
    sig: "xr.DataArray",
    bad: np.ndarray,
    x_coord: str = "x",
    y_coord: str = "y",
    ch_dim: str = "ch",
    method: str = "linear",
) -> "xr.DataArray":
    """Interpolate bad channels in an xarray DataArray using stored coordinates.

    Reads ``(x, y)`` electrode coordinates from non-dimension coordinates
    attached to ``ch_dim``, calls :func:`interpolate_bads_coords`, and returns
    a DataArray with the original shape and metadata preserved.

    Parameters
    ----------
    sig : xr.DataArray
        Signal with a channel dimension (``ch_dim``) and ``x``, ``y``
        non-dimension coordinates along that dimension. Typically loaded
        from a BIDS ``_ieeg.lfp`` with matching ``_electrodes.tsv``.
    bad : array-like, shape (n_channels,)
        Boolean mask along ``ch_dim``.
    x_coord, y_coord : str, default "x", "y"
        Names of the x and y coordinate variables along ``ch_dim``.
    ch_dim : str, default "ch"
        Name of the channel dimension.
    method : {"linear", "nearest", "cubic"}, default "linear"
        Passed to :func:`interpolate_bads_coords`.

    Returns
    -------
    out : xr.DataArray
        Same shape, dims, and coordinates as ``sig`` with bad-channel values
        replaced by their interpolated counterparts.

    See Also
    --------
    interpolate_bads_coords : Plain numpy primitive.
    """
    import xarray as xr

    if ch_dim not in sig.dims:
        raise ValueError(f"sig must have a '{ch_dim}' dimension; got dims={sig.dims}")
    for name in (x_coord, y_coord):
        if name not in sig.coords:
            raise ValueError(f"sig must have a '{name}' coordinate along '{ch_dim}'")

    sig_ch_first = sig.transpose(ch_dim, ...)
    coords_xy = np.column_stack(
        [
            np.asarray(sig_ch_first.coords[x_coord].values),
            np.asarray(sig_ch_first.coords[y_coord].values),
        ]
    )
    out_arr = interpolate_bads_coords(
        sig_ch_first.values,
        coords_xy,
        np.asarray(bad),
        method=method,
    )
    out = xr.DataArray(
        out_arr,
        dims=sig_ch_first.dims,
        coords=sig_ch_first.coords,
        attrs=sig.attrs,
    )
    return out.transpose(*sig.dims)


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
