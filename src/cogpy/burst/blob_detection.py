"""
Functions in this module are used to detect blobs in n-dimensional data arrays.

detect_bursts: Detects bursts in a n-dimensional data array using the h-maxima transform.
get_coords_fs_dict: Gets the sampling rate for each dimension in dataarray.
set_sigma_dict: Sets the sigma values for the blob detection.
separate_min_max_sigma_dict: Separates the sigma dictionary into min and max sigma dictionaries.
get_blobs_df: Gets the blobs in a data array and returns a dataframe with their coordinates and amplitudes.

dependencies between functions:

get_coords_fs_dict, separate_min_max_sigma_dict -> set_sigma_dict
set_sigma_dict -> get_blobs_df
detect_bursts -> get_blobs_df
"""

import numpy as np
import xarray as xr
import pandas as pd
import warnings
from cogpy.utils.imports import import_optional

import_optional("skimage")
from skimage.feature import blob_log  # , blob_dog, blob_doh
from skimage.morphology import extrema
from ..preprocess.filtering import get_coord_fs


def detect_hmaxima(
    datax: xr.DataArray, h_quantile: float = 0.8, h=None, footprint=None
) -> pd.DataFrame:
    """
    Detect bursts (h-maxima) in a n-dimensional data array using the h-maxima transform.
    The h-maxima transform is a morphological operation that identifies local maxima in the data.
    The function uses the quantile of the data to determine the height of the maxima.
    The function returns a dataframe with the coordinates of the detected bursts and their amplitudes.

    Parameters
    ----------
    datax : xarray.DataArray
        The input data array with dimensions and coordinates.

    quantile : float
        The quantile used to determine the height of the maxima. Default is 0.8.

    Returns
    -------
    bursts_df : pd.DataFrame
        A dataframe with the coordinates of the detected bursts and their amplitudes.
        The dataframe has the following columns:
            - AP: Anterior-Posterior coordinate
            - ML: Medio-Lateral coordinate
            - amp: Amplitude of the burst
            - time: Time coordinate
            - iAP: Index of the Anterior-Posterior coordinate
            - iML: Index of the Medio-Lateral coordinate
            - iTime: Index of the time coordinate
    """  # Calculate the high threshold based on the quantile
    if h is None:
        h = np.quantile(datax.data, h_quantile)
        if h == 0:
            h = 0.1

    # Detect local maxima above the threshold
    if h > np.ptp(datax.data):
        return pd.DataFrame(columns=datax.dims)
    h_maxima = extrema.h_maxima(datax.data, h, footprint=footprint)
    bursts_df = get_coo_df(datax, h_maxima)
    return bursts_df


def get_coo_df(datax, h_maxima):
    # Get the indices of the detected bursts
    bursts_icoos = np.where(h_maxima)

    # Create a DataFrame to store burst indices, using coordinate names from the DataArray
    bursts_icoos_df = pd.DataFrame(
        {dim: bursts_icoos[dim_idx] for dim_idx, dim in enumerate(datax.dims)},
        columns=datax.dims,
    )

    # Add amplitude information
    bursts_icoos_df.loc[:, "amp"] = datax.data[bursts_icoos]

    # Add columns for coordinates using the coordinate values from the DataArray
    for dim in datax.dims:
        bursts_icoos_df[f"{dim}"] = datax.coords[dim].values[bursts_icoos_df[dim]]

    # Optionally, if you want the index of coordinates, assuming the DataArray has a corresponding index
    # This part assumes datax has a multi-index that corresponds to the coordinate values.
    # If this is not the case, you may need to adjust the logic to fit your specific data structure.
    if hasattr(datax, "indexes") and all(dim in datax.indexes for dim in datax.dims):
        for dim in datax.dims:
            bursts_icoos_df[f"i{dim}"] = bursts_icoos_df.apply(
                lambda row: datax.indexes[dim].get_loc(row[f"{dim}"]), axis=1
            )

    return bursts_icoos_df


def get_coords_fs_dict(datax):
    """
    Get the sampling rate for each dimension in dataarray

    Parameters
    ----------
    datax : xarray.DataArray

    Returns
    -------
    fs_dict : dict
        A dictionary of the sampling rate for each dimension in datax
    """
    fs_dict = {}
    for dim, coo in datax.coords.items():
        fs_dict[dim] = get_coord_fs(coo)
    return fs_dict


def set_sigma_dict(sigma_dict, datax, return_tuple=True):
    """
    Set the sigma values for the blob detection.
    Parameters
    ----------
    sigma_dict : dict
        A dictionary with the sigma values for each dimension. The values should be tuples of (min sigma, max sigma) for each dimension.
    datax : xarray.DataArray
        The input data array with dimensions and coordinates.
    return_tuple : bool
        If True, the function returns a tuple of (min_sigma, max_sigma). If False, it returns a dictionary with the min and max sigma values for each dimension.
    Returns
    -------
    min_sigma_dict : dict or tuple
        A dictionary or tuple with the minimum sigma values for each dimension.
    max_sigma_dict : dict or tuple
        A dictionary or tuple with the maximum sigma values for each dimension.
    """

    min_sigma_dict, max_sigma_dict = separate_min_max_sigma_dict(sigma_dict)
    fs_dict = get_coords_fs_dict(datax)
    min_sigma_dict = {
        dim: int(val * fs_dict[dim]) for dim, val in min_sigma_dict.items()
    }
    max_sigma_dict = {
        dim: int(val * fs_dict[dim]) for dim, val in max_sigma_dict.items()
    }
    # assert all numbers in min_sigma_tuple are greater than 0
    for dim, val in min_sigma_dict.items():
        if val == 0:
            warn_message = f"{dim} min_sigma is too small: setting to the corresponding dimesnion sampling period: {1/fs_dict[dim]}"
            # raise warning
            warnings.warn(warn_message)
            min_sigma_dict[dim] = 1

    # assert all numbers in max_sigma_tuple are greater than min_sigma_tuple
    for dim, val in max_sigma_dict.items():
        if val <= min_sigma_dict[dim]:
            warn_message = f"{dim} max_sigma is too small: setting to the corresponding 2 * min_sigma: {2*min_sigma_dict[dim]}"
            # raise warning
            warnings.warn(warn_message)
            max_sigma_dict[dim] = 2 * min_sigma_dict[dim]
    if return_tuple:
        min_sigma_dict = tuple(min_sigma_dict.values())
        max_sigma_dict = tuple(max_sigma_dict.values())
    return min_sigma_dict, max_sigma_dict


def separate_min_max_sigma_dict(sigma_dict_raw):
    """
    Separate the sigma dictionary into min and max sigma dictionaries.
    Parameters
    ----------
    sigma_dict_raw : dict
        A dictionary with the sigma values for each dimension. The values should be tuples of (min
        sigma, max sigma) for each dimension.
    Returns
    -------
    min_sigma_dict : dict
        A dictionary with the minimum sigma values for each dimension.
    max_sigma_dict : dict
        A dictionary with the maximum sigma values for each dimension.
    """
    min_sigma_dict = {dim: val[0] for dim, val in sigma_dict_raw.items()}
    max_sigma_dict = {dim: val[1] for dim, val in sigma_dict_raw.items()}
    return min_sigma_dict, max_sigma_dict


def detect_blobs(
    datax: xr.DataArray,
    num_sigma: int = 10,
    sigma_dict_raw=None,
    *,
    threshold_rel: float = 0.2,
    exclude_border: int | bool = 1,
) -> pd.DataFrame:
    """
    Get the blobs (Laplace of Gaussian blobs) in a data array and return a dataframe with their coordinates and amplitudes.
    Parameters
    ----------
    datax : xarray.DataArray
        The input data array with dimensions and coordinates.
    num_sigma : int
        The number of sigma values to use for the blob detection. Default is 10.
    sigma_dict_raw : dict, optional
        A dictionary with the sigma values for each dimension. If None, the function will use the
        default values of (1/fs, 5/fs) for each dimension, where fs is
        the sampling rate of the dimension. Default is None.
    Returns
    -------
    blob_df : pd.DataFrame
        A dataframe with the coordinates of the detected blobs and their amplitudes.
        The dataframe has the following columns:
            - AP: Anterior-Posterior coordinate
            - ML: Medio-Lateral coordinate
            - time: Time coordinate
            - iAP: Index of the Anterior-Posterior coordinate
            - iML: Index of the Medio-Lateral coordinate
            - iTime: Index of the time coordinate
            - iAP_sigma: Index of the Anterior-Posterior coordinate sigma
            - iML_sigma: Index of the Medio-Lateral coordinate sigma
            - iTime_sigma: Index of the time coordinate sigma
            - amp: Amplitude of the blob
    """
    fs_dict = get_coords_fs_dict(datax)
    if sigma_dict_raw is None:
        sigma_dict_raw = {dim: (1 / fs, 5 / fs) for dim, fs in fs_dict.items()}
    min_sigma, max_sigma = set_sigma_dict(sigma_dict_raw, datax)
    blobs_doh = blob_log(
        datax.data,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold_rel=float(threshold_rel),
        exclude_border=exclude_border,
    )
    coords = datax.coords
    dims = datax.dims
    blob_df = pd.DataFrame(
        blobs_doh,
        columns=["i" + dim for dim in dims] + ["i" + dim + "_sigma" for dim in dims],
    )
    blob_df.index.name = "blob_idx"

    # add physical coordinates
    for dim in dims:
        fs_dim = fs_dict[dim]
        blob_df.loc[:, dim] = coords[dim].values[blob_df["i" + dim].values.astype(int)]
        blob_df.loc[:, dim + "_sigma"] = blob_df["i" + dim + "_sigma"].values / fs_dim

    # reordering columns (physical coordinates first, then indices, then sigmas)
    dim_sigmas = [dim + "_sigma" for dim in dims]
    idims = ["i" + dim for dim in dims]
    idim_sigmas = ["i" + dim + "_sigma" for dim in dims]
    col_order = list(dims) + dim_sigmas + idims + idim_sigmas

    # convert idims to int
    blob_df[idims] = blob_df[idims].astype(int)
    blob_df = blob_df[col_order]

    # add amplitude
    def _amp_at_center(row) -> float:
        idx = {dim: int(getattr(row, "i" + dim)) for dim in dims}
        v = np.asarray(datax.isel(**idx).values)
        if v.size == 0:
            return float("nan")
        if v.size == 1:
            return float(v.reshape(-1)[0].item())
        # Defensive fallback: squeeze unexpected non-scalar selections.
        return float(v.reshape(-1)[0].item())

    # Use itertuples to avoid pandas apply() expanding non-scalar returns into a DataFrame.
    blob_df.loc[:, "amp"] = [_amp_at_center(r) for r in blob_df.itertuples(index=False)]
    return blob_df
