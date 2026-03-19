"""
Module: channel_features_nanmedian.
Status: DEPRECATED (legacy compatibility only)
Last Updated: 2025-08-26
Author: Arash Shahidi, A.Shahidi@campus.lmu.de

Summary:
        This module provides functionality for extracting channel features from neural data, with a focus on handling NaN values.
        Canonical replacements live under:
            - `cogpy.preprocess.badchannel.channel_features`
            - `cogpy.preprocess.badchannel.spatial`
            - `cogpy.preprocess.badchannel.pipeline`

Functions:
        - anticorrelation
        - noise_to_signal
        - relative_variance
        - deviation
        - standard_deviation
        - amplitude
        - kurtosis
        - time_derivative
        - hurst_exponent
        - is_dead
        - laplacian
        - spatial_gradient
        - gradient

Classes:

Constants:

Example:
"""

import warnings as _warnings

_warnings.warn(
    "cogpy.preprocess.channel_feature_functions is deprecated. "
    "Use cogpy.preprocess.badchannel instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
import xarray as xr
import dask.array as da
import scipy.ndimage as nd
import scipy.stats as sts
from scipy import signal
from functools import partial
from ..utils.reshape import ravel_dims
from ..utils.grid_neighborhood import adjacency_edges, adjacency_matrix, make_footprint
import warnings

EPSILON = 0.000001


# %% utils
def logeps(x):
    error = "logeps input should be nonnegative"
    if isinstance(x, (int, float)):
        assert x >= 0, print(x)
    if isinstance(x, np.ndarray):
        assert (x >= 0).all(), print(x)
    return np.log(x + EPSILON)


def swap_axis_to_last(func):
    def wrapper(a, *args, axis=-1, **kwargs):
        a = np.swapaxes(a, axis, -1)
        return func(a, *args, axis=-1, **kwargs)

    return wrapper


# %% neighborhood functions:
def local_robust_zscore(input_arr, footprint=None):
    """Compute local robust z-score of input array using neighborhood footprint from gneigh."""
    # ndof = input_arr.ndim - 2  # number of leading dimensions before AP, ML
    # footprint_full_shape = (1,) * ndof + gneigh.footprint.shape
    # footprint = gneigh.footprint.reshape(footprint_full_shape)
    if footprint is None:
        footprint = make_footprint(rank=2, connectivity=1, niter=2)
    filter_kwargs = dict(footprint=footprint, mode="constant", cval=np.nan)
    scaled_mad_func = partial(
        sts.median_abs_deviation, scale="normal", nan_policy="omit"
    )
    local_med = nd.generic_filter(input_arr, function=np.nanmedian, **filter_kwargs)
    local_mad = nd.generic_filter(input_arr, function=scaled_mad_func, **filter_kwargs)
    denom = np.where(local_mad > 0, local_mad, np.nan)
    robust_zscore_center = (input_arr - local_med) / denom
    return robust_zscore_center


def local_robust_zscore_dask(
    input_arr, footprint=None, *, mode="constant", cval=np.nan
):
    """Local robust z-score using dask.

    This implementation matches `local_robust_zscore(...)` by using
    `scipy.ndimage.generic_filter` under `dask.array.map_overlap`, preserving the
    NaN-ignoring semantics of `np.nanmedian` and `median_abs_deviation(..., nan_policy="omit")`.
    """
    is_xr = isinstance(input_arr, xr.DataArray)
    x = da.asarray(input_arr.data if is_xr else input_arr)
    if x.ndim < 2:
        raise ValueError("Need at least 2D input.")
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype("f8")

    if mode != "constant" or not (np.isnan(cval) or cval is None):
        warnings.warn(
            "local_robust_zscore_dask currently matches the numpy implementation only for "
            "mode='constant', cval=np.nan; other modes may differ.",
            RuntimeWarning,
            stacklevel=2,
        )

    if footprint is None:
        footprint = make_footprint(rank=2, connectivity=1, niter=2)
    footprint = np.asarray(footprint, dtype=bool)
    assert footprint.ndim == 2, "footprint must be 2D"
    f = footprint.reshape((1,) * (x.ndim - 2) + footprint.shape)

    radius0 = int(footprint.shape[0] // 2)
    radius1 = int(footprint.shape[1] // 2)
    depth = (0,) * (x.ndim - 2) + (radius0, radius1)

    scaled_mad_func = partial(sts.median_abs_deviation, scale="normal", nan_policy="omit")

    def _block_fn(block: np.ndarray, *, footprint_full: np.ndarray) -> np.ndarray:
        filter_kwargs = dict(footprint=footprint_full, mode="constant", cval=np.nan)
        local_med = nd.generic_filter(block, function=np.nanmedian, **filter_kwargs)
        local_mad = nd.generic_filter(block, function=scaled_mad_func, **filter_kwargs)
        denom = np.where(local_mad > 0, local_mad, np.nan)
        return (block - local_med) / denom

    z = da.map_overlap(
        _block_fn,
        x,
        depth=depth,
        boundary=np.nan,
        trim=True,
        dtype="f8",
        footprint_full=f,
    )

    if is_xr:
        return xr.DataArray(
            z,
            coords=input_arr.coords,
            dims=input_arr.dims,
            name="robust_z",
            attrs=input_arr.attrs,
        )
    return z


# %% Features
# identity
def anticorrelation(arr, adj=None):
    """
    double ch -> median over neighbors

    Parameters:
    a: array (AP, ML, time)

    Returns
    -------
    1-mean_corr: array (AP, ML)
    """
    # flatten
    grid_shape = arr.shape[:2]
    a_flat = ravel_dims(arr, 0, 1)
    if adj is None:
        adj = adjacency_matrix(grid_shape, exclude=True)

    # Use errstate to ignore RuntimeWarning from corrcoef
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(a_flat)
    corr = np.nan_to_num(corr)
    neighbor_corr = np.where(adj, corr, np.nan)
    med_corr = np.nanmedian(neighbor_corr, axis=1)
    anticorr = 1 - med_corr
    return anticorr.reshape(grid_shape)


# diff exclude
@swap_axis_to_last
def noise_to_signal(arr, fs, axis=-1, low_freq=(0.1, 30), high_freq=(30, 80)):
    """
    a: array (*dof, time)
    """
    # nyquist above 80 Hz
    dims_dof = [f"dim{i}" for i in range(arr.ndim - 1)]
    nperseg = 256
    if fs / 2 <= 50:
        warnings.warn("Sampling frequency must be above 100 Hz")
    _, psd = signal.welch(arr, fs=fs, axis=axis)
    psd = xr.DataArray(
        psd, dims=dims_dof + ["freq"], coords={"freq": np.fft.rfftfreq(nperseg, 1 / fs)}
    )
    power_low = psd.sel(freq=slice(*low_freq)).reduce(np.nanmean, dim="freq")
    power_high = psd.sel(freq=slice(*high_freq)).reduce(np.nanmean, dim="freq")
    nsr = power_high / (power_low + EPSILON)
    return nsr.data


# ratio all log
@swap_axis_to_last
def relative_variance(arr, axis=-1):
    """
    single ch
    a: array (*dof, time)
    """
    rel_var = np.var(arr, axis=axis)
    return rel_var


# diff exclude log
@swap_axis_to_last
def deviation(arr, axis=-1):
    """
    single ch
    a: array (*dof, time)
    """
    rel_mean = np.nanmean(arr, axis=axis)
    return rel_mean


# ratio exclude log
@swap_axis_to_last
def standard_deviation(arr, axis=-1):
    """
    single ch
    a: array (*dof, time)
    """
    std = np.std(arr, axis=axis)
    return std


# diff exclude log
@swap_axis_to_last
def amplitude(arr, axis=-1):
    """
    single ch
    a: array (*dof, time)
    """
    rel_amp = np.max(arr, axis=axis) - np.min(arr, axis=axis)
    return rel_amp


# diff exclude
@swap_axis_to_last
def kurtosis(arr, axis=-1):
    """
    a: array (*dof, time)
    """
    kurt_ch = sts.kurtosis(arr, axis=axis)
    return kurt_ch


# ratio exclude
@swap_axis_to_last
def time_derivative(arr, axis=-1):
    """
    temporal
    a: array (*dof, time)
    """
    tder_ch = np.nanmean(np.abs(np.array(np.gradient(arr, axis=axis))), axis=axis)
    return tder_ch


# diff exclude
@swap_axis_to_last
def hurst_exponent(arr: np.ndarray, axis=-1):
    """
    temporal
    a: array (*dof, time)
    """
    y = arr - np.nanmean(arr, axis=axis, keepdims=True)
    z = np.cumsum(y, axis=axis)
    r = np.max(z, axis=axis) - np.min(z, axis=axis)
    std = np.std(arr)  # axis or not
    hurst_ch = logeps(r / std) / 2
    return hurst_ch


@swap_axis_to_last
def is_dead(arr, axis=-1):
    """
    temporal
    a: array (*dof, time)
    Returns
     (channel, ) bool True: dead (constant)
    """
    return np.max(arr, axis=axis) == np.min(arr, axis=axis)


# %% spatial features
# ratio exclude
def laplacian(arr, gridshape, stencil="9"):
    """
    neighborhood
    a: array (AP, ML, time)
    """
    if stencil == "9":
        laplacian_stencil = [[1, 4, 1], [4, -20, 4], [1, 4, 1]]
    elif stencil == "13":
        laplacian_stencil = [
            [0, 0, 1, 0, 0],
            [0, 2, -8, 2, 0],
            [1, -8, 20, -8, 1],
            [0, 2, -8, 2, 0],
            [0, 0, 1, 0, 0],
        ]
    else:
        raise NotImplementedError("Unsupported stencil type")
    a_laplacian = nd.convolve(
        arr, np.expand_dims(laplacian_stencil, axis=-1), mode="mirror"
    )
    return a_laplacian


def temporal_mean_laplacian(arr):
    """
    a: array (AP, ML, time)
    """
    gridshape = (arr.shape[-3], arr.shape[-2])
    a_laplacian = laplacian(arr, gridshape)
    tmed_laplacian = np.mean(np.abs(a_laplacian), axis=-1)
    return tmed_laplacian


# ratio exclude log
def spatial_gradient(arr):
    """
    a: array (AP, ML, time)
    """
    rel_grad = np.nanmean(
        np.linalg.norm(np.array(np.gradient(arr, axis=(-3, -2))), axis=-4), axis=-1
    )
    return rel_grad


# ratio exclude
def gradient(arr, adj=None):
    """
    spatial
    a: array (AP, ML, time)
    """
    # pad a with nan
    grid_shape = arr.shape[:2]
    a_flat = ravel_dims(arr, 0, 1)
    if adj is None:
        adj = adjacency_matrix(grid_shape, exclude=True)

    med_grad = np.nanmedian(
        np.where(
            np.expand_dims(adj, axis=-1),
            np.abs(a_flat[None, :, :] - a_flat[:, None, :]),
            np.nan,
        ),
        axis=1,
    )
    return np.nanmean(med_grad, axis=-1).reshape(
        grid_shape
    )  # temporal_mean_neighbor_med_grad_ch


def gradient_fast(arr, adj_src=None, adj_dst=None):
    """
    WIP: Make it DASK compatible

    Compute, for each node i, the temporal mean of |x_i - x_j| over time,
    then take the median over neighbors j. Shape: (AP, ML).
    """
    AP, ML, T = arr.shape
    N = AP * ML
    a_flat = arr.reshape(N, T)

    if adj_src is None or adj_dst is None:
        adj_src, adj_dst = adjacency_edges((AP, ML), exclude=True)  # boolean (N, N)

    # Compute pairwise diffs only on edges (E, T), not on all N×N×T
    diffs = a_flat[adj_src] - a_flat[adj_dst]  # (E, T)
    edge_mean = np.nanmean(np.abs(diffs), axis=1)  # (E,)

    # Aggregate per source node: median over its incident edges
    # Sort by src to allow fast grouping
    counts = np.bincount(adj_src, minlength=N)  # (#edges per node)
    offsets = np.concatenate(([0], np.cumsum(counts[:-1])))

    out = np.full(N, np.nan, dtype=edge_mean.dtype)
    for i in range(N):
        c = counts[i]
        if c:
            s = offsets[i]
            e = s + c
            print(edge_mean[s:e].shape)
            out[i] = np.nanmedian(edge_mean[s:e])

    return out.reshape(AP, ML)


def gradient_rms_fast(arr, adj_src=None, adj_dst=None):
    """
    L2 version: per edge sqrt(E[(xi-xj)^2]) via vars + covs.
    Much faster; then median over neighbors. Returns (AP, ML).
    """
    arr = np.asarray(arr)
    AP, ML, T = arr.shape
    N = AP * ML
    X = arr.reshape(N, T).astype(np.float64)

    # center
    X = X - np.nanmean(X, axis=1, keepdims=True)
    # replace remaining nans if any (optional; or mask them properly)
    X = np.nan_to_num(X)

    var = (X * X).mean(axis=1)  # (N,)

    # edges
    if adj_src is None or adj_dst is None:
        adj_src, adj_dst = adjacency_edges((AP, ML), exclude=True)
    # covariance per edge is just mean of elementwise product (since centered)
    cov_edges = (X[adj_src] * X[adj_dst]).mean(axis=1)
    # σ^2(xi - xj) = var_i + var_j - 2*cov_ij
    sigma2 = var[adj_src] + var[adj_dst] - 2.0 * cov_edges
    edge_val = np.sqrt(np.maximum(sigma2, 0.0))

    # median over neighbors (group by src)
    counts = np.bincount(adj_src, minlength=N)
    offsets = np.concatenate(([0], np.cumsum(counts[:-1])))

    out = np.full(N, np.nan)
    for i in range(N):
        c = counts[i]
        if c:
            s = offsets[i]
            e = s + c
            out[i] = np.nanmedian(edge_val[s:e])
    return out.reshape(AP, ML)


# ratio exclude
def gradient_original(arr, gridshape, loc_exclude_df):
    """
    a: array (time, channel)
    """
    nchan = np.prod(gridshape)
    grad = np.abs(
        arr[:, loc_exclude_df["ch_neighbor"]] - arr[:, loc_exclude_df["ch_ref"]]
    )
    med_grad = np.array(
        [
            np.nanmedian(grad[:, loc_exclude_df["ch_ref"] == ch], axis=1)
            for ch in range(nchan)
        ]
    )
    mean_med_grad_ch = np.nanmean(np.abs(med_grad), axis=1)
    return mean_med_grad_ch
