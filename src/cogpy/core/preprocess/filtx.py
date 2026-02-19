# spatiotemporal filtering
import numpy as np
import xarray as xr
from scipy import signal
import scipy.ndimage as nd
from functools import partial
import scipy.stats as sts
from ..utils import xarr as xut
from .filt import bandpass_filt_params
from ..utils.grid_neighborhood import make_footprint


def get_coord_fs(coo):
    """
    Get the sampling rate for a coordinate in dataarray
    The sampling rate is computed as the inverse of the smallest non-zero difference between consecutive values in the coordinate.
    Parameters
    ----------
    coo : xarray.DataArray
            The coordinate to compute the sampling rate for.
    Returns
    -------
    fs : float
            The sampling rate for the coordinate.
    """
    diff_vals = np.abs(np.diff(coo.values))
    smallest_nonzero_diff_val = np.min(diff_vals[diff_vals > 0])
    fs = 1 / smallest_nonzero_diff_val
    return fs


def butterworth_bandpass_shoulder(
    x: xr.DataArray,
    fs: float,
    low: float,
    high: float,
    shoulder: float = 25.0,
    rp: float = 1.0,
    rs: float = 40.0,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    General Butterworth bandpass filter with shoulder specification (zero-phase).

    This function designs a Butterworth filter using buttord() to meet the
    passband and stopband specifications, then applies it with zero-phase filtering.

    Parameters
    ----------
    x : xr.DataArray
            Input time series data.
    fs : float
            Sampling rate in Hz.
    low, high : float
            Passband edges in Hz.
    shoulder : float, optional
            Frequency shoulder width in Hz. Stopband edges will be at
            [low - shoulder, high + shoulder]. Default is 25.0 Hz.
    rp : float, optional
            Maximum ripple in the passband (dB). Default is 1.0 dB.
    rs : float, optional
            Minimum attenuation in the stopband (dB). Default is 40.0 dB.
    time_dim : str, optional
            Name of the time dimension. Default is 'time'.

    Returns
    -------
    xr.DataArray
            Filtered data with same shape and coordinates as input.

    Raises
    ------
    AssertionError
            If the specified time dimension is not found in the input array.
    ValueError
            If filter design parameters result in invalid filter specifications.
    """
    assert time_dim in x.dims, f"x must have a '{time_dim}' dimension"

    # Define passband and stopband frequencies
    wp = [low, high]  # Passband edges
    ws = [low - shoulder, high + shoulder]  # Stopband edges

    # Normalize frequencies to Nyquist frequency
    nyquist = fs / 2
    wp_norm = np.array(wp) / nyquist
    ws_norm = np.array(ws) / nyquist

    # Design the filter order and cutoff frequencies
    N, Wn = signal.buttord(wp=wp_norm, ws=ws_norm, gpass=rp, gstop=rs)

    # Create the filter in second-order sections form for numerical stability
    sos = signal.butter(N, Wn, btype="bandpass", output="sos")

    # Apply zero-phase filtering
    time_axis = x.get_axis_num(time_dim)
    filtered_data = signal.sosfiltfilt(sos, x.values, axis=time_axis)

    # Create output DataArray with same structure as input
    result_name = x.name + "_bandpass" if x.name else "bandpass_filtered"
    attrs = x.attrs.copy()
    attrs.update(
        {
            "filter_type": "butterworth_bandpass",
            "passband_hz": f"{low}-{high}",
            "stopband_hz": f"{low-shoulder}-{high+shoulder}",
            "filter_order": N,
            "passband_ripple_db": rp,
            "stopband_attenuation_db": rs,
        }
    )

    return xr.DataArray(
        filtered_data, coords=x.coords, dims=x.dims, name=result_name, attrs=attrs
    )


def bandpassx(
    sigx: xr.DataArray, wl: float, wh: float, order: int, axis: str
) -> xr.DataArray:
    b, a = bandpass_filt_params(order, wl, wh, sigx.fs)
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Bandpass ({}-{} Hz)".format(wl, wh)
    return sigx_bp


def lowpassx(sigx: xr.DataArray, wl: float, order: int, axis: str) -> xr.DataArray:
    b, a = signal.butter(order, wl / (0.5 * sigx.fs), btype="low")
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Lowpass ({} Hz)".format(wl)
    return sigx_bp


def highpassx(sigx: xr.DataArray, wh: float, order: int, axis: str) -> xr.DataArray:
    b, a = signal.butter(order, wh / (0.5 * sigx.fs), btype="high")
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Highpass ({} Hz)".format(wh)
    return sigx_bp


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


def median(input_arr, footprint=None):
    """Compute local median of input array using neighborhood footprint from gneigh."""
    # ndof = input_arr.ndim - 2  # number of leading dimensions before AP, ML
    # footprint_full_shape = (1,) * ndof + gneigh.footprint.shape
    # footprint = gneigh.footprint.reshape(footprint_full_shape)
    if footprint is None:
        footprint = make_footprint(rank=2, connectivity=1, niter=2)
    filter_kwargs = dict(footprint=footprint, mode="constant", cval=np.nan)
    local_med = nd.generic_filter(input_arr, function=np.nanmedian, **filter_kwargs)
    return local_med


def mad(input_arr, footprint=None):
    """Compute local median absolute deviation of input array using neighborhood footprint from gneigh."""
    # ndof = input_arr.ndim - 2  # number of leading dimensions before AP, ML
    # footprint_full_shape = (1,) * ndof + gneigh.footprint.shape
    # footprint = gneigh.footprint.reshape(footprint_full_shape)
    if footprint is None:
        footprint = make_footprint(rank=2, connectivity=1, niter=2)
    filter_kwargs = dict(footprint=footprint, mode="constant", cval=np.nan)
    scaled_mad_func = partial(
        sts.median_abs_deviation, scale="normal", nan_policy="omit"
    )
    local_mad = nd.generic_filter(input_arr, function=scaled_mad_func, **filter_kwargs)
    return local_mad
