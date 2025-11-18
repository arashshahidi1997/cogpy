import numpy as np
import scipy.ndimage as nd
import xarray as xr
from scipy.interpolate import griddata


def compress_specgram(mt):
    mt_dict = {}
    for key, val in mt.__dict__.items():
        if val is not None:
            if key in ["specgram", "times", "freqs"]:
                val = val.astype(np.float32)
            mt_dict |= {key: val}
    return mt_dict


def clip_freq(mt):
    return mt.clip(freq_range=[1, 30])


def median_spec(mtx, med_size):
    """
    med_size = (3, 3, 1, 1) # h, w, f, t
    """
    mtx_filt = xr.apply_ufunc(lambda x: nd.median_filter(x, size=med_size), mtx)
    if "_filt_log" not in mtx.attrs:
        mtx.attrs |= dict(_filt_log=[])
    mtx_filt.attrs = mtx.attrs.copy()
    mtx_filt.attrs["_filt_log"].append(
        {"name": "spatial lowpass median", "params": {"size": med_size}}
    )
    return mtx_filt


def gaussian_spec(mtx, gauss_sigma):
    """
    gauss_sigma = (0.5, 0.5, 0.5, 10) # AP, ML, freq, time
    """
    mtx_filt = xr.apply_ufunc(lambda x: nd.gaussian_filter(x, sigma=gauss_sigma), mtx)
    if "_filt_log" not in mtx.attrs:
        mtx.attrs |= dict(_filt_log=[])
    mtx_filt.attrs = mtx.attrs.copy()
    mtx_filt.attrs["_filt_log"].append(
        {
            "name": "spatio-spectro-temporal lowpass gaussian",
            "params": {"sigma": gauss_sigma},
        }
    )
    return mtx_filt


def process_specgram(mtx, filt=False):
    mtx = mtx.sel(freq=slice(8, 30))
    if "_filt_log" not in mtx.attrs:
        mtx.attrs |= dict(_filt_log=[])
    if filt:
        med_size = (3, 3, 3, 5)  # h, w, f, t
        mtx = median_spec(mtx, med_size)
        # gauss_sigma = (0.5, 0.5, 0.5, 10) # h, w, f, t
        # mtx = gaussian_spec(mtx, gauss_sigma)
    return mtx.stack(ch=("h", "w"))


def get_center(x):
    return x[tuple([k // 2 for k in x.shape])]


def is_outlier(x, xcenter, threshold=2):
    return np.abs(xcenter - np.nanmedian(x)) > threshold * np.std(x)


def nan_if_outlier(x, threshold=2):
    xcenter = get_center(x)
    return np.nan if is_outlier(x, threshold) else xcenter


def nan_if_outlier_generic(arr, kernel, threshold=2):
    """
    arr: numpy array
    kernel: tuple of ints
        window size for outlier detection
    threshold: float
        thresholdold for outlier detection
    """
    arr_filt = nd.generic_filter(
        arr, lambda x: nan_if_outlier(x, threshold), size=kernel, mode="reflect"
    )
    return arr_filt


def interpolate(x):
    refcoos = np.where(np.invert(np.isnan(x)))
    iarr = griddata(refcoos, x[refcoos], xcenter, method="linear")
    xcenter = get_center(iarr)
    return xcenter


def interpolate_if_nan(x):
    xcenter = get_center(x)
    return interpolate(x) if np.isnan(xcenter) else xcenter


def median_if_nan(x):
    xcenter = get_center(x)
    return np.nanmedian(x) if np.isnan(xcenter) else xcenter


def fix_outliers(x, kernel, threshold=2):
    """
    x: numpy array
    kernel: tuple of ints
        window size for outlier detection
    threshold: float
        thresholdold for outlier detection
    """
    # remove outliers
    print("removing outliers")
    x_filt = nd.generic_filter(
        x, lambda x: nan_if_outlier(x, threshold), size=kernel, mode="reflect"
    )
    # interpolate nans
    print("interpolating nans")
    x_filt = nd.generic_filter(x_filt, interpolate_if_nan, size=kernel, mode="reflect")
    # median filter nans
    print("median filtering remaining nans")
    x_filt = nd.generic_filter(x_filt, median_if_nan, size=kernel, mode="reflect")
    return x_filt
