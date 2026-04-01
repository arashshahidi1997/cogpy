"""Internal utilities shared across filtering submodules."""

import numpy as np
import xarray as xr
from scipy import signal


def bandpass_filt_params(order=4, low=1, high=50, fs=625):
    """Return Butterworth bandpass filter coefficients (b, a).

    Parameters
    ----------
    order : int
        Filter order.
    low, high : float
        Low and high cutoff frequencies in Hz.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    b, a : ndarray
        Numerator and denominator of the IIR filter.
    """
    high = min(high, fs / 2 - 0.1)
    b, a = signal.butter(order, [low, high], btype="bandpass", output="ba", fs=fs)
    return b, a


def _fs_scalar(sigx: xr.DataArray) -> float:
    """Return a Python float sampling rate from ``sigx.fs``/attrs."""
    fs = getattr(sigx, "fs", None)
    if fs is None:
        fs = sigx.attrs.get("fs", None)
    if fs is None:
        raise ValueError(
            "Input must have a `.fs` coordinate/attribute indicating sampling frequency."
        )
    # xarray coord returns a 0D DataArray; convert robustly
    if hasattr(fs, "item"):
        fs = fs.item()
    return float(fs)


def _apply_full_array(
    sigx: xr.DataArray,
    func,
    *,
    output_dtype=None,
    **kwargs,
) -> xr.DataArray:
    """Apply a NumPy/SciPy function to the full DataArray values, preserving metadata.

    Uses ``xr.apply_ufunc`` with all dims as core dims, which works for both NumPy
    and Dask-backed arrays (when feasible).
    """
    if not isinstance(sigx, xr.DataArray):
        raise TypeError("Expected an xarray.DataArray")
    core_dims = [list(sigx.dims)]
    output_dtype = sigx.dtype if output_dtype is None else output_dtype
    out = xr.apply_ufunc(
        func,
        sigx,
        input_core_dims=core_dims,
        output_core_dims=core_dims,
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[output_dtype],
    )
    out = xr.DataArray(
        out.data,
        dims=sigx.dims,
        coords=sigx.coords,
        attrs=dict(sigx.attrs),
        name=sigx.name,
    )
    return out


def get_coord_fs(coo):
    """Get the sampling rate for a coordinate in dataarray.

    The sampling rate is computed as the inverse of the smallest non-zero
    difference between consecutive values in the coordinate.

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
