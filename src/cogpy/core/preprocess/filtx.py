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


def _fs_scalar(sigx: xr.DataArray) -> float:
    """Return a Python float sampling rate from ``sigx.fs``/attrs."""
    fs = getattr(sigx, "fs", None)
    if fs is None:
        fs = sigx.attrs.get("fs", None)
    if fs is None:
        raise ValueError("Input must have a `.fs` coordinate/attribute indicating sampling frequency.")
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
    out = xr.DataArray(out.data, dims=sigx.dims, coords=sigx.coords, attrs=dict(sigx.attrs), name=sigx.name)
    return out


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
    b, a = bandpass_filt_params(order, wl, wh, _fs_scalar(sigx))
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Bandpass ({}-{} Hz)".format(wl, wh)
    return sigx_bp


def lowpassx(sigx: xr.DataArray, wl: float, order: int, axis: str) -> xr.DataArray:
    b, a = signal.butter(order, wl / (0.5 * _fs_scalar(sigx)), btype="low")
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Lowpass ({} Hz)".format(wl)
    return sigx_bp


def highpassx(sigx: xr.DataArray, wh: float, order: int, axis: str) -> xr.DataArray:
    b, a = signal.butter(order, wh / (0.5 * _fs_scalar(sigx)), btype="high")
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Highpass ({} Hz)".format(wh)
    return sigx_bp


def notchx(
    sigx: xr.DataArray,
    *,
    w0: float = 60.0,
    Q: float = 30.0,
    time_dim: str = "time",
) -> xr.DataArray:
    """Notch filter (IIR notch) applied along ``time_dim`` using zero-phase filtering."""
    fs = _fs_scalar(sigx)
    axis = sigx.get_axis_num(time_dim)
    b, a = signal.iirnotch(float(w0), float(Q), fs=fs)
    out = _apply_full_array(sigx, lambda x: signal.filtfilt(b, a, x, axis=axis))
    out.name = (sigx.name + "_notch") if sigx.name else "notch_filtered"
    out.attrs.update({"filter_type": "notch", "w0_hz": float(w0), "Q": float(Q), "fs": fs})
    return out


def decimatex(
    sigx: xr.DataArray,
    *,
    factor: int = 2,
    time_dim: str = "time",
    ftype: str = "iir",
    zero_phase: bool = True,
) -> xr.DataArray:
    """Decimate along ``time_dim`` and update the time coordinate and fs.

    Notes
    -----
    - This changes the length of the time axis.
    - If ``sigx`` has a time coordinate, it is subsampled to match the output length.
    """
    fs = _fs_scalar(sigx)
    q = int(factor)
    if q <= 0:
        raise ValueError("factor must be >= 1")
    if q == 1:
        return sigx

    axis = sigx.get_axis_num(time_dim)
    y = signal.decimate(sigx.data, q, axis=axis, ftype=ftype, zero_phase=bool(zero_phase))

    # Build output coords/dims
    coords = dict(sigx.coords)
    dims = tuple(sigx.dims)
    attrs = dict(sigx.attrs)
    name = (sigx.name + "_decimate") if sigx.name else "decimated"

    # Update time coordinate if present
    if time_dim in coords:
        t = np.asarray(coords[time_dim].values)
        t_new = t[::q]
        # signal.decimate should produce len ~= ceil(n/q); clip to match
        new_len = y.shape[axis]
        t_new = t_new[:new_len]
        coords[time_dim] = t_new

    out = xr.DataArray(y, dims=dims, coords=coords, attrs=attrs, name=name)
    out_fs = fs / q
    out.attrs["fs"] = float(out_fs)
    out = out.assign_coords({"fs": float(out_fs)})
    return out


def gaussian_spatialx(
    sigx: xr.DataArray,
    *,
    sigma: float | tuple[float, float] = 1.0,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
    mode: str = "reflect",
) -> xr.DataArray:
    """Spatial Gaussian lowpass over (AP, ML), leaving other dims untouched."""
    if ap_dim not in sigx.dims or ml_dim not in sigx.dims:
        raise ValueError(f"Expected dims '{ap_dim}' and '{ml_dim}' in sigx.dims={tuple(sigx.dims)}")

    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigma_ap, sigma_ml = float(sigma[0]), float(sigma[1])
    else:
        sigma_ap = sigma_ml = float(sigma)

    sigma_by_axis = []
    for d in sigx.dims:
        if d == ap_dim:
            sigma_by_axis.append(sigma_ap)
        elif d == ml_dim:
            sigma_by_axis.append(sigma_ml)
        else:
            sigma_by_axis.append(0.0)

    out = _apply_full_array(sigx, nd.gaussian_filter, sigma=tuple(sigma_by_axis), mode=str(mode))
    out.name = (sigx.name + "_gauss_spatial") if sigx.name else "gaussian_spatial"
    out.attrs.update({"filter_type": "gaussian_spatial", "sigma": (sigma_ap, sigma_ml)})
    return out


def median_spatialx(
    sigx: xr.DataArray,
    *,
    size: int | tuple[int, int] = 3,
    ap_dim: str = "AP",
    ml_dim: str = "ML",
) -> xr.DataArray:
    """Spatial median lowpass over (AP, ML), leaving other dims untouched."""
    if ap_dim not in sigx.dims or ml_dim not in sigx.dims:
        raise ValueError(f"Expected dims '{ap_dim}' and '{ml_dim}' in sigx.dims={tuple(sigx.dims)}")

    if isinstance(size, (list, tuple, np.ndarray)):
        size_ap, size_ml = int(size[0]), int(size[1])
    else:
        size_ap = size_ml = int(size)

    size_by_axis = []
    for d in sigx.dims:
        if d == ap_dim:
            size_by_axis.append(size_ap)
        elif d == ml_dim:
            size_by_axis.append(size_ml)
        else:
            size_by_axis.append(1)

    out = _apply_full_array(sigx, nd.median_filter, size=tuple(size_by_axis))
    out.name = (sigx.name + "_median_spatial") if sigx.name else "median_spatial"
    out.attrs.update({"filter_type": "median_spatial", "size": (size_ap, size_ml)})
    return out


def median_subtractx(
    sigx: xr.DataArray,
    *,
    dims: tuple[str, ...] = ("AP", "ML"),
    skipna: bool = True,
) -> xr.DataArray:
    """Subtract the median across spatial dims (common average/median reference)."""
    for d in dims:
        if d not in sigx.dims:
            raise ValueError(f"median_subtractx expected dim {d!r} in sigx.dims={tuple(sigx.dims)}")
    axes = tuple(sigx.get_axis_num(d) for d in dims)

    def _medsub_full(x: np.ndarray) -> np.ndarray:
        if bool(skipna):
            med = np.nanmedian(x, axis=axes, keepdims=True)
        else:
            med = np.median(x, axis=axes, keepdims=True)
        return x - med

    out = _apply_full_array(sigx, _medsub_full, output_dtype=sigx.dtype)
    out.name = (sigx.name + "_medsub") if sigx.name else "median_subtract"
    out.attrs.update({"filter_type": "median_subtract", "dims": tuple(dims)})
    return out


def median_highpassx(
    sigx: xr.DataArray,
    *,
    size: int | tuple[int, int, int] = (7, 7, 101),
    time_dim: str = "time",
    ap_dim: str = "AP",
    ml_dim: str = "ML",
) -> xr.DataArray:
    """Highpass-like filter via subtraction of a spatiotemporal median.

    This mirrors :class:`TemporalHighpassMedian` in ``filt.py``:
    ``out = sig - median_filter(sig, size=...)``.

    Parameters
    ----------
    sigx
        Input signal DataArray.
    size
        Median filter window size. If an int, applies that spatial size to AP/ML
        and a default temporal size of 101 samples. If a 3-tuple, interpreted as
        ``(AP, ML, time)`` window sizes in samples.
    time_dim, ap_dim, ml_dim
        Dimension names.
    """
    if ap_dim not in sigx.dims or ml_dim not in sigx.dims or time_dim not in sigx.dims:
        raise ValueError(
            f"Expected dims '{time_dim}', '{ap_dim}', '{ml_dim}' in sigx.dims={tuple(sigx.dims)}"
        )

    if isinstance(size, (list, tuple, np.ndarray)):
        if len(size) != 3:
            raise ValueError("size must be an int or a 3-tuple (AP, ML, time).")
        size_ap, size_ml, size_t = int(size[0]), int(size[1]), int(size[2])
    else:
        size_ap = size_ml = int(size)
        size_t = 101

    if size_ap < 1 or size_ml < 1 or size_t < 1:
        raise ValueError("All median filter sizes must be >= 1.")

    size_by_axis = []
    for d in sigx.dims:
        if d == ap_dim:
            size_by_axis.append(size_ap)
        elif d == ml_dim:
            size_by_axis.append(size_ml)
        elif d == time_dim:
            size_by_axis.append(size_t)
        else:
            size_by_axis.append(1)

    med = _apply_full_array(sigx, nd.median_filter, size=tuple(size_by_axis))
    out = sigx - med
    out.attrs = dict(sigx.attrs)
    out.name = (sigx.name + "_medhp") if sigx.name else "median_highpass"
    out.attrs.update(
        {"filter_type": "median_highpass", "size": (size_ap, size_ml, size_t), "dims": (ap_dim, ml_dim, time_dim)}
    )
    return out


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
