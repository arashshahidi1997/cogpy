"""Temporal IIR filters and decimation for xarray signals.

Butterworth bandpass/lowpass/highpass, notch, and polyphase decimation.
All functions accept ``xarray.DataArray`` with an ``fs`` coordinate.
"""

import numpy as np
import xarray as xr
from scipy import signal

from ...utils import xarr as xut
from ._utils import _apply_full_array, _fs_scalar


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
    """Butterworth bandpass filter with shoulder specification (zero-phase).

    Designs a Butterworth filter using buttord() to meet the passband and
    stopband specifications, then applies it with zero-phase filtering.

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

    See Also
    --------
    bandpassx : Butterworth bandpass with explicit order (simpler interface).
    notchx : IIR notch filter for line-noise removal.

    Examples
    --------
    >>> import cogpy
    >>> sigx = cogpy.datasets.load_sample()
    >>> filt = butterworth_bandpass_shoulder(sigx, fs=float(sigx.fs), low=4, high=50)
    >>> filt.shape == sigx.shape
    True
    """
    assert time_dim in x.dims, f"x must have a '{time_dim}' dimension"

    wp = [low, high]
    ws = [low - shoulder, high + shoulder]

    nyquist = fs / 2
    wp_norm = np.array(wp) / nyquist
    ws_norm = np.array(ws) / nyquist

    N, Wn = signal.buttord(wp=wp_norm, ws=ws_norm, gpass=rp, gstop=rs)
    sos = signal.butter(N, Wn, btype="bandpass", output="sos")

    time_axis = x.get_axis_num(time_dim)
    filtered_data = signal.sosfiltfilt(sos, x.values, axis=time_axis)

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
    """Butterworth bandpass filter (zero-phase, SOS form).

    For shoulder-controlled stopband specs use butterworth_bandpass_shoulder instead.

    See Also
    --------
    butterworth_bandpass_shoulder : Bandpass with shoulder-controlled stopband specs.
    notchx : IIR notch filter for line-noise removal.

    Examples
    --------
    >>> import cogpy
    >>> sigx = cogpy.datasets.load_sample()
    >>> filt = bandpassx(sigx, wl=4.0, wh=50.0, order=4, axis="time")
    >>> filt.shape == sigx.shape
    True
    """
    fs = _fs_scalar(sigx)
    wh = min(wh, fs / 2 - 0.1)
    sos = signal.butter(order, [wl, wh], btype="bandpass", fs=fs, output="sos")
    axis_num = sigx.get_axis_num(axis)
    out = _apply_full_array(sigx, lambda x: signal.sosfiltfilt(sos, x, axis=axis_num))
    out.name = "Bandpass ({}-{} Hz)".format(wl, wh)
    return out


def lowpassx(sigx: xr.DataArray, wl: float, order: int, axis: str) -> xr.DataArray:
    """Butterworth lowpass filter (zero-phase)."""
    b, a = signal.butter(order, wl / (0.5 * _fs_scalar(sigx)), btype="low")
    axis = sigx.get_axis_num(axis)
    bp_filt = lambda x: signal.filtfilt(b, a, x, axis=axis)
    sigx_bp = xut.xarr_wrap(bp_filt)(sigx)
    sigx_bp.name = "Lowpass ({} Hz)".format(wl)
    return sigx_bp


def highpassx(sigx: xr.DataArray, wh: float, order: int, axis: str) -> xr.DataArray:
    """Butterworth highpass filter (zero-phase)."""
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
    """Notch filter (IIR notch) applied along ``time_dim`` using zero-phase filtering.

    See Also
    --------
    bandpassx : Butterworth bandpass filter.
    butterworth_bandpass_shoulder : Bandpass with shoulder-controlled stopband specs.
    """
    fs = _fs_scalar(sigx)
    axis = sigx.get_axis_num(time_dim)
    b, a = signal.iirnotch(float(w0), float(Q), fs=fs)
    out = _apply_full_array(sigx, lambda x: signal.filtfilt(b, a, x, axis=axis))
    out.name = (sigx.name + "_notch") if sigx.name else "notch_filtered"
    out.attrs.update(
        {"filter_type": "notch", "w0_hz": float(w0), "Q": float(Q), "fs": fs}
    )
    return out


def notchesx(
    sigx: xr.DataArray,
    *,
    freqs: list[float] | tuple[float, ...] | np.ndarray,
    Q: float = 30.0,
    time_dim: str = "time",
) -> xr.DataArray:
    """Apply multiple IIR notches along ``time_dim`` using zero-phase filtering.

    This is equivalent to sequentially calling ``notchx`` for each frequency,
    but applies all notches within a single xarray apply (reduces wrapper overhead).
    """
    fs = _fs_scalar(sigx)
    axis = sigx.get_axis_num(time_dim)

    freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
    if freqs_arr.size == 0:
        return sigx

    nyq = float(fs) / 2.0
    if np.any(freqs_arr <= 0) or np.any(freqs_arr >= nyq):
        raise ValueError(
            f"All notch frequencies must be in (0, fs/2). Got freqs={freqs_arr.tolist()}, fs={fs}"
        )

    bas = [
        signal.iirnotch(float(f), float(Q), fs=float(fs)) for f in freqs_arr.tolist()
    ]

    def _multi_notch(x: np.ndarray) -> np.ndarray:
        y = x
        for b, a in bas:
            y = signal.filtfilt(b, a, y, axis=axis)
        return y

    out = _apply_full_array(sigx, _multi_notch)
    out.name = (sigx.name + "_notch") if sigx.name else "notch_filtered"
    out.attrs.update(
        {
            "filter_type": "notch",
            "w0_hz": freqs_arr.tolist(),
            "Q": float(Q),
            "fs": float(fs),
        }
    )
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
    y = signal.decimate(
        sigx.data, q, axis=axis, ftype=ftype, zero_phase=bool(zero_phase)
    )

    coords = dict(sigx.coords)
    dims = tuple(sigx.dims)
    attrs = dict(sigx.attrs)
    name = (sigx.name + "_decimate") if sigx.name else "decimated"

    if time_dim in coords:
        t = np.asarray(coords[time_dim].values)
        t_new = t[::q]
        new_len = y.shape[axis]
        t_new = t_new[:new_len]
        coords[time_dim] = t_new

    out = xr.DataArray(y, dims=dims, coords=coords, attrs=attrs, name=name)
    out_fs = fs / q
    out.attrs["fs"] = float(out_fs)
    out = out.assign_coords({"fs": float(out_fs)})
    return out
