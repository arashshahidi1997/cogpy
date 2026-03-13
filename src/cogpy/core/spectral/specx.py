"""
Spectral transforms with xarray interface.

This module wraps core spectral functions from ``cogpy.core.spectral``
(``psd.py``, ``multitaper.py``, ``bivariate.py``) to work seamlessly with
``xarray.DataArray``, preserving coordinates, dimensions, and metadata.

Convention
----------
- Input:  ``xr.DataArray`` with a time dimension
- Output: ``xr.DataArray`` with a ``freq`` dimension (or ``freq``×``time`` for spectrograms)
- Preserves all non-time dimensions and coordinates that do not depend on time
- Uses SciPy/NumPy/Ghostipy backends via ``cogpy.core.spectral``

Status
------
STATUS: ACTIVE
Reason: xarray interface for spectral analysis to match filtx.py pattern
Superseded by: n/a
Safe to remove: no
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from .bivariate import coherence as _coherence
from .psd import psd_multitaper, psd_welch

__all__ = [
    "psdx",
    "spectrogramx",
    "coherencex",
    "normalize_spectrogram",
]


def _fs_scalar(sigx: xr.DataArray) -> float:
    """Return a Python float sampling rate from ``sigx.fs``/attrs."""
    fs = getattr(sigx, "fs", None)
    if fs is None:
        fs = sigx.attrs.get("fs", None)
    if fs is None:
        raise ValueError("Input must have a `.fs` coordinate/attribute indicating sampling frequency.")
    if hasattr(fs, "item"):
        fs = fs.item()
    return float(fs)


def _coords_without_dim(sigx: xr.DataArray, *, dim: str) -> dict[str, object]:
    """Keep only coordinates that do not depend on ``dim``."""
    out: dict[str, object] = {}
    for name, coord in sigx.coords.items():
        try:
            coord_dims = coord.dims
        except Exception:  # noqa: BLE001
            coord_dims = ()
        if dim not in coord_dims:
            out[name] = coord
    return out


def psdx(
    sigx: xr.DataArray,
    *,
    axis: str = "time",
    method: Literal["multitaper", "welch"] = "multitaper",
    bandwidth: float = 4.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    fmin: float = 0.0,
    fmax: float | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Power spectral density with xarray interface.

    Wraps ``psd_welch`` or ``psd_multitaper`` from ``cogpy.core.spectral.psd``.

    Notes
    -----
    - ``method='welch'`` uses ``nperseg``/``noverlap`` as in SciPy.
    - ``method='multitaper'`` operates on the full window length ``N`` along ``axis``.
      ``bandwidth`` is interpreted in Hz and mapped to ``NW = bandwidth * N / fs``.
    """
    if not isinstance(sigx, xr.DataArray):
        raise TypeError("Expected xr.DataArray")
    axis = str(axis)
    if axis not in sigx.dims:
        raise ValueError(f"axis {axis!r} not in sigx.dims={tuple(sigx.dims)}")

    fs = _fs_scalar(sigx)
    time_axis = int(sigx.get_axis_num(axis))
    N = int(sigx.sizes[axis])
    non_time_dims = [d for d in sigx.dims if d != axis]

    arr = np.moveaxis(sigx.data, time_axis, -1)

    if method == "multitaper":
        NW = float(bandwidth) * float(N) / float(fs)
        psd_vals, freqs = psd_multitaper(
            arr,
            fs=fs,
            NW=NW,
            axis=-1,
            fmin=fmin,
            fmax=fmax,
            detrend=kwargs.pop("detrend", True),
            **kwargs,
        )
        meta = {"method": "multitaper", "bandwidth_hz": float(bandwidth), "NW": float(NW)}
    elif method == "welch":
        if noverlap is None:
            noverlap = int(nperseg) // 2
        psd_vals, freqs = psd_welch(
            arr,
            fs=fs,
            nperseg=int(nperseg),
            noverlap=int(noverlap),
            axis=-1,
            fmin=fmin,
            fmax=fmax,
            detrend=kwargs.pop("detrend", "constant"),
            **kwargs,
        )
        meta = {
            "method": "welch",
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
        }
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'multitaper' or 'welch'.")

    out_dims = [*non_time_dims, "freq"]

    out_coords = _coords_without_dim(sigx, dim=axis)
    out_coords["freq"] = np.asarray(freqs, dtype=np.float64)

    out_attrs = dict(sigx.attrs)
    out_attrs.update({"units": "power/Hz", "fs": float(fs), **meta})

    return xr.DataArray(
        np.asarray(psd_vals),
        dims=out_dims,
        coords=out_coords,
        attrs=out_attrs,
        name="psd",
    )


def spectrogramx(
    sigx: xr.DataArray,
    *,
    axis: str = "time",
    bandwidth: float = 4.0,
    nperseg: int = 256,
    noverlap: int | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Multitaper spectrogram (time-frequency representation).

    Wraps ``mtm_spectrogram`` from ``cogpy.core.spectral.multitaper`` (Ghostipy backend).
    """
    if not isinstance(sigx, xr.DataArray):
        raise TypeError("Expected xr.DataArray")
    axis = str(axis)
    if axis not in sigx.dims:
        raise ValueError(f"axis {axis!r} not in sigx.dims={tuple(sigx.dims)}")

    fs = _fs_scalar(sigx)
    time_axis = int(sigx.get_axis_num(axis))

    if noverlap is None:
        noverlap = int(nperseg) // 8

    try:
        from .multitaper import mtm_spectrogram
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "spectrogramx requires the optional 'ghostipy' dependency "
            "(install with `cogpy[signal]`)."
        ) from e

    mtspec, freqs, times = mtm_spectrogram(
        sigx.data,
        bandwidth=float(bandwidth),
        axis=time_axis,
        fs=float(fs),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        **kwargs,
    )

    out_dims = [d for d in sigx.dims if d != axis] + ["freq", "time"]
    out_coords = _coords_without_dim(sigx, dim=axis)
    out_coords["freq"] = np.asarray(freqs, dtype=np.float64)
    out_coords["time"] = np.asarray(times, dtype=np.float64)

    out_attrs = dict(sigx.attrs)
    out_attrs.update(
        {
            "method": "multitaper_spectrogram",
            "units": "power/Hz",
            "fs": float(fs),
            "bandwidth_hz": float(bandwidth),
            "nperseg": int(nperseg),
            "noverlap": int(noverlap),
        }
    )

    return xr.DataArray(
        mtspec,
        dims=out_dims,
        coords=out_coords,
        attrs=out_attrs,
        name="spectrogram",
    )


def coherencex(
    sigx: xr.DataArray,
    sigy: xr.DataArray,
    *,
    axis: str = "time",
    NW: float = 4.0,
    nfft: int | None = None,
    K_max: int | None = None,
    detrend: bool = True,
) -> xr.DataArray:
    """
    Multitaper magnitude squared coherence between two signals.

    Wraps ``multitaper_fft`` + ``cogpy.core.spectral.bivariate.coherence``.
    """
    if not isinstance(sigx, xr.DataArray) or not isinstance(sigy, xr.DataArray):
        raise TypeError("Both inputs must be xr.DataArray")
    axis = str(axis)

    if sigx.dims != sigy.dims:
        raise ValueError(
            f"sigx and sigy must have matching dims: {tuple(sigx.dims)} != {tuple(sigy.dims)}"
        )
    if axis not in sigx.dims:
        raise ValueError(f"axis {axis!r} not in dims={tuple(sigx.dims)}")

    fs = _fs_scalar(sigx)
    time_axis = int(sigx.get_axis_num(axis))
    N = int(sigx.sizes[axis])
    if nfft is None:
        nfft = N

    non_time_dims = [d for d in sigx.dims if d != axis]
    x_arr = np.moveaxis(sigx.data, time_axis, -1)
    y_arr = np.moveaxis(sigy.data, time_axis, -1)

    from .multitaper import multitaper_fft

    mtfft_x = multitaper_fft(
        x_arr, axis=-1, NW=float(NW), nfft=int(nfft), K_max=K_max, detrend=bool(detrend)
    )
    mtfft_y = multitaper_fft(
        y_arr, axis=-1, NW=float(NW), nfft=int(nfft), K_max=K_max, detrend=bool(detrend)
    )

    coh_vals = _coherence(mtfft_x, mtfft_y)
    freqs = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs)).astype(np.float64, copy=False)

    out_dims = [*non_time_dims, "freq"]
    out_coords = _coords_without_dim(sigx, dim=axis)
    out_coords["freq"] = freqs

    out_attrs = dict(sigx.attrs)
    out_attrs.update(
        {
            "method": "multitaper_coherence",
            "NW": float(NW),
            "K_max": int(K_max) if K_max is not None else int(2 * float(NW) - 1),
            "fs": float(fs),
        }
    )

    return xr.DataArray(
        np.asarray(coh_vals),
        dims=out_dims,
        coords=out_coords,
        attrs=out_attrs,
        name="coherence",
    )


def normalize_spectrogram(
    spec: xr.DataArray,
    *,
    method: str = "robust_zscore",
    dim: str = "freq",
) -> xr.DataArray:
    """
    Normalize a spectrogram along a dimension.

    Parameters
    ----------
    spec : xr.DataArray
        Input spectrogram (any dim order).
    method : {"robust_zscore", "db"}
        ``"robust_zscore"`` — ``(x - median) / (MAD * 1.4826)`` along *dim*
        via :func:`cogpy.core.preprocess.filtering.normalization.zscorex`.
        ``"db"`` — ``10 * log10(x)``, no dim reduction.
    dim : str
        Dimension to normalize along.  Typically ``"freq"``.

    Returns
    -------
    xr.DataArray
        Same shape/dims as input.  ``attrs`` updated with normalization
        metadata (``normalization``, ``normalization_dim``).
    """
    if not isinstance(spec, xr.DataArray):
        raise TypeError("Expected xr.DataArray")

    if method == "robust_zscore":
        from cogpy.core.preprocess.filtering.normalization import zscorex

        out = zscorex(spec, dim=dim, robust=True)
    elif method == "db":
        out = spec.copy(data=10.0 * np.log10(np.maximum(spec.values, np.finfo(float).tiny)))
        out.attrs = dict(spec.attrs)
        out.attrs["units"] = "dB"
    else:
        raise ValueError(
            f"Unknown normalization method: {method!r}. Use 'robust_zscore' or 'db'."
        )

    out.attrs["normalization"] = method
    out.attrs["normalization_dim"] = dim
    return out
