"""Separable N-D multitaper spectral estimation.

Constructs N-dimensional tapers as outer products of 1-D DPSS
(discrete prolate spheroidal sequences) and applies them to estimate
variance-reduced k-ω spectra.

References
----------
.. [1] Hanssen, "Multidimensional multitaper spectral estimation",
   Signal Processing, 1997. DOI: 10.1016/S0165-1684(97)00076-5
.. [2] Thomson, "Spectrum estimation and harmonic analysis",
   Proc. IEEE, 1982. DOI: 10.1109/PROC.1982.12433
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.signal.windows import dpss as dpss_1d

from ._types import Geometry

__all__ = ["dpss_nd", "multitaper_kw_spectrum"]


def dpss_nd(
    shape: tuple[int, ...],
    bw: tuple[float, ...],
) -> list[np.ndarray]:
    """Build separable N-D DPSS tapers.

    Each dimension gets its own set of 1-D DPSS sequences.  The N-D
    tapers are formed as outer products, one per combination of 1-D
    taper indices.

    Parameters
    ----------
    shape : tuple of int
        Array shape (e.g. ``(n_time, n_ap, n_ml)``).
    bw : tuple of float
        Half-bandwidth (NW) for each dimension.  The number of tapers
        per dimension is ``floor(2*NW - 1)``.

    Returns
    -------
    list of ndarray
        N-D taper arrays, each with the given *shape*.

    References
    ----------
    .. [1] Hanssen (1997), DOI: 10.1016/S0165-1684(97)00076-5
    .. [2] Thomson (1982), DOI: 10.1109/PROC.1982.12433
    """
    if len(shape) != len(bw):
        raise ValueError("shape and bw must have the same length")

    per_dim: list[np.ndarray] = []
    for n, nw in zip(shape, bw):
        k = max(1, int(2 * nw - 1))
        seqs = dpss_1d(n, nw, Kmax=k)  # (k, n)
        per_dim.append(seqs)

    # Outer product of all combinations.
    import itertools

    indices = [range(s.shape[0]) for s in per_dim]
    tapers = []
    for combo in itertools.product(*indices):
        taper = per_dim[0][combo[0]]
        for d in range(1, len(per_dim)):
            taper = np.multiply.outer(taper, per_dim[d][combo[d]])
        tapers.append(taper)

    return tapers


def multitaper_kw_spectrum(
    data: xr.DataArray,
    geometry: Geometry,
    bw_time: float,
    bw_space: float,
) -> xr.DataArray:
    """Variance-reduced k–ω spectrum via N-D multitaper.

    Parameters
    ----------
    data : DataArray
        Signal with dims ``(time, AP, ML)`` and ``fs`` coordinate.
    geometry : Geometry
        Regular grid geometry.
    bw_time : float
        Half-bandwidth (NW) for the time axis.
    bw_space : float
        Half-bandwidth (NW) for each spatial axis.

    Returns
    -------
    DataArray
        Power spectrum with dims ``(freq, kx, ky)``.

    References
    ----------
    .. [1] Hanssen (1997), DOI: 10.1016/S0165-1684(97)00076-5
    """
    if not geometry.is_regular:
        raise ValueError("multitaper_kw_spectrum requires a regular grid")

    fs = float(data.coords["fs"])
    vals = data.values  # (time, AP, ML)
    n_t, n_ap, n_ml = vals.shape

    tapers = dpss_nd((n_t, n_ap, n_ml), (bw_time, bw_space, bw_space))

    # Full FFT on all axes, then take positive temporal frequencies.
    n_freq = n_t // 2 + 1
    spectrum_sum = np.zeros((n_freq, n_ap, n_ml))

    for taper in tapers:
        tapered = vals * taper
        S = np.fft.fftn(tapered, axes=(0, 1, 2))
        # Keep only positive temporal frequencies (first axis).
        S_pos = S[:n_freq]
        spectrum_sum += np.abs(S_pos) ** 2

    spectrum = spectrum_sum / len(tapers)
    # Shift spatial axes to center zero-frequency.
    spectrum = np.fft.fftshift(spectrum, axes=(1, 2))

    freqs = np.fft.rfftfreq(n_t, 1.0 / fs)
    kx = np.fft.fftshift(np.fft.fftfreq(n_ap, geometry.dx))
    ky = np.fft.fftshift(np.fft.fftfreq(n_ml, geometry.dy))

    return xr.DataArray(
        spectrum,
        dims=("freq", "kx", "ky"),
        coords={"freq": freqs, "kx": kx, "ky": ky},
    )
